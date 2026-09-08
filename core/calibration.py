"""
core.calibration
================

Concentration → IFT calibration, loaded from an external data table.

A gas–liquid system is defined entirely by data:

    1. a CSV of digitized/measured equilibrium points
       (columns: c_eq, gamma_eq — real concentration units), and
    2. a small config entry (``config.py``) giving gamma0, gamma_inf
       and the saturation concentration c_sat for the run.

Switching systems = new CSV + new config entry, zero code change.
There is deliberately only ONE interpolation implementation — the
*data* is what is swappable, not the code.

Units bridge (the critical gotcha)
----------------------------------
The simulation's interface concentration ``Cs`` is dimensionless
(0 → 1, fraction of saturation, because the FEM interface BC uses
c_eq = 1).  The calibration table is in real concentration units.
So every lookup first converts::

    c_real = Cs * c_sat

where ``c_sat = c_eq(P)`` is the saturation concentration at the run's
pressure — system- and pressure-specific, hence config, not code.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Where the one canonical, currently-active calibration lives.  There is
# exactly ONE such file; every upload atomically replaces it (os.replace),
# so there is never a half-written file and never a pile of old uploads.
# ----------------------------------------------------------------------
ACTIVE_CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent / 'data' / 'active_calibration.csv')

# Guardrails against pathological uploads.  A calibration curve is an
# equilibrium relationship — inherently a handful to a few hundred points;
# anything larger is a mistake or a hostile file, and we refuse it before
# doing any parsing work.
MAX_UPLOAD_BYTES = 1_000_000      # 1 MB
MAX_DATA_ROWS    = 10_000
# Override vs endpoint disagreeing by more than this (relative) is almost
# always a unit slip (e.g. mol/L typed where the column is mol/m³, ~1000x);
# non-blocking — we warn but honour the user's explicit value.
CSAT_MISMATCH_RTOL = 0.01


class CalibrationError(Exception):
    """Raised when an uploaded calibration file fails validation.

    Carries a user-facing message precise enough to fix the file
    (which row, what was wrong).  The caller shows it verbatim and
    leaves any previously-loaded calibration untouched.
    """


def parse_calibration_template(path):
    """Parse + fully validate an uploaded calibration template.

    Returns a dict ``{'c', 'gamma', 'c_sat', 'unit', 'warnings'}`` on
    success, or raises :class:`CalibrationError` with a specific,
    fixable message.  Performs NO writes and NO global state changes —
    this is the "candidate" stage of the all-or-nothing commit.

    Accepted layout (the downloadable template)::

        c_sat,                      # blank -> use endpoint; or a value
        concentration_unit,mol/m3
        concentration,ift
        0,70.55
        ...

    Metadata rows may appear in any order; blank lines are ignored.
    Only the two data columns are ever interpreted as numbers, so a
    spreadsheet formula string (``=...``) simply fails the float parse
    and is rejected — it is never executed.
    """
    path = Path(path)

    # --- vulnerability guard: size, before reading a single byte ------
    try:
        size = path.stat().st_size
    except OSError as e:
        raise CalibrationError(f"cannot open file: {e}") from e
    if size > MAX_UPLOAD_BYTES:
        raise CalibrationError(
            f"file is {size/1e6:.1f} MB — a calibration curve should be "
            f"tiny (a few KB). Refusing to load; is this the right file?")

    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
    except UnicodeDecodeError as e:
        raise CalibrationError(
            "file isn't UTF-8 text — is it really a CSV? Re-export the "
            "template as CSV and try again.") from e
    except OSError as e:
        raise CalibrationError(f"cannot read file: {e}") from e

    if len(rows) > MAX_DATA_ROWS:
        raise CalibrationError(
            f"file has {len(rows)} rows — far more than a calibration "
            f"curve should. Refusing to load.")

    c_sat_raw = None
    unit = None
    header_seen = False
    c_vals, g_vals = [], []
    warnings = []

    for lineno, row in enumerate(rows, start=1):
        if not row or all(cell.strip() == '' for cell in row):
            continue                       # blank line
        key = row[0].strip().lower()

        if key == 'c_sat':
            c_sat_raw = row[1].strip() if len(row) > 1 else ''
            continue
        if key in ('concentration_unit', 'unit'):
            unit = row[1].strip() if len(row) > 1 else ''
            continue
        if key in ('concentration', 'c', 'c_eq', 'conc'):
            header_seen = True             # column header row
            continue

        # Anything else is a data row.  It must be two finite numbers.
        if not header_seen:
            raise CalibrationError(
                f"line {lineno}: data before the 'concentration,ift' "
                f"header row — is this the template? First cell: {row[0]!r}")
        if len(row) < 2 or row[0].strip() == '' or row[1].strip() == '':
            raise CalibrationError(
                f"line {lineno}: expected two values 'concentration,ift', "
                f"got {row!r}")
        try:
            c = float(row[0])
            g = float(row[1])
        except ValueError as e:
            raise CalibrationError(
                f"line {lineno}: not a number — got "
                f"concentration={row[0]!r}, ift={row[1]!r}") from e
        if not (np.isfinite(c) and np.isfinite(g)):
            raise CalibrationError(
                f"line {lineno}: NaN/Inf is not a valid data point")
        c_vals.append(c)
        g_vals.append(g)

    # --- structural validation ---------------------------------------
    if not header_seen:
        raise CalibrationError(
            "no 'concentration,ift' header row found — this doesn't look "
            "like the template. Download a fresh template and fill that.")
    if len(c_vals) < 2:
        raise CalibrationError(
            f"need at least 2 data points, found {len(c_vals)}.")

    c = np.asarray(c_vals, dtype=float)
    g = np.asarray(g_vals, dtype=float)

    # Sort ascending by concentration (fix, don't fail — but tell them).
    if np.any(np.diff(c) < 0):
        warnings.append("concentration column wasn't sorted — sorted "
                        "ascending automatically.")
        order = np.argsort(c)
        c, g = c[order], g[order]
    # Duplicate concentrations make np.interp ambiguous — reject.
    if np.any(np.diff(c) == 0):
        dup = c[:-1][np.diff(c) == 0][0]
        raise CalibrationError(
            f"duplicate concentration {dup:g} — each point needs a "
            f"distinct concentration.")

    # --- resolve c_sat: blank -> endpoint, else honour + sanity-check -
    endpoint = float(c[-1])                # largest concentration = saturation
    if c_sat_raw in (None, ''):
        if endpoint <= 0:
            raise CalibrationError(
                "c_sat is blank so the last row would be used as "
                "saturation, but the last concentration is not positive.")
        c_sat = endpoint
    else:
        try:
            c_sat = float(c_sat_raw)
        except ValueError as e:
            raise CalibrationError(
                f"c_sat must be a number or blank, got {c_sat_raw!r}") from e
        if c_sat <= 0:
            raise CalibrationError(f"c_sat must be positive, got {c_sat}")
        if endpoint > 0 and abs(c_sat - endpoint) / endpoint > CSAT_MISMATCH_RTOL:
            warnings.append(
                f"c_sat = {c_sat:g} differs from the last concentration "
                f"({endpoint:g}). If that's a unit mismatch, fix it; "
                f"otherwise this override is used as-is.")

    return dict(c=c, gamma=g, c_sat=c_sat, unit=unit, warnings=warnings)


def _serialize_template(parsed):
    """Render a validated/normalized parse back into template CSV text
    (sorted rows, resolved c_sat filled in) for the canonical file."""
    lines = [f"c_sat,{parsed['c_sat']:g}",
             f"concentration_unit,{parsed['unit'] or ''}",
             "concentration,ift"]
    lines += [f"{c:g},{g:g}" for c, g in zip(parsed['c'], parsed['gamma'])]
    return "\n".join(lines) + "\n"


def _atomic_write(text, dest):
    """Write ``text`` to ``dest`` atomically: temp file in the same
    directory, then ``os.replace`` (atomic on Windows and POSIX).  A
    reader never sees a partial file, and a crash mid-write leaves the
    previous canonical file intact."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def install_calibration(src_path, gamma0=None, gamma_inf=None,
                        canonical_path=ACTIVE_CALIBRATION_PATH):
    """Validate an uploaded file and, only if it fully passes, commit it.

    All-or-nothing (ACID): the file is parsed into a candidate and every
    check must pass *before* anything is written.  Only then is the
    normalized curve written to the single canonical file (atomically)
    and a ready-to-use :class:`Calibration` returned.  If validation
    fails, ``CalibrationError`` is raised, nothing is written, and any
    previously-active calibration is untouched.

    ``gamma0`` / ``gamma_inf`` default to the curve endpoints (clean and
    saturated IFT), which is what the reference constants are for.

    Returns ``(calibration, warnings)``.
    """
    parsed = parse_calibration_template(src_path)      # raises on any problem

    _atomic_write(_serialize_template(parsed), canonical_path)

    g0 = float(parsed['gamma'][0])  if gamma0   is None else gamma0
    gi = float(parsed['gamma'][-1]) if gamma_inf is None else gamma_inf
    calib = Calibration.from_arrays(
        parsed['c'], parsed['gamma'], gamma0=g0, gamma_inf=gi,
        c_sat=parsed['c_sat'], source=canonical_path)
    return calib, parsed['warnings']


def clear_active_calibration(canonical_path=ACTIVE_CALIBRATION_PATH):
    """Delete the canonical active-calibration file if present.

    Pairs with the in-memory reset in the GUI so that "reset" leaves no
    stale curve behind — neither in memory nor on disk.  Missing file is
    a no-op (already cleared).  Returns True if a file was removed.
    """
    try:
        os.remove(canonical_path)
        return True
    except FileNotFoundError:
        return False


class Calibration:
    """γ(c) interpolation table plus the units bridge; the whole
    gas–liquid system lives in the data passed to the constructor."""

    def __init__(self, csv_path, gamma0, gamma_inf, c_sat):
        """
        Parameters
        ----------
        csv_path : str or Path
            CSV with ``c_eq`` in column 0 and ``gamma_eq`` in column 1
            (comment lines start with '#').  Any further columns are
            metadata and ignored here.
        gamma0 : float
            Clean-surface IFT (mN/m) — for reference/reporting.
        gamma_inf : float
            Fully saturated IFT (mN/m) — for reference/reporting.
        c_sat : float
            Saturation concentration at the run's pressure, in the
            same units as the CSV's c_eq column.
        """
        data = np.loadtxt(csv_path, delimiter=',', comments='#',
                          usecols=(0, 1), ndmin=2)
        order = np.argsort(data[:, 0])
        self.c_points = data[order, 0]
        self.gamma_points = data[order, 1]

        self.csv_path = str(csv_path)
        self.gamma0 = float(gamma0)
        self.gamma_inf = float(gamma_inf)
        self.c_sat = float(c_sat)

        if len(self.c_points) < 2:
            raise ValueError(
                f"calibration table {csv_path} needs >= 2 data points")
        if self.c_sat <= 0:
            raise ValueError(f"c_sat must be positive, got {c_sat}")

    @classmethod
    def from_arrays(cls, c_points, gamma_points, gamma0, gamma_inf, c_sat,
                    source='<uploaded>'):
        """Build directly from validated arrays (no file read).

        Used by :func:`install_calibration` after the uploaded file has
        already been parsed and checked, so this constructor only re-does
        the two invariants the rest of the code relies on.
        """
        obj = cls.__new__(cls)
        c = np.asarray(c_points, dtype=float)
        g = np.asarray(gamma_points, dtype=float)
        order = np.argsort(c)
        obj.c_points = c[order]
        obj.gamma_points = g[order]
        obj.csv_path = str(source)
        obj.gamma0 = float(gamma0)
        obj.gamma_inf = float(gamma_inf)
        obj.c_sat = float(c_sat)
        if len(obj.c_points) < 2:
            raise ValueError("calibration needs >= 2 data points")
        if obj.c_sat <= 0:
            raise ValueError(f"c_sat must be positive, got {c_sat}")
        return obj

    def gamma(self, Cs):
        """Dimensionless interface concentration Cs (0..1) → IFT (mN/m).

        Applies the units bridge, then interpolates the table.
        ``np.interp`` clamps at both table ends — it never extrapolates,
        which is the intended behaviour.
        """
        c_real = np.asarray(Cs, dtype=float) * self.c_sat
        return np.interp(c_real, self.c_points, self.gamma_points)

    # Name used in some project docs; same method.
    gamma_of_Cs = gamma

    def __repr__(self):
        return (f"Calibration({self.csv_path!r}, "
                f"gamma0={self.gamma0}, gamma_inf={self.gamma_inf}, "
                f"c_sat={self.c_sat})")


def c_sat_for_pressure(csv_path, pressure_MPa, atol=0.01):
    """Saturation concentration c_eq(P) for a run at ``pressure_MPa``,
    read from column 2 of a 3-column calibration CSV
    (c_eq, gamma_eq, pressure).

    The pressure must match a table row (within ``atol`` MPa) — the
    calibration rows ARE the equilibrium states at the test pressures,
    so an in-between pressure has no digitized c_sat to offer.
    """
    data = np.loadtxt(csv_path, delimiter=',', comments='#',
                      usecols=(0, 2), ndmin=2)
    match = np.abs(data[:, 1] - pressure_MPa) <= atol
    if not np.any(match):
        available = ", ".join(f"{p:g}" for p in data[:, 1])
        raise ValueError(
            f"no calibration row at P = {pressure_MPa} MPa "
            f"(available: {available} MPa)")
    return float(data[match, 0][0])
