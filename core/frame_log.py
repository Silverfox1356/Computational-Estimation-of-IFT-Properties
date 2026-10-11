"""
core.frame_log
==============

The per-frame log: one row per logged frame, kept as ``log.csv`` inside a
session folder (see :mod:`core.session`).

Columns
-------
Always logged: ``time_s``, ``frame_index``, ``fit_ok``, ``qc_pass``,
``error`` (plus ``wall_clock`` for camera sessions).

The other columns are optional and chosen by the user.  Fitting a frame
produces all of them at once, so leaving one out saves almost no time; the
choice shapes the file and what is shown.  An enabled analysis locks the
columns it reads (:data:`REQUIRED_BY`), and a column left out can be
rebuilt later from the stored frames (:meth:`core.session.SessionFolder.rebuild_log`).

Column names match :data:`core.frame_series.TABLE_COLUMNS` where the two
overlap, so a log reads with :func:`core.frame_series.load_frame_table_rows`.

Nothing here touches Qt.
"""

import csv
import io
import math
from collections import namedtuple

import numpy as np

Column = namedtuple('Column', 'key label unit description')

ALWAYS = (
    Column('time_s', 'Time', 's', 'seconds after the drop formed'),
    Column('frame_index', 'Frame', '', 'frame number in the source'),
    Column('fit_ok', 'Fit OK', '', '1 if the Young-Laplace fit converged'),
    Column('qc_pass', 'QC pass', '', '1 if every quality check passed'),
    Column('error', 'Error', '', 'why the fit failed, if it did'),
)
WALL_CLOCK = Column('wall_clock', 'Clock time', '', 'local time the frame was grabbed (camera)')

OPTIONAL = (
    Column('ift_mNm', 'IFT', 'mN/m', 'interfacial tension'),
    Column('ift_unc_mNm', 'IFT uncertainty', 'mN/m', 'total IFT uncertainty'),
    Column('beta', 'Shape factor beta', '', 'Bashforth-Adams shape factor'),
    Column('beta_unc', 'beta uncertainty', '', 'uncertainty of beta'),
    Column('R0_mm', 'Apex radius R0', 'mm', 'radius of curvature at the apex'),
    Column('volume_uL', 'Volume', 'uL', 'drop volume below the baseline'),
    Column('area_mm2', 'Surface area', 'mm2', 'drop surface area below the baseline'),
    Column('Wo', 'Worthington number', '', 'drop size relative to the largest that can hang'),
    Column('ds_de', 'Ds/De', '', 'ratio of the diameter at height De to De'),
    Column('ift_dsde_mNm', 'IFT (Ds/De method)', 'mN/m', 'IFT from the classical selected-plane method'),
    Column('rmse_px', 'Fit residual', 'px', 'RMS distance between contour and fit'),
    Column('tilt', 'Axis tilt', '', 'slope of the drop axis (dx/dy)'),
    Column('axis_shift_px', 'Axis shift', 'px', 'RMS sideways offset of the drop axis'),
)

COLUMNS = {c.key: c for c in ALWAYS + (WALL_CLOCK,) + OPTIONAL}
OPTIONAL_KEYS = tuple(c.key for c in OPTIONAL)
TEXT_COLUMNS = ('error', 'wall_clock')
BOOL_COLUMNS = ('fit_ok', 'qc_pass')

# Columns each analysis reads.  Enabling an analysis ticks and locks them.
REQUIRED_BY = {
    'swelling': ('volume_uL',),
    'bagalkot': ('volume_uL',),
    'yang': ('ift_mNm',),
}

MAX_ROWS = 200_000          # a 55-hour camera run at one frame per second


class FrameLogError(ValueError):
    """A log file that can't be read or a column that doesn't exist."""


def required_columns(analyses):
    """Optional columns that the given analyses need, in canonical order."""
    need = set()
    for a in analyses:
        if a not in REQUIRED_BY:
            raise FrameLogError(f"unknown analysis {a!r}")
        need.update(REQUIRED_BY[a])
    return [k for k in OPTIONAL_KEYS if k in need]


def column_keys(selected=OPTIONAL_KEYS, analyses=(), camera=False):
    """Full ordered header: the always-logged columns, then the chosen
    optional ones plus those the analyses need (canonical order)."""
    selected = list(selected)
    bad = [k for k in selected if k not in OPTIONAL_KEYS]
    if bad:
        raise FrameLogError(f"unknown column(s): {', '.join(map(str, bad))}")
    want = set(selected) | set(required_columns(analyses))
    keys = [c.key for c in ALWAYS]
    if camera:
        keys.insert(2, WALL_CLOCK.key)
    return keys + [k for k in OPTIONAL_KEYS if k in want]


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def record_to_row(rec, keys):
    """One log row from a :func:`core.frame_series.analyse_frame` record
    (with 't_s' and 'frame_index' set, and optionally 'wall_clock')."""
    res = rec.get('results') or {}
    post = res.get('post_asym') or {}
    values = dict(
        time_s=_num(rec.get('t_s')),
        frame_index=int(rec.get('frame_index', -1)),
        wall_clock=str(rec.get('wall_clock', '') or ''),
        fit_ok=bool(rec.get('fit_ok')),
        qc_pass=bool(rec.get('qc_pass')),
        error=str(rec.get('error', '') or ''),
        ift_mNm=_num(rec.get('ift_mNm')),
        ift_unc_mNm=_num(rec.get('ift_unc_mNm')),
        beta=_num(rec.get('beta')),
        beta_unc=_num(res.get('beta_uncertainty')),
        R0_mm=_num(rec.get('R0_mm')),
        volume_uL=_num(rec.get('volume_uL')),
        area_mm2=_num(rec.get('area_mm2')),
        Wo=_num(res.get('Wo')),
        ds_de=_num(res.get('ds_de_ratio')),
        ift_dsde_mNm=_num(res.get('ift_dsde_mNm')),
        rmse_px=_num(res.get('rmse_px')),
        tilt=_num(post.get('tilt_slope')),
        axis_shift_px=_num(post.get('trans_rms_px')),
    )
    return {k: values[k] for k in keys}


def format_cell(key, v):
    if key in BOOL_COLUMNS:
        return '1' if v else '0'
    if key in TEXT_COLUMNS:
        return str(v).replace('\r', ' ').replace('\n', ' ')
    if key == 'frame_index':
        return str(int(v))
    if key == 'time_s':
        return f"{v:.4f}"
    return '' if v is None or not math.isfinite(v) else f"{v:.6g}"


def format_row(keys, row):
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(
        [format_cell(k, row[k]) for k in keys])
    return buf.getvalue()


def format_header(keys, header_lines=()):
    head = ''.join(f"# {line}\n" for line in header_lines)
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(keys)
    return head + buf.getvalue()


class FrameLog:
    """Rows of the log in memory.  ``columns`` is the full header."""

    def __init__(self, columns):
        unknown = [k for k in columns if k not in COLUMNS]
        if unknown:
            raise FrameLogError(f"unknown column(s): {', '.join(unknown)}")
        missing = [c.key for c in ALWAYS if c.key not in columns]
        if missing:
            raise FrameLogError(f"the log needs column(s): {', '.join(missing)}")
        self.columns = list(columns)
        self.rows = []
        self.warnings = []

    def __len__(self):
        return len(self.rows)

    def has(self, key):
        return key in self.columns

    def append_record(self, rec):
        row = record_to_row(rec, self.columns)
        self.rows.append(row)
        return row

    def column(self, key):
        """A numeric column as a float array (NaN where missing)."""
        if key not in self.columns:
            raise FrameLogError(f"column {key!r} was not logged")
        if key in TEXT_COLUMNS:
            raise FrameLogError(f"column {key!r} is text")
        return np.array([float(r[key]) for r in self.rows], dtype=float)

    def to_text(self, header_lines=()):
        return format_header(self.columns, header_lines) + ''.join(
            format_row(self.columns, r) for r in self.rows)

    @classmethod
    def from_text(cls, text):
        lines = [ln for ln in text.splitlines() if not ln.startswith('#')]
        reader = csv.reader(lines)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise FrameLogError("the log has no header row") from None
        known = [h for h in header if h in COLUMNS]
        log = cls(known)
        extra = [h for h in header if h not in COLUMNS]
        if extra:
            log.warnings.append(f"ignored unknown column(s): {', '.join(extra)}")
        idx = {h: i for i, h in enumerate(header)}
        for lineno, cells in enumerate(reader, start=2):
            if not any(c.strip() for c in cells):
                continue
            if len(log.rows) >= MAX_ROWS:
                raise FrameLogError(f"the log has more than {MAX_ROWS} rows")
            row = {}
            for k in known:
                c = cells[idx[k]].strip() if idx[k] < len(cells) else ''
                if k in BOOL_COLUMNS:
                    row[k] = c == '1'
                elif k in TEXT_COLUMNS:
                    row[k] = c
                elif k == 'frame_index':
                    try:
                        row[k] = int(float(c))
                    except ValueError:
                        raise FrameLogError(
                            f"row {lineno}: frame_index {c!r} is not a number") from None
                else:
                    row[k] = _num(c) if c else math.nan
            if not math.isfinite(row['time_s']):
                raw = cells[idx['time_s']] if idx['time_s'] < len(cells) else ''
                raise FrameLogError(f"row {lineno}: time_s {raw!r} is not a number")
            log.rows.append(row)
        return log
