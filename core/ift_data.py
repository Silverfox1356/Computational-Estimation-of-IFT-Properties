"""
core.ift_data
=============

Load a measured dynamic-IFT curve — interfacial tension vs time — from a
CSV file, so Step 8 (estimate D, k) can run on data that did not come out
of this app's own camera/video analysis: a curve exported from another
tensiometer, or digitised from a paper.

Accepted layout::

    # optional comment lines
    time_s,ift_mNm          <- optional header; names pick the columns
    19.5,65.803
    51.9,64.398
    ...

Without a header the first two columns are (time in s, IFT in mN/m).  With
one, a column named like ``time``/``t`` and one like ``ift``/``gamma``/
``...tension`` are used, in whichever order they appear; extra columns are
ignored.

Guardrails mirror the calibration upload (``core.calibration``): size and
row caps before parsing, UTF-8 only, every data cell goes through float()
so a spreadsheet formula is rejected — never executed — and NaN/Inf,
negative times, non-positive IFT and duplicate times are refused with the
offending line number.  Nothing here touches global state.
"""

import csv
from pathlib import Path

import numpy as np

MAX_FILE_BYTES = 5_000_000        # a dynamic-IFT curve is a few KB
MAX_ROWS = 200_000
MIN_POINTS = 2                    # the fit's own minimum is enforced in Step 8


class IFTDataError(Exception):
    """An IFT-curve file failed validation.  The message says which line
    and what to fix; callers show it verbatim and keep their current data."""


def _is_number(cell):
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _is_time(cell):
    c = cell.strip().lower()
    return c in ('t', 't_s', 't (s)', 't[s]') or c.startswith('time')


def _is_ift(cell):
    c = cell.strip().lower()
    return c.startswith(('ift', 'gamma', 'γ')) or 'tension' in c


def _ift_nonpositive(lineno, g):
    return (f"line {lineno}: IFT must be > 0 mN/m, got {g:g}. This is "
            f"usually a failed drop-shape fit; the fit divides by IFT, so "
            f"remove the point.")


def _ift_unit_warnings(g):
    median = float(np.median(g))
    if median < 1.0:
        return [f"median IFT is {median:.3g} — that looks like N/m; "
                f"this loader expects mN/m (multiply by 1000)."]
    if median > 200.0:
        return [f"median IFT is {median:.3g} mN/m — unusually high; "
                f"check the units (1 dyn/cm = 1 mN/m)."]
    return []


def load_ift_curve(path):
    """Parse and validate an IFT-vs-time CSV.

    Returns ``{'t': ndarray [s], 'gamma': ndarray [mN/m], 'warnings': [str]}``
    sorted by time, or raises :class:`IFTDataError`.
    """
    d = load_xy_curve(path, _is_ift, "an IFT", "IFT (mN/m)",
                      _ift_nonpositive, _ift_unit_warnings)
    return dict(t=d['t'], gamma=d['y'], warnings=d['warnings'])


def load_xy_curve(path, is_value_header, value_desc, value_label,
                  nonpositive_msg, unit_warnings, error_cls=None,
                  skip_blank_values=False):
    """Shared parser for two-column (time, value) CSV files.

    ``is_value_header(cell)`` recognises the value column's header,
    ``value_desc``/``value_label`` name it in messages, ``nonpositive_msg``
    builds the error for a value <= 0 and ``unit_warnings(y)`` returns any
    unit warnings.  ``skip_blank_values`` skips rows whose value cell is
    empty (with one warning) instead of rejecting the file.  Returns
    ``{'t', 'y', 'warnings'}`` sorted by time.
    """
    err = error_cls or IFTDataError
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        raise err(f"cannot open file: {e}") from e
    if size > MAX_FILE_BYTES:
        raise err(
            f"file is {size / 1e6:.1f} MB — a curve file should be a few KB. "
            f"Refusing to load; is this the right file?")
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
    except UnicodeDecodeError as e:
        raise err(
            "file isn't UTF-8 text — is it really a CSV? Re-export it as "
            "CSV and try again.") from e
    except OSError as e:
        raise err(f"cannot read file: {e}") from e
    if len(rows) > MAX_ROWS:
        raise err(
            f"file has {len(rows)} rows — far more than a curve file should. "
            f"Refusing to load.")

    i_t, i_g = 0, 1
    header_seen = False
    warnings = []
    t_vals, g_vals, line_nos = [], [], []
    n_blank = 0

    for lineno, row in enumerate(rows, start=1):
        cells = [c.strip() for c in row]
        if not cells or all(c == '' for c in cells) or cells[0].startswith('#'):
            continue                          # blank or comment line

        if not header_seen and not t_vals and not _is_number(cells[0]):
            header_seen = True                # first non-numeric row: header
            it = next((i for i, c in enumerate(cells) if _is_time(c)), None)
            ig = next((i for i, c in enumerate(cells) if is_value_header(c)), None)
            if it is not None and ig is not None and it != ig:
                i_t, i_g = it, ig
            else:
                warnings.append(
                    f"header {row!r} doesn't name a time and {value_desc} column — "
                    f"using the first two columns as time (s), {value_label}.")
            continue

        if (skip_blank_values and len(cells) > max(i_t, i_g)
                and cells[i_t] != '' and cells[i_g] == ''):
            n_blank += 1
            continue
        if len(cells) <= max(i_t, i_g) or cells[i_t] == '' or cells[i_g] == '':
            raise err(
                f"line {lineno}: expected a time and {value_desc} value, got {row!r}")
        try:
            t = float(cells[i_t])
            g = float(cells[i_g])
        except ValueError as e:
            raise err(
                f"line {lineno}: not a number — got time={cells[i_t]!r}, "
                f"value={cells[i_g]!r}") from e
        if not (np.isfinite(t) and np.isfinite(g)):
            raise err(f"line {lineno}: NaN/Inf is not a valid data point")
        if t < 0:
            raise err(f"line {lineno}: negative time {t:g} s")
        if g <= 0:
            raise err(nonpositive_msg(lineno, g))
        t_vals.append(t)
        g_vals.append(g)
        line_nos.append(lineno)

    if n_blank:
        warnings.append(f"{n_blank} row(s) without {value_desc} value were "
                        f"skipped (failed frames).")
    if len(t_vals) < MIN_POINTS:
        raise err(
            f"need at least {MIN_POINTS} data points, found {len(t_vals)}.")

    t = np.asarray(t_vals, dtype=float)
    g = np.asarray(g_vals, dtype=float)
    ln = np.asarray(line_nos)

    if np.any(np.diff(t) < 0):
        warnings.append("time column wasn't sorted — sorted ascending "
                        "automatically.")
        order = np.argsort(t, kind='stable')
        t, g, ln = t[order], g[order], ln[order]
    dup = np.nonzero(np.diff(t) == 0)[0]
    if dup.size:
        i = int(dup[0])
        raise err(
            f"lines {ln[i]} and {ln[i + 1]}: duplicate time {t[i]:g} s — each "
            f"measurement needs its own time. A file holding several series "
            f"(e.g. one per pressure) must be split into one file per series.")

    warnings.extend(unit_warnings(g))

    return dict(t=t, y=g, warnings=warnings)
