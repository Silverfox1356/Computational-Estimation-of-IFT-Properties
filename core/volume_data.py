"""
core.volume_data
================

Load a measured drop-volume curve, volume vs time, from a CSV file, for
the Bagalkot volume method.  It may come from this app's own video
analysis (Export frames), another tensiometer, or a curve digitised from
a paper.

Accepted layout (same rules as ``core.ift_data``)::

    # optional comment lines
    time_s,volume_uL        <- optional header; names pick the columns
    0,12.49
    10,12.61
    ...

Volume is in µL (= mm³).  A table exported by the video analysis
(``Export frames``) loads directly: its ``time_s`` and ``volume_uL``
columns are picked by name, and frames without a volume are skipped.  Validation is shared with the IFT loader:
size/row caps, UTF-8 only, every cell goes through float(), NaN/Inf,
negative times, non-positive volumes and duplicate times are refused with
the line number.
"""

import numpy as np

from core.ift_data import IFTDataError, load_xy_curve


class VolumeDataError(IFTDataError):
    """A volume-curve file failed validation."""


def _is_volume(cell):
    c = cell.strip().lower()
    return c.startswith(('vol', 'v_', 'v (', 'v[')) or c == 'v'


def _nonpositive(lineno, v):
    return (f"line {lineno}: volume must be > 0 µL, got {v:g}. A zero or "
            f"negative volume is a failed drop measurement; remove the row.")


def _unit_warnings(v):
    median = float(np.median(v))
    if median < 0.01:
        return [f"median volume is {median:.3g} — that looks like mL or m³; "
                f"this loader expects µL (= mm³)."]
    if median > 1000.0:
        return [f"median volume is {median:.3g} µL — unusually large for a "
                f"pendant drop; check the units (1 mm³ = 1 µL)."]
    return []


def load_volume_curve(path):
    """Parse and validate a volume-vs-time CSV.

    Returns ``{'t': ndarray [s], 'volume': ndarray [µL], 'warnings': [str]}``
    sorted by time, or raises :class:`VolumeDataError`.
    """
    d = load_xy_curve(path, _is_volume, "a volume", "volume (µL)",
                      _nonpositive, _unit_warnings, error_cls=VolumeDataError,
                      skip_blank_values=True)
    return dict(t=d['t'], volume=d['y'], warnings=d['warnings'])
