"""
core.result_export
==================

Write a stored analysis result (core.analyses, ``analyses/<name>.json``) as
CSV for a spreadsheet or another program, as the Bagalkot window's Export
results did: the settings and scalar results as ``# key,value`` comment
lines, then the curve as a table.  Bagalkot and Yang also get their search
sweep in a second file next to it (``<name>_sweep.csv``).

Nothing here touches Qt.
"""

import csv
import math
from pathlib import Path

# name -> (title, curve columns: (key in data['curve'], CSV header), sweep)
LAYOUT = {
    'swelling': ('Swelling check', [('t', 'time_s'), ('V', 'volume_uL')], None),
    'bagalkot': ('Bagalkot (2019) volume method',
                 [('t', 'time_s'), ('V_exp', 'volume_measured_uL'),
                  ('V_model', 'volume_model_uL')],
                 [('D', 'D_m2_per_s'), ('F', 'F_percent'), ('SF', 'SF')]),
    'yang': ('Yang et al. (2006) IFT method',
             [('t', 'time_s'), ('gamma_exp', 'ift_measured_mNm'),
              ('gamma_fit', 'ift_fitted_mNm')],
             [('kD', 'kD'), ('D', 'best_D_m2_per_s'), ('E', 'E_percent')]),
    'simulation': ('Forward diffusion simulation',
                   [('t', 'time_s'), ('Cs', 'Cs_over_csat'), ('gamma', 'ift_mNm')], None),
}


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        return '' if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def _scalars(prefix, d):
    """``(key, value)`` for every scalar or short list in ``d``."""
    out = []
    for k, v in (d or {}).items():
        if isinstance(v, (list, tuple)) and len(v) <= 4 and all(
                not isinstance(x, (list, dict)) for x in v):
            out.append((f"{prefix}{k}", ','.join(_cell(x) for x in v)))
        elif not isinstance(v, (list, tuple, dict)):
            out.append((f"{prefix}{k}", _cell(v)))
    return out


def export_result(name, data, dest, source=''):
    """Write ``data`` (a stored document's 'data') to ``dest``; returns the
    list of files written."""
    if name not in LAYOUT:
        raise ValueError(f"no CSV layout for {name!r}")
    title, columns, sweep_cols = LAYOUT[name]
    dest = Path(dest)
    curve = data.get('curve') or {}
    cols = [(k, h) for k, h in columns if curve.get(k) is not None]
    n = max((len(curve[k]) for k, _ in cols), default=0)
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        f.write(f"# {title}" + (f", session: {source}" if source else '') + "\n")
        for k, v in _scalars('setting.', data.get('settings')) + _scalars('', data.get('result')):
            f.write(f"# {k},{v}\n")
        w = csv.writer(f)
        w.writerow([h for _, h in cols])
        for i in range(n):
            w.writerow([_cell(float(curve[k][i]) if curve[k][i] is not None else None)
                        for k, _ in cols])
    files = [dest]
    sweep = data.get('sweep')
    if sweep_cols and sweep:
        p = dest.with_name(dest.stem + '_sweep.csv')
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([h for _, h in sweep_cols])
            for row in zip(*(sweep[k] for k, _ in sweep_cols)):
                w.writerow([_cell(None if x is None else float(x)) for x in row])
        files.append(p)
    return files
