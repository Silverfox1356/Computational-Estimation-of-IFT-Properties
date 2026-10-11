"""
Video sampling and per-frame analysis (core.frame_series) on a synthetic
video of a swelling drop with a known volume.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_frame_series.py      (~40 s)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _synthetic_video import render_drop, write_swelling_video
from core.frame_series import (analyse_frame, analyse_video, apply_prep,
                               frames_for_times, load_frame_table_rows,
                               sample_times, save_frame_table, table_arrays,
                               video_info)
from core.swelling import swelling_check

_passed = 0
_failed = 0
PX = 1e-5                          # 10 µm per pixel
PARAMS = dict(baseline_y=60, pixel_to_m=PX, density_inner=1000.0,
              density_outer=1.2, needle_diameter_m=80 * PX)


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("── sampling ──")
    t = sample_times(600, 'dense_early', n_frames=120, first_step=0.6)
    check("dense early: 120 times, 0 first, span last",
          len(t) == 120 and t[0] == 0 and abs(t[-1] - 600) < 1e-9)
    check("dense early: steps grow", np.all(np.diff(np.diff(t[1:])) > 0))
    t = sample_times(600, 'interval', interval=5)
    check("interval: 0, 5, ..., 600", len(t) == 121 and t[1] == 5 and t[-1] == 600)
    idx, ta = frames_for_times([0, 0.001, 0.02, 1.0], fps=60, t_start=2.0,
                               n_total=1000)
    check("snapping to frames removes duplicates", list(idx) == [120, 121, 180]
          and abs(ta[1] - 1 / 60) < 1e-12, f"{list(idx)}")
    for bad in (dict(mode='interval', interval=0),
                dict(mode='dense_early', n_frames=2),
                dict(mode='interval', interval=0.01)):
        try:
            sample_times(600, **bad)
            check(f"rejects {bad}", False)
        except ValueError:
            check(f"rejects {bad}", True)

    print("── preparation ──")
    img, vol, _ = render_drop(0.30, 70.0, tip_y=60)
    big = np.full((img.shape[0] + 40, img.shape[1] + 60, 3), 225, np.uint8)
    big[20:20 + img.shape[0], 30:30 + img.shape[1]] = img
    prepared = apply_prep(big, [('crop', (30, 20, img.shape[1], img.shape[0])),
                                ('threshold', 120)])
    rec = analyse_frame(prepared, PARAMS)
    v_true = vol * PX ** 3 * 1e9
    check("crop + threshold, then fit: volume within 1.5 %",
          rec['fit_ok'] and abs(rec['volume_uL'] / v_true - 1) < 0.015,
          f"{rec['volume_uL']:.4f} vs {v_true:.4f} µL")

    print("── synthetic video (6 % swelling) ──")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "swell.avi"
        swell = lambda tt: 1.0 + 0.06 * (1 - np.exp(-tt / 2.0))
        tt, vols = write_swelling_video(path, swell, duration=8.0, fps=10.0)
        info = video_info(path)
        check("video info", info['n_frames'] == 80 and abs(info['fps'] - 10) < 1e-6,
              f"{info}")
        times = sample_times(6.0, 'interval', interval=1.0)
        recs = analyse_video(path, times, 1.0, [], PARAMS)
        a = table_arrays(recs)
        check("times are measured from t_start", np.allclose(a['t'], np.arange(7)),
              f"{a['t']}")
        truth = vols[[r['frame_index'] for r in recs]] * PX ** 3 * 1e9
        ratio_err = np.max(np.abs(a['volume'] / a['volume'][0]
                                  - truth / truth[0]))
        check("V/V0 within 0.3 % of the truth", ratio_err < 3e-3,
              f"max |error| {ratio_err:.2e}")
        check("every frame fitted", bool(np.all(a['fit_ok'])))
        check("first frame keeps its image", recs[0].get('image') is not None
              and 'image' not in recs[1])
        sw = swelling_check(a['t'], a['volume'])
        check("swelling check: significant", sw['verdict'] == 'significant',
              f"S = {100 * sw['S']:.2f} %")
        recs2 = analyse_video(path, times[:3], 1.0, [], PARAMS)
        check("same video, same numbers",
              all(r1['volume_uL'] == r2['volume_uL'] and r1['ift_mNm'] == r2['ift_mNm']
                  for r1, r2 in zip(recs, recs2)))
        out = Path(tmp) / "frames.csv"
        save_frame_table(out, recs, ["test"])
        rows = load_frame_table_rows(out)
        check("exported table reads back",
              len(rows) == len(recs) and abs(rows[3]['volume_uL'] - recs[3]['volume_uL'])
              < 1e-5 * recs[3]['volume_uL'])

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
