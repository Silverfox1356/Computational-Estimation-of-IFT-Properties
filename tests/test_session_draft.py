"""
Setting up a session (core.session_draft): sources, crop and threshold,
fluids and needle, scale, baseline, sampling plans, what is missing, and
the hand-over to a session folder.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_session_draft.py      (about 30 s)
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from _synthetic_video import render_drop, write_swelling_video
from core.image_processing import measure_reference
from core.session import SessionFolder
from core.session_draft import DraftError, SessionDraft

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def raises(fn, exc=DraftError):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        print(f"       (raised {type(e).__name__}: {e})")
        return False
    return False


def main():
    tmp = Path(tempfile.mkdtemp(prefix='draft test µ '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    img, _, apex = render_drop(0.30, 70)            # needle 80 px wide, tip at row 60
    png = tmp / 'drop é.png'
    cv2.imencode('.png', img)[1].tofile(str(png))

    print("── reference measurement ──")
    px, marker = measure_reference(img[5:45, 110:210], 'needle')
    check("needle width found", abs(px - 80) <= 2, f"{px} px")
    check("marker is a line", marker[0] == 'line')

    print("── image source ──")
    d = SessionDraft.from_image(png)
    check("image read from a non-ASCII path", d.reference.shape == img.shape)
    check("needs scale and baseline", [k for k, _ in d.missing()] == ['scale', 'baseline'])
    check("image has no series analyses", d.analyses == [])
    check("bad file rejected", raises(lambda: SessionDraft.from_image(tmp / 'nope.png')))

    print("── scale ──")
    px, marker = d.measure_box(110, 5, 100, 40)
    check("box on the needle", abs(px - 80) <= 2 and marker[1][0] >= 110, f"{px} px")
    check("tiny box rejected", raises(lambda: d.measure_box(0, 0, 3, 3)))
    d.set_scale_from_length(px, 0.8, 'needle width', 'test')
    check("scale from length", abs(d.scale['pixel_to_m'] - 0.8e-3 / px) < 1e-12)
    check("too short rejected", raises(lambda: d.set_scale_from_length(2, 1, 'x', 'y')))
    d.set_scale_direct(10)
    check("scale typed in", abs(d.scale['pixel_to_m'] - 1e-5) < 1e-15)
    check("absurd scale rejected", raises(lambda: d.set_scale_direct(0)))

    print("── baseline ──")
    y = d.auto_baseline()
    check("auto baseline near the needle tip", 55 <= y <= 70, f"y = {y}")
    check("baseline outside the frame rejected", raises(lambda: d.set_baseline(10000)))
    check("nothing missing", d.missing() == [])

    print("── crop and threshold ──")
    d.set_crop(20, 0, 280, 340)
    check("crop clears the baseline", d.baseline is None)
    check("stored frame cropped", d.stored_frame().shape[:2] == (340, 280))
    d.set_crop(10, 0, 200, 300)
    check("crops compose", d.crop_box() == (30, 0, 200, 300), str(d.crop_box()))
    check("crop too small rejected", raises(lambda: d.set_crop(0, 0, 5, 5)))
    d.clear_crop()
    check("crop removed", d.crop_box() is None and d.stored_frame().shape == img.shape)
    d.set_threshold(120)
    check("threshold set", d.threshold() == 120 and len(np.unique(d.prepared_frame())) <= 2)
    check("stored frame not thresholded", len(np.unique(d.stored_frame())) > 2)
    d.set_threshold(None)
    check("threshold back to automatic", d.threshold() is None)
    check("bad threshold rejected", raises(lambda: d.set_threshold(300)))

    print("── fluids and needle ──")
    check("equal densities rejected",
          raises(lambda: d.set_fluids(density_inner=1000.0, density_outer=1000.0)))
    d.set_fluids(density_inner=1000.0, density_outer=1.2, name='water / air')
    d.set_needle(0.8, 0.01, 'test needle')
    check("fluids and needle set", d.fluids['density_outer'] == 1.2 and d.needle['od_mm'] == 0.8)

    print("── measure and hand over ──")
    d.auto_baseline()
    rec = d.measure_reference()
    check("reference frame measured", rec['fit_ok'], f"IFT {rec['ift_mNm']:.2f} mN/m")
    check("preview kept for the overlay", d.preview is rec)
    s = d.to_session()
    check("session made", s.source['kind'] == 'image' and s.baseline['y_px'] == d.baseline['y_px'])
    folder = SessionFolder.create(tmp / 'from draft', s)
    r2 = folder.log_image(d.reference)
    folder.finish()
    check("logged like the preview", r2['fit_ok'] and abs(r2['ift_mNm'] - rec['ift_mNm']) < 1e-9)

    print("── video source ──")
    vid = tmp / 'swell.avi'
    write_swelling_video(vid, lambda t: 1 + 0.04 * t / 5, duration=5, fps=10)
    v = SessionDraft.from_video(vid)
    check("video info read", v.source['n_frames'] == 50 and abs(v.source['fps'] - 10) < 1e-6)
    check("series analyses enabled", v.analyses == ['swelling', 'bagalkot', 'yang'])
    keys = v.column_keys()
    check("volume and IFT always logged", 'volume_uL' in keys and 'ift_mNm' in keys)
    v.columns = ['beta']
    check("required columns stay", {'volume_uL', 'ift_mNm', 'beta'} <= set(v.column_keys()))
    p = v.sample_plan()
    check("dense-early plan", 2 <= p['n'] <= 50 and p['t_last'] <= 5.0, f"{p['n']} frames")
    v.sampling.update(mode='interval', interval_s=1.0)
    p = v.sample_plan()
    check("interval plan (0..5 s, the last snapped to the last frame)", p['n'] == 6,
          f"{p['n']} frames")
    v.sampling.update(interval_s=0)
    check("zero interval reported as missing", [k for k, _ in v.missing()][-1] == 'logger')
    v.sampling.update(interval_s=1.0)
    first = v.reference.copy()
    v.set_t_start(4.0)
    check("start time moves the reference frame", not np.array_equal(first, v.reference))
    check("start outside the video rejected", raises(lambda: v.set_t_start(99)))
    p = v.sample_plan()
    check("plan counts from the start time", p['n'] == 2, f"{p['n']} frames")
    check("image has no start time", raises(lambda: d.set_t_start(1)))

    print("── camera source ──")
    c = SessionDraft.from_camera(img, 'Webcam')
    check("camera interval from presets", c.sampling['interval_s'] > 0)
    c.sampling.update(interval_s=0.5, duration_s=2.0)
    check("camera plan", c.sample_plan()['n'] == 5)
    check("camera logs the clock time", 'wall_clock' in c.column_keys())
    c.set_reference(render_drop(0.3, 75)[0])
    check("reference replaced", c.preview is None and not np.array_equal(c.reference, img))


if __name__ == "__main__":
    main()
