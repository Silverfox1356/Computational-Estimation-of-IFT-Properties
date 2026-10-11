"""
Session folders (core.session) and the frame log (core.frame_log):
create -> log -> finish -> open round trip, integrity checks, rebuild,
column choice, analysis availability, calibration and analysis storage.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_session.py      (about 30 s)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _synthetic_video import render_drop, write_swelling_video
from core import frame_log as fl
from core.frame_series import load_frame_table_rows, sample_times
from core.session import (FRAMES_DIR, LOG_FILE, SESSION_FILE, Session,
                          SessionError, SessionFolder, split_prep)

_passed = 0
_failed = 0
PX = 1e-5                          # 10 µm per pixel
REPO = Path(__file__).resolve().parent.parent


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def raises(fn, exc=SessionError):
    try:
        fn()
    except exc:
        return True
    except Exception as e:                       # wrong exception type
        print(f"       (raised {type(e).__name__}: {e})")
        return False
    return False


def make_session(kind='image', analyses=(), columns=None, prep_ops=(), **src):
    return Session(source=dict(kind=kind, name='synthetic', **src),
                   scale=dict(pixel_to_m=PX, method='manual'),
                   baseline=dict(y_px=60, method='auto'),
                   needle=dict(od_mm=0.8, tol_rel=0.01),
                   fluids=dict(density_inner=1000.0, density_outer=1.2,
                               name='water / air'),
                   prep_ops=list(prep_ops), analyses=list(analyses),
                   columns=columns)


def log_drops(folder, R0s, dt=10.0):
    recs = []
    for k, R0 in enumerate(R0s):
        img, _, _ = render_drop(0.30, R0)
        recs.append(folder.log_frame(img, k * dt, k))
    folder.finish()
    return recs


def main():
    tmp = Path(tempfile.mkdtemp(prefix='session test µ é '))   # non-ASCII on purpose
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    print("── columns ──")
    keys = fl.column_keys(['beta'], analyses=['swelling'])
    check("required column added", 'volume_uL' in keys and 'beta' in keys, ','.join(keys))
    check("always-logged columns first", keys[:5] == [c.key for c in fl.ALWAYS])
    check("camera adds clock column", 'wall_clock' in fl.column_keys(camera=True))
    check("unknown column rejected", raises(lambda: fl.column_keys(['nope']), fl.FrameLogError))
    check("unknown analysis rejected", raises(lambda: fl.required_columns(['x']), fl.FrameLogError))
    check("missing required column rejected",
          raises(lambda: make_session(analyses=['yang'],
                                      columns=fl.column_keys(['beta']))))

    print("── session validation ──")
    check("equal densities rejected", raises(lambda: Session(
        source=dict(kind='image'), scale=dict(pixel_to_m=PX), baseline=dict(y_px=1),
        needle=dict(od_mm=1), fluids=dict(density_inner=1, density_outer=1))))
    check("bad source kind rejected", raises(lambda: make_session(kind='mp3')))
    check("bad threshold rejected", raises(lambda: make_session(prep_ops=[('threshold', 300)])))
    s = make_session()
    d = s.to_dict()
    d['schema_version'] = 99
    check("newer schema rejected", raises(lambda: Session.from_dict(d)))
    check("round trip through dict", Session.from_dict(s.to_dict()).to_dict() == s.to_dict())
    check("split prep", split_prep([('threshold', 9), ('crop', (0, 0, 5, 5))])
          == ([('crop', (0, 0, 5, 5))], [('threshold', 9)]))

    print("── log a few frames ──")
    p1 = tmp / 'drops'
    folder = SessionFolder.create(p1, make_session(analyses=['swelling']))
    R0s = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]     # 10: Yang's minimum
    recs = log_drops(folder, R0s)
    frames = sorted((p1 / FRAMES_DIR).glob('*.png'))
    check("one PNG per frame", len(frames) == len(R0s), f"{len(frames)} files")
    check("frame name has index and time", frames[1].name == '000001_t10.000.png', frames[1].name)
    check("all fits ok", all(r['fit_ok'] for r in recs))
    vols = folder.log.column('volume_uL')
    check("volume grows with R0", np.all(np.diff(vols) > 0), str(np.round(vols, 4)))
    check("duplicate frame rejected",
          raises(lambda: folder.log_frame(render_drop(0.3, 60)[0], 99, 0)))
    check("negative time rejected",
          raises(lambda: folder.log_frame(render_drop(0.3, 60)[0], -1, 50)))
    meta = json.loads((p1 / SESSION_FILE).read_text(encoding='utf-8'))
    check("hash recorded", meta['log_sha256'] and meta['n_logged'] == len(R0s))
    rows = load_frame_table_rows(p1 / LOG_FILE)
    check("log readable by frame_series loader", len(rows) == len(R0s)
          and np.allclose([r['volume_uL'] for r in rows], vols, rtol=1e-5))

    print("── open again ──")
    f2 = SessionFolder.open(p1)
    check("no problems", f2.problems == [], str(f2.problems))
    check("same rows", len(f2.log) == len(R0s) and np.allclose(
        f2.log.column('volume_uL'), vols, rtol=1e-5))
    check("same columns", f2.log.columns == folder.log.columns)
    row, img = f2.first_measured()
    check("first measured frame", row['frame_index'] == 0 and img.ndim == 3)
    av = f2.availability()
    check("swelling available", av['swelling'][0], av['swelling'][1])
    check("yang needs calibration", not av['yang'][0] and 'calibration' in av['yang'][1],
          av['yang'][1])
    calib = tmp / 'calib.csv'
    calib.write_text("c_sat,888.2\nconcentration_unit,mol/m3\nconcentration,ift\n"
                     "0,70.55\n155.1,57.66\n400,45.1\n888.2,30.2\n", encoding='utf-8')
    f2.set_calibration(calib)
    check("calibration copied", (p1 / 'calibration.csv').is_file())
    check("yang available after calibration", f2.availability()['yang'][0],
          f2.availability()['yang'][1])
    check("bad calibration rejected",
          raises(lambda: f2.set_calibration(REPO / 'data' / 'ift_curve_5.67MPa.csv')))

    print("── analyses ──")
    f2.save_analysis('swelling', dict(S=np.float64(0.12), flags=np.array([1, 2]),
                                      nan=float('nan')))
    doc = f2.load_analysis('swelling')
    check("analysis stored", doc['data']['S'] == 0.12 and doc['data']['flags'] == [1, 2]
          and doc['data']['nan'] is None)
    check("bad analysis name rejected", raises(lambda: f2.save_analysis('../x', {})))
    check("analysis not out of date", not f2.analysis_out_of_date('swelling'))

    print("── integrity ──")
    lp = p1 / LOG_FILE
    lp.write_text(lp.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    f3 = SessionFolder.open(p1)
    check("edited log noticed", [p.code for p in f3.problems] == ['log_modified'],
          str([p.code for p in f3.problems]))
    f3.rebuild_log()
    check("rebuild clears it", SessionFolder.open(p1).problems == [])
    check("rebuild reproduces values", np.allclose(f3.log.column('volume_uL'), vols, rtol=1e-6))
    check("identical rebuild keeps analysis current", not f3.analysis_out_of_date('swelling'))

    f3.rebuild_log(columns=fl.column_keys(['volume_uL', 'Wo']))
    f4 = SessionFolder.open(p1)
    check("rebuild with new columns", f4.log.has('Wo') and not f4.log.has('beta')
          and np.all(np.isfinite(f4.log.column('Wo'))))
    check("changed log marks analysis out of date", f4.analysis_out_of_date('swelling'))
    check("rebuild can't drop a required column",
          raises(lambda: f4.rebuild_log(columns=fl.column_keys(['beta']))))

    lp.unlink()
    f5 = SessionFolder.open(p1)
    check("missing log noticed", [p.code for p in f5.problems] == ['log_missing'])
    f5.rebuild_log()
    check("log rebuilt from frame names", len(f5.log) == len(R0s) and np.allclose(
        f5.log.column('time_s'), np.arange(len(R0s)) * 10.0))

    next((p1 / FRAMES_DIR).glob('000003_*.png')).unlink()
    f6 = SessionFolder.open(p1)
    check("missing frame noticed", [p.code for p in f6.problems] == ['frames_missing'])
    f6.rebuild_log()
    check("missing frame kept as a failed row",
          not f6.log.rows[3]['fit_ok'] and 'missing' in f6.log.rows[3]['error'])

    (p1 / SESSION_FILE).write_text('{not json', encoding='utf-8')
    check("broken session.json rejected", raises(lambda: SessionFolder.open(p1)))
    check("non-session folder rejected", raises(lambda: SessionFolder.open(tmp)))
    check("create refuses a non-empty folder",
          raises(lambda: SessionFolder.create(p1, make_session())))

    print("── unfinished logging ──")
    p2 = tmp / 'crash'
    fc = SessionFolder.create(p2, make_session())
    for k in range(3):
        fc.log_frame(render_drop(0.3, 60 + k)[0], k, k)
    # no finish(): the app "crashed"
    f7 = SessionFolder.open(p2)
    check("unfinished log noticed", [p.code for p in f7.problems] == ['log_unfinished'])
    check("rows written so far kept", len(f7.log) == 3)

    print("── single image ──")
    p3 = tmp / 'image'
    fi = SessionFolder.create(p3, make_session())
    rec = fi.log_image(render_drop(0.3, 70)[0])
    fi.finish()
    av = SessionFolder.open(p3).availability()
    check("image measured", rec['fit_ok'], f"IFT {rec['ift_mNm']:.2f} mN/m")
    check("time series analyses not available", not any(ok for ok, _ in av.values()))
    check("reason names the single image", 'single image' in av['swelling'][1], av['swelling'][1])

    print("── crop and threshold ──")
    img, _, _ = render_drop(0.3, 70)
    H, W = img.shape[:2]
    pc = tmp / 'prep'
    fp = SessionFolder.create(pc, make_session(prep_ops=[('crop', (10, 0, W - 20, H)),
                                                         ('threshold', 120)]))
    rec = fp.log_frame(img, 0, 0)
    fp.finish()
    stored = fp.read_frame(0, prepared=False)
    check("stored frame cropped", stored.shape[:2] == (H, W - 20), str(stored.shape))
    check("stored frame not thresholded", len(np.unique(stored)) > 2)
    check("threshold applied when read", len(np.unique(fp.read_frame(0))) <= 2)
    check("prepared frame measured", rec['fit_ok'])
    fp.rebuild_log(prep_ops=[('threshold', 100)])
    check("rebuild with a new threshold", SessionFolder.open(pc).session.prep_ops
          == [('crop', (10, 0, W - 20, H)), ('threshold', 100)])
    check("rebuild can't crop again",
          raises(lambda: fp.rebuild_log(prep_ops=[('crop', (0, 0, 5, 5))])))

    print("── video ──")
    vid = tmp / 'swell.avi'
    t_all, v_true = write_swelling_video(vid, lambda t: 1 + 0.05 * t / 10, duration=10, fps=10)
    pv = tmp / 'video'
    fv = SessionFolder.create(pv, make_session(kind='video', analyses=['swelling'],
                                               path=str(vid), t_start_s=0.0))
    ts = sample_times(9.5, mode='interval', interval=1.0)
    n = fv.log_video(ts)
    fv.finish()
    fo = SessionFolder.open(pv)
    v = fo.log.column('volume_uL')
    check("video frames logged", n == len(ts) == len(fo.log), f"{n} of {len(ts)}")
    check("video volume swells about 5 %", 0.03 < v[-1] / v[0] - 1 < 0.07,
          f"{100 * (v[-1] / v[0] - 1):.2f} %")
    check("swelling available on video", fo.availability()['swelling'][0])
    check("log_video refuses an image session",
          raises(lambda: fi.log_video([0.0])))


if __name__ == "__main__":
    main()
