"""
Analyses on a session folder (core.analyses): swelling check, Bagalkot
and Yang run from what the folder holds, store their results with the
log they used, refuse clearly when an input is missing, and can be
cancelled.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_analyses.py      (about 2 min)
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _synthetic_video import render_drop, write_swelling_video
from core import analyses as A
from core.frame_series import FrameSeriesCancelled, sample_times
from core.session import Session, SessionError, SessionFolder

ROOT = Path(__file__).resolve().parent.parent

_passed = 0
_failed = 0
CALIB = ("c_sat,888.2\nconcentration_unit,mol/m3\nconcentration,ift\n"
         "0,70.55\n155.1,57.66\n400,45.1\n888.2,30.2\n")


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def raises(fn, exc=A.AnalysisError):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        print(f"       (raised {type(e).__name__}: {e})")
        return False
    return False


def session(kind='video', **src):
    return Session(source=dict(kind=kind, name='synthetic', **src),
                   scale=dict(pixel_to_m=1e-5, method='manual'),
                   baseline=dict(y_px=60, method='auto'),
                   needle=dict(od_mm=0.8, tol_rel=0.01),
                   fluids=dict(density_inner=1000.0, density_outer=1.2),
                   analyses=['swelling', 'bagalkot', 'yang'])


def series_folder(tmp, name='series', swell=0.06, n=14):
    vid = tmp / f'{name}.avi'
    write_swelling_video(vid, lambda t: 1 + swell * (1 - np.exp(-t / 8)), duration=30, fps=10)
    f = SessionFolder.create(tmp / name, session(path=str(vid), t_start_s=0.0))
    f.log_video(sample_times(29.5, 'dense_early', n_frames=n, first_step=0.5))
    f.finish()
    return f


def main():
    tmp = Path(tempfile.mkdtemp(prefix='analyses test '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    print("── refusals ──")
    one = SessionFolder.create(tmp / 'one', session(kind='image'))
    one.log_image(render_drop(0.30, 70)[0])
    one.finish()
    for name in ('swelling', 'bagalkot', 'yang'):
        check(f"{name} refuses a single image", raises(lambda n=name: A.RUNNERS[n](one)))

    single_image_yang(tmp, one)

    f = series_folder(tmp)
    n_qc = sum(r['qc_pass'] for r in f.log.rows)
    check("series logged", len(f.log) >= 12, f"{len(f.log)} frames, {n_qc} passed QC")
    check("yang refuses without a calibration", raises(lambda: A.run_yang(f)))
    check("bad swelling threshold refused", raises(lambda: A.run_swelling(f, threshold=2)))
    check("bad D range refused", raises(lambda: A.run_bagalkot(f, D_range=(1e-8, 1e-9))))

    print("── swelling ──")
    d = A.run_swelling(f, threshold=0.03)
    r = d['result']
    check("6 % swelling is significant", r['verdict'] == 'significant' and r['recommend'] == 'bagalkot',
          r['message'][:80])
    check("swelling close to the true 5.9 %", abs(100 * r['S'] - 5.9) < 1.0, f"{100 * r['S']:.2f} %")
    doc = f.load_analysis('swelling')
    check("stored with the log hash", doc['log_sha256'] == f.session.log_sha256)
    check("settings stored", doc['data']['settings']['threshold'] == 0.03)
    d2 = A.run_swelling(f, threshold=0.10)
    check("higher threshold: negligible", d2['result']['verdict'] != 'significant',
          d2['result']['verdict'])

    print("── Bagalkot ──")
    seen = []
    t0 = time.time()
    d = A.run_bagalkot(f, progress=lambda i, n, t: seen.append(i))
    r = d['result']
    check("D found", 1e-11 < r['D'] < 1e-8 and not r['at_edge'],
          f"D = {r['D']:.3e} m2/s in {time.time() - t0:.1f} s")
    check("progress reported", len(seen) > 10)
    check("model curve stored", len(d['curve']['V_model']) == len(d['curve']['t']))
    stop = [0]

    def cancel_soon():
        stop[0] += 1
        return stop[0] > 3
    check("Bagalkot can be cancelled",
          raises(lambda: A.run_bagalkot(f, D_range=(1e-10, 1e-9), is_cancelled=cancel_soon),
                 FrameSeriesCancelled))
    check("cancelled run kept the earlier result", f.load_analysis('bagalkot')['data']['result']['D'] == r['D'])

    print("── Yang ──")
    calib = tmp / 'calib.csv'
    calib.write_text(CALIB, encoding='utf-8')
    f.set_calibration(calib)
    if n_qc < 10:
        check("yang refuses fewer than 10 QC-passed frames", raises(lambda: A.run_yang(f)))
        return
    model = A.build_model(f)
    check("model built from the first measured frame",
          model['quality']['n_nodes'] > 100 and model['frame']['frame_index'] == 0,
          f"{model['quality']['n_nodes']} nodes")
    check("negative time offset refused", raises(lambda: A.run_yang(f, t_offset=-100)))
    check("bad kD range refused", raises(lambda: A.run_yang(f, kD_range=(10, 1))))
    sims = [0]

    def cancel_after_sims():
        sims[0] += 1
        return sims[0] > 2
    check("Yang can be cancelled",
          raises(lambda: A.run_yang(f, progress=lambda i, n, t: None, is_cancelled=cancel_after_sims),
                 FrameSeriesCancelled))
    check("nothing stored by the cancelled run", f.load_analysis('yang') is None)
    t0 = time.time()
    d = A.run_yang(f, kD_range=(0.5, 50.0), D_bounds=(1e-10, 1e-8))
    r = d['result']
    check("Yang ran on a manual range", np.isfinite(r['D']) and np.isfinite(r['k']),
          f"D = {r['D']:.3e} m2/s, k = {r['k']:.3e} m/s in {time.time() - t0:.0f} s")
    check("fit curve on the measured grid", len(d['curve']['gamma_fit']) == len(d['curve']['t']))
    check("mesh stored for the Mesh tab", len(d['model']['triangles']) == d['model']['n_elements'])
    check("manual settings stored", d['settings']['auto'] is False
          and list(d['settings']['kD_range']) == [0.5, 50.0])

    print("── out of date ──")
    f.rebuild_log(prep_ops=[('threshold', 110)])
    check("threshold rebuild marks results out of date",
          all(f.analysis_out_of_date(n) for n in ('swelling', 'bagalkot', 'yang')))


def single_image_yang(tmp, one):
    """Yang on one logged image: the outline from that frame, IFT(t) from a
    CSV (as the Pendant window's Static mode with Load IFT curve)."""
    print("── Yang on a single image + IFT curve file ──")
    calib = tmp / 'calib1.csv'
    calib.write_text(CALIB, encoding='utf-8')
    one.set_calibration(calib)
    ok, why = one.yang_availability()
    check("single image without a curve: Yang asks for one", not ok and 'IFT curve' in why, why)
    bad = tmp / 'bad.csv'
    bad.write_text("time_s,ift_mNm\n1,abc\n", encoding='utf-8')
    check("a bad curve file is refused", raises(lambda: one.set_ift_curve(bad), SessionError))
    check("nothing stored after the refusal", one.ift_curve_path is None)
    src = ROOT / 'data' / 'ift_curve_5.67MPa.csv'
    c = one.set_ift_curve(src)
    check("curve copied into the session", one.ift_curve_path is not None
          and one.ift_curve_path.read_bytes() == src.read_bytes(), f"{len(c['t'])} points")
    check("file route available", one.yang_availability('file') == (True, ''))
    check("log route still refused", not one.yang_availability('log')[0],
          one.yang_availability('log')[1])
    check("log route refused by run_yang", raises(lambda: A.run_yang(one, curve='log')))
    check("unknown frame refused",
          raises(lambda: A.run_yang(one, curve='file', frame_index=99, kD_range=(0.5, 50.0),
                                    D_bounds=(1e-10, 1e-8))))
    t0 = time.time()
    d = A.run_yang(one, curve='file', frame_index=0, kD_range=(0.5, 50.0),
                   D_bounds=(1e-10, 1e-8), t_offset=2.0)
    r = d['result']
    check("Yang ran on one image + the curve file", np.isfinite(r['D']) and np.isfinite(r['k']),
          f"D = {r['D']:.3e} m2/s, k = {r['k']:.3e} m/s in {time.time() - t0:.0f} s")
    check("fitted on the file's points", len(d['curve']['t']) == len(c['t'])
          and np.allclose(d['curve']['gamma_exp'], c['gamma']))
    check("outline from frame 0", d['model']['frame_index'] == 0)
    st = d['settings']
    check("curve source stored", st['curve'] == 'file' and st['frame_index'] == 0
          and len(st['curve_sha256']) == 64)
    check("result is current", not one.analysis_out_of_date('yang'))
    edited = tmp / 'edited.csv'
    edited.write_text(src.read_text(encoding='utf-8').replace('39.782', '39.700'), encoding='utf-8')
    one.set_ift_curve(edited)
    check("a new curve file marks the result out of date", one.analysis_out_of_date('yang'))


if __name__ == "__main__":
    main()
