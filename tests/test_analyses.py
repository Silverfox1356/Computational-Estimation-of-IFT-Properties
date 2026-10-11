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
    single_image_bagalkot(tmp, one)

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
    sim = one.load_analysis('simulation')
    check("Yang also stored a simulation at its best fit", sim is not None
          and sim['data']['settings']['source'] == 'yang'
          and np.isclose(sim['data']['settings']['D'], r['D'])
          and np.isclose(sim['data']['settings']['k'], r['k']))
    check("model summary has the domain and FEM checks",
          d['model']['domain']['n_points'] > 10 and len(d['model']['fem']['checks']) == 8
          and d['model']['fem']['all_pass'])
    simulation_checks(tmp, one)
    grid_checks(one)
    edited = tmp / 'edited.csv'
    edited.write_text(src.read_text(encoding='utf-8').replace('39.782', '39.700'), encoding='utf-8')
    one.set_ift_curve(edited)
    check("a new curve file marks the result out of date", one.analysis_out_of_date('yang'))


def single_image_bagalkot(tmp, one):
    """Bagalkot on a session of one image with V(t) from a CSV (the Bagalkot
    window's Load volume curve)."""
    print("── Bagalkot on a single image + volume curve file ──")
    ok, why = one.bagalkot_availability()
    check("single image without a curve: Bagalkot not available, says why",
          not ok and 'volume curve file' in why, why)
    src = ROOT / 'data' / 'volume_curve_template.csv'
    c = one.set_volume_curve(src)
    check("volume curve copied into the session", one.volume_curve_path is not None,
          f"{len(c['t'])} points")
    check("file route available", one.bagalkot_availability('file') == (True, ''))
    check("log route refused", raises(lambda: A.run_bagalkot(one, curve='log')))
    d = A.run_bagalkot(one, curve='file')
    r = d['result']
    check("Bagalkot ran on the curve file", 1e-12 < r['D'] < 1e-7, f"D = {r['D']:.3e} m2/s")
    check("fitted on the file's points", len(d['curve']['t']) == len(c['t']))
    check("curve source stored", d['settings']['curve'] == 'file'
          and d['settings']['curve_file'] == 'volume_curve.csv')
    check("result is current", not one.analysis_out_of_date('bagalkot'))
    edited = tmp / 'vol_edited.csv'
    edited.write_text(src.read_text(encoding='utf-8').replace('12.62444', '12.62000'),
                      encoding='utf-8')
    yang_before = one.analysis_out_of_date('yang')
    one.set_volume_curve(edited)
    check("a new volume file marks Bagalkot out of date", one.analysis_out_of_date('bagalkot'))
    check("... and leaves Yang's state alone (it has its own file)",
          one.analysis_out_of_date('yang') == yang_before)


def grid_checks(one):
    """Yang's own kD grid (his Fig 7a) and the E-vs-D curves."""
    print("── Yang kD grid ──")
    check("kD values from, to, step + extras", A.kD_grid(1, 5, 1, [0.1, 0.5]) ==
          [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    check("bad step refused", raises(lambda: A.kD_grid(1, 5, 0)))
    check("too many values refused", raises(lambda: A.kD_grid(0.01, 100, 0.01)))
    check("bad D axis refused", raises(lambda: A.run_yang(one, curve='file', kD_values=[1],
                                                          D_grid=(1e-8, 1e-12, 60))))
    t0 = time.time()
    d = A.run_yang(one, curve='file', kD_values=[0.5, 1, 2, 4], D_grid=(1e-12, 1e-8, 40))
    r, curves = d['result'], d['curves']
    check("one curve per kD", [c['kD'] for c in curves] == [0.5, 1, 2, 4]
          and all(len(c['D']) == 40 for c in curves), f"{time.time() - t0:.0f} s")
    check("one simulation per kD", r['n_evaluations'] == 4, str(r['n_evaluations']))
    check("exact minimum at or below the plotted points",
          all(c['E_min'] <= min(c['E']) + 1e-9 for c in curves))
    best = min(curves, key=lambda c: c['E_min'])
    check("answer is the lowest of all curves", best['kD'] == r['kD']
          and np.isclose(best['D_min'], r['D']))
    rn = d['model']['domain']['r_inner_m']
    check("k = kD D / rn", np.isclose(r['k'], r['kD'] * r['D'] / rn))
    check("grid settings stored", d['settings']['search'] == 'grid'
          and d['settings']['D_grid'] == [1e-12, 1e-8, 40])
    a = A.run_yang(one, curve='file', kD_range=(0.5, 50.0), D_bounds=(1e-10, 1e-8))
    check("automatic search also stores curves, best included",
          0 < len(a['curves']) <= 8 and any(np.isclose(c['kD'], a['result']['kD'])
                                            for c in a['curves']),
          f"{len(a['curves'])} curves")


def simulation_checks(tmp, one):
    """The forward simulation (the Pendant window's Step 7) on one image."""
    print("── forward simulation ──")
    D0, k0, dur0, frame0 = A.simulation_defaults(one)
    yang = one.load_analysis('yang')['data']
    check("defaults come from the Yang fit", np.isclose(D0, yang['result']['D'])
          and frame0 == 0, f"D {D0:.3g}, k {k0:.3g}, {dur0:g} s")
    check("bad D refused", raises(lambda: A.run_simulation(one, D=-1)))
    check("bad time step refused", raises(lambda: A.run_simulation(one, dt_s=0)))
    check("too many steps refused", raises(lambda: A.run_simulation(one, duration_s=1e6,
                                                                       dt_s=1e-3)))
    stop = [0]

    def cancel_soon():
        stop[0] += 1
        return stop[0] > 20
    before = one.load_analysis('simulation')['data']['settings']
    check("simulation can be cancelled",
          raises(lambda: A.run_simulation(one, D=2e-9, k=2e-5, duration_s=300, dt_s=0.01,
                                          is_cancelled=cancel_soon), FrameSeriesCancelled))
    check("cancelled run kept the earlier result",
          one.load_analysis('simulation')['data']['settings'] == before)
    t0 = time.time()
    seen = []
    d = A.run_simulation(one, D=2e-9, k=3e-5, duration_s=300, dt_s=0.5,
                         progress=lambda i, n, t: seen.append(i))
    r = d['result']
    check("uniform-step simulation (the old Step 7)", r['n_steps'] == 600
          and abs(r['simulated_s'] - 300) < 1e-6, f"{time.time() - t0:.1f} s")
    check("kD = k rn / D", np.isclose(r['kD'], 3e-5 * d['model']['domain']['r_inner_m'] / 2e-9))
    check("concentration rises towards saturation", 0 < r['Cs_final'] <= 1.01
          and np.all(np.diff(d['curve']['Cs']) > -1e-6), f"Cs {r['Cs_final']:.4f}")
    check("gamma(t) from the calibration, falling", r['calibrated']
          and d['curve']['gamma'][0] > d['curve']['gamma'][-1],
          f"{d['curve']['gamma'][0]:.2f} -> {d['curve']['gamma'][-1]:.2f} mN/m")
    check("snapshots over the run", len(d['snapshots']) >= 6
          and len(d['snapshots'][0]['C']) == d['model']['n_nodes']
          and abs(d['snapshots'][-1]['t'] - 300) < 1e-6)
    check("no NaN in the field", not r['has_nan'])
    check("measured IFT kept for the overlay", d['measured'] is not None)
    check("step log kept short", 0 < len(d['diagnostics']) <= A.SIM_LOG_LINES + 1)
    check("progress reported", len(seen) > 10)
    check("settings stored", d['settings']['source'] == 'manual'
          and d['settings']['stepping'] == 'uniform' and d['settings']['dt_s'] == 0.5)
    g = A.run_simulation(one, D=2e-9, k=3e-5, duration_s=300)
    check("automatic steps agree with fine uniform steps",
          abs(g['result']['Cs_final'] - r['Cs_final']) < 0.01,
          f"{g['result']['n_steps']} graded steps, Cs {g['result']['Cs_final']:.4f}")
    check("simulation listed as available", one.availability()['simulation'] == (True, ''))


if __name__ == "__main__":
    main()
