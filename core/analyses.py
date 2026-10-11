"""
core.analyses
=============

The analyses that run on a session folder (core.session.SessionFolder),
each reading only what the folder holds and storing its settings and
result in ``analyses/<name>.json`` (stamped with the log it used, so a
rebuilt log marks it out of date):

    swelling   core.swelling.swelling_check on V(t) from the log; recommends
               Yang (drop static enough) or Bagalkot (it swells)
    bagalkot   core.bagalkot_estimator.estimate_D_bagalkot on V(t): D only;
               V(t) from the log, or from a volume-curve CSV stored in the
               session (the Bagalkot window's Load volume curve)
    yang       Yang et al. 2006 on IFT(t): the drop geometry from one
               measured frame (the first, or one the user picks) -> domain
               -> mesh -> FEM, the gamma(c) calibration stored in the
               session, then estimate_D_k_yang.  IFT(t) comes from the log,
               or from an IFT-curve CSV stored in the session, so a session
               of a single image can run it (as the Pendant window's Static
               mode with Load IFT curve did); afterwards one forward
               simulation at the best (D, k) is stored as 'simulation'
    simulation the forward diffusion model for a chosen (D, k) on the drop
               outline of one frame (the Pendant window's Step 7 and its
               Simulation window): concentration snapshots, Cs(t) and,
               with a gamma(c) calibration, gamma(t)

Each ``run_*`` takes ``progress(i, n, text)`` and ``is_cancelled()`` like
the logger, raises ``AnalysisError`` with a message for the user, and
returns the stored document's ``data``.  The same code the Pendant and
Bagalkot windows call does the science; only the inputs come from the
session.

Nothing here touches Qt.
"""

import math

import numpy as np

from core.frame_series import FrameSeriesCancelled
from core.session import YANG_MIN_POINTS     # same minimum as the Pendant window

SWELLING_DEFAULT_THRESHOLD = 0.03
LABELS = dict(swelling='Swelling check', bagalkot='Diffusivity (Bagalkot)',
              yang='Diffusivity (Yang)', simulation='Forward simulation')
# The Pendant window's Step 7 always simulated these; here they are only the
# starting values when no Yang result is stored.
SIM_DEFAULT_D = 1e-9            # m2/s
SIM_DEFAULT_K = 1e-5            # m/s
SIM_DEFAULT_DURATION_S = 600.0  # the Pendant window's 10 min
SIM_SNAPSHOT_FRACTIONS = (0.01, 0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 1.0)
SIM_LOG_LINES = 500             # step-log lines kept (head and tail)
# Yang's kD grid search (his Fig 7a): defaults and limits.  The D axis is his
# 0.001-10 x 1e-9 m2/s; the kD set is his 0.1, 0.5, 1, 2, 3, 4, 5.
YANG_GRID_KD = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
YANG_GRID_D = (1e-12, 1e-8, 60)
MAX_GRID_KD = 60
CURVE_POINTS = 60               # E(D) points per kD curve (automatic search)


def kD_grid(start, stop, step, extras=()):
    """kD values from start to stop (inclusive) in steps of ``step``, plus
    ``extras``; sorted, duplicates removed.  Raises AnalysisError."""
    vals = set()
    for v in (start, stop, step):
        if not (math.isfinite(v) and v > 0):
            raise AnalysisError("kD from, to and step must be positive numbers")
    if stop < start:
        raise AnalysisError("kD to must not be below kD from")
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    if n > MAX_GRID_KD:
        raise AnalysisError(f"that is {n} kD values; use at most {MAX_GRID_KD} "
                            f"(a larger step)")
    vals.update(float(round(start + i * step, 10)) for i in range(n))
    for v in extras:
        if not (math.isfinite(v) and v > 0):
            raise AnalysisError(f"extra kD value {v:g} must be a positive number")
        vals.add(float(v))
    if len(vals) > MAX_GRID_KD:
        raise AnalysisError(f"that is {len(vals)} kD values; use at most {MAX_GRID_KD}")
    return sorted(vals)


class AnalysisError(ValueError):
    """An analysis can't run on this session; message is user-facing."""


def _check(folder, name):
    ok, why = folder.availability()[name]
    if not ok:
        raise AnalysisError(why)


def _series(folder, col, qc_only=False):
    """(t, values) of the measured frames, sorted by time."""
    rows = sorted(folder.log.rows, key=lambda r: r['time_s'])
    keep = [r for r in rows if r['fit_ok'] and math.isfinite(r[col]) and r[col] > 0
            and (r['qc_pass'] or not qc_only)]
    return (np.array([r['time_s'] for r in keep], float),
            np.array([r[col] for r in keep], float),
            len(rows) - len(keep))


# ---------------------------------------------------------------------------
# Swelling
# ---------------------------------------------------------------------------
def run_swelling(folder, threshold=SWELLING_DEFAULT_THRESHOLD, progress=None,
                 is_cancelled=None):
    from core.swelling import swelling_check
    _check(folder, 'swelling')
    if not 0 < threshold < 1:
        raise AnalysisError("the swelling threshold must be between 0 and 100 %")
    t, V, n_skip = _series(folder, 'volume_uL')
    sw = swelling_check(t, V, threshold=threshold)
    data = dict(settings=dict(threshold=float(threshold)),
                result=dict(sw, n_skipped=n_skip),
                curve=dict(t=t, V=V))
    folder.save_analysis('swelling', data)
    if progress:
        progress(1, 1, sw['message'])
    return data


# ---------------------------------------------------------------------------
# Bagalkot (volume) -> D
# ---------------------------------------------------------------------------
def volume_curve(folder, curve):
    """(t, V, n_skipped, curve_sha256) of the volume curve Bagalkot fits:
    'log' = the logged frames, 'file' = volume_curve.csv."""
    from core.session import SessionError, sha256_file
    if curve == 'log':
        t, V, n_skip = _series(folder, 'volume_uL')
        return t, V, n_skip, None
    if curve != 'file':
        raise AnalysisError(f"unknown volume curve source {curve!r}")
    try:
        c = folder.volume_curve()
    except SessionError as e:
        raise AnalysisError(str(e)) from None
    if c is None:
        raise AnalysisError("the session has no volume curve file; load one first")
    return c['t'], c['volume'], 0, sha256_file(folder.volume_curve_path)


def run_bagalkot(folder, D_range=None, moving=False, curve='log', progress=None,
                 is_cancelled=None):
    from core.bagalkot_estimator import (DEFAULT_D_RANGE, BagalkotCancelled,
                                         estimate_D_bagalkot)
    from core.session import VOLUME_CURVE_FILE
    ok, why = folder.bagalkot_availability(curve)
    if not ok:
        raise AnalysisError(why)
    D_range = tuple(D_range or DEFAULT_D_RANGE)
    if not 0 < D_range[0] < D_range[1]:
        raise AnalysisError("the D range needs 0 < minimum < maximum")
    t, V, n_skip, curve_sha = volume_curve(folder, curve)
    try:
        r = estimate_D_bagalkot(
            t, V, D_range=D_range, moving=bool(moving),
            progress_cb=(lambda i, n, D, F: progress(i, n, f"D = {D:.3g} m2/s"))
            if progress else None,
            is_cancelled=is_cancelled)
    except BagalkotCancelled:
        raise FrameSeriesCancelled() from None
    except ValueError as e:
        raise AnalysisError(str(e)) from None
    keys = ('D', 'D_per_min', 'SF', 'F_min', 'V0', 'radius_m', 'moving', 'D_near',
            'near_open', 'near_tol', 'at_edge', 'D_range', 'sf_converged', 'n_evaluations')
    data = dict(settings=dict(D_range=D_range, moving=bool(moving), curve=curve,
                              curve_sha256=curve_sha,
                              curve_file=VOLUME_CURVE_FILE if curve == 'file' else None),
                result={k: r[k] for k in keys},
                curve=dict(t=r['t'], V_exp=r['V_exp'], V_model=r['V_model']),
                sweep=dict(D=r['sweep_D'], F=r['sweep_F'], SF=r['sweep_SF']),
                n_skipped=n_skip)
    folder.save_analysis('bagalkot', data)
    return data


# ---------------------------------------------------------------------------
# Yang (IFT) -> D, k
# ---------------------------------------------------------------------------
def load_session_calibration(folder):
    """The session's gamma(c) curve as a core.calibration.Calibration."""
    from core.calibration import Calibration, CalibrationError, parse_calibration_template
    p = folder.calibration_path
    if p is None or not p.is_file():
        raise AnalysisError("the session has no gamma(c) calibration curve")
    try:
        d = parse_calibration_template(p)
        return Calibration.from_arrays(d['c'], d['gamma'], gamma0=float(d['gamma'][0]),
                                       gamma_inf=float(d['gamma'][-1]), c_sat=d['c_sat'],
                                       source=str(p))
    except (CalibrationError, ValueError) as e:
        raise AnalysisError(f"the stored calibration can't be used: {e}") from None


def build_model(folder, frame_index=None):
    """Domain, mesh and FEM system from frame ``frame_index`` (default: the
    first measured frame)."""
    from core.domain_builder import contour_to_domain
    from core.fem_assembly import assemble_fem_system
    from core.mesh_generator import generate_mesh
    from core.pendant_calcs import extract_pendant_edges
    row, img = folder.first_measured(frame_index)
    if row is None:
        raise AnalysisError(
            "no stored frame with a successful fit for the drop geometry" if frame_index is None
            else f"frame {frame_index} is not a stored frame with a successful fit")
    s = folder.session
    edges = extract_pendant_edges(img)
    if edges is None:
        raise AnalysisError(f"no drop outline in frame {row['frame_index']}")
    try:
        polygon, meta = contour_to_domain(edges, s.baseline['y_px'], s.scale['pixel_to_m'],
                                          needle_od_mm=s.needle['od_mm'],
                                          needle_id_mm=s.needle.get('id_mm'))
        points, triangles, boundary, quality = generate_mesh(polygon, domain_metadata=meta)
        fem = assemble_fem_system(points, triangles, boundary, meta)
    except AnalysisError:
        raise
    except Exception as e:                         # reported to the user, with its type
        raise AnalysisError(f"the model could not be built: {type(e).__name__}: {e}") from None
    return dict(frame=row, polygon=polygon, meta=meta, points=points, triangles=triangles,
                quality=quality, fem=fem)


def yang_curve(folder, curve):
    """(t, gamma, n_skipped, curve_sha256) of the IFT curve Yang fits:
    'log' = the logged frames that passed QC, 'file' = ift_curve.csv."""
    from core.session import SessionError, sha256_file
    if curve == 'log':
        t, g, n_skip = _series(folder, 'ift_mNm', qc_only=True)
        return t, g, n_skip, None
    if curve != 'file':
        raise AnalysisError(f"unknown IFT curve source {curve!r}")
    try:
        c = folder.ift_curve()
    except SessionError as e:
        raise AnalysisError(str(e)) from None
    if c is None:
        raise AnalysisError("the session has no IFT curve file; load one first")
    return c['t'], c['gamma'], 0, sha256_file(folder.ift_curve_path)


def model_summary(model):
    """What the Model tab shows: mesh, domain outline and statistics, and the
    FEM system's sanity checks (the Domain, Mesh and FEM windows)."""
    meta, fem, q, poly = model['meta'], model['fem'], model['quality'], model['polygon']
    s = fem['sanity']
    k_status = s.get('K_symmetry_status', 'fail')
    checks = [
        ('H symmetric', bool(s['H_symmetric']), f"error {s['H_symmetry_error']:.2e}"),
        ('K symmetric', k_status != 'fail',
         f"relative error {s['K_symmetry_error']:.2e} ({k_status})"),
        ('K conserves mass', bool(s['K_conservative']),
         f"max column sum {s['K_colsum_rel']:.1e}"),
        ('H diagonal >= 0', bool(s['H_diag_all_positive']), f"min {s['H_diag_min']:.2e}"),
        ('K diagonal > 0', bool(s['K_diag_positive']), f"min {s['K_diag_min']:.2e}"),
        ('Sparse (< 1 %)', bool(s['sparsity_ok']),
         f"H {s['H_sparsity_pct']:.3f} %, K {s['K_sparsity_pct']:.3f} %"),
        ('Boundary terms', bool(s['boundary_terms_ok']),
         f"Kb non-zeros {s['Kb_nnz']}, F non-zeros {s['F_nonzero']}"),
        ('F on interface only', bool(s['F_only_on_interface']),
         f"max {s['F_max']:.2e}, min > 0 {s['F_min_nonzero']:.2e}"),
    ]
    return dict(frame_index=int(model['frame']['frame_index']),
                frame_time_s=float(model['frame']['time_s']),
                n_nodes=int(q['n_nodes']), n_elements=int(q['n_elements']),
                min_angle=float(q['min_angle']), area_ratio=float(q.get('area_ratio', math.nan)),
                points=model['points'], triangles=model['triangles'], polygon=poly,
                domain=dict(n_points=int(meta['n_points']),
                            r_max_m=float(np.max(poly[:, 0])),
                            height_m=float(np.max(poly[:, 1])),
                            neck_height_m=float(meta['neck_height_m']),
                            drop_height_m=float(meta['drop_height_m']),
                            r_inner_m=float(meta['r_inner_m']),
                            r_outer_m=float(meta['r_outer_m']),
                            segment_indices={k: [int(a), int(b)] for k, (a, b)
                                             in meta.get('segment_indices', {}).items()}),
                fem=dict(size=int(fem['H'].shape[0]), H_nnz=int(fem['H'].nnz),
                         K_nnz=int(fem['K'].nnz), Kb_nnz=int(fem['Kb'].nnz),
                         n_interface_nodes=int(s['n_interface_nodes']),
                         n_wall_nodes=int(s['n_wall_nodes']),
                         n_interior_nodes=int(s['n_interior_nodes']),
                         checks=checks, all_pass=bool(s['all_pass'])))


def run_yang(folder, kD_range=None, D_bounds=None, t_offset=0.0, curve='log',
             frame_index=None, kD_values=None, D_grid=None, kD_spec=None, progress=None,
             is_cancelled=None):
    """Yang's fit.  ``kD_values`` given = his grid search (Fig 7a): the
    exact best D for each of those kD within ``D_grid`` = (D_min, D_max,
    points per curve); otherwise the automatic nested search (``kD_range`` /
    ``D_bounds`` pin it, else it widens itself).  Both store E(D) curves."""
    from core.estimator import EstimationCancelled, estimate_D_k_yang
    from core.session import IFT_CURVE_FILE
    ok, why = folder.yang_availability(curve)
    if not ok:
        raise AnalysisError(why)
    t, g, n_skip, curve_sha = yang_curve(folder, curve)
    if len(t) < YANG_MIN_POINTS:
        raise AnalysisError(f"the fit needs at least {YANG_MIN_POINTS} IFT points; "
                            f"this curve has {len(t)}")
    if np.any(t + t_offset < 0):
        raise AnalysisError(f"a time offset of {t_offset:g} s puts measurements before the "
                            f"drop formed; use a larger offset")
    for rng, name in ((kD_range, 'kD'), (D_bounds, 'D')):
        if rng is not None and not 0 < rng[0] < rng[1]:
            raise AnalysisError(f"the {name} range needs 0 < minimum < maximum")
    grid = kD_values is not None
    if grid:
        kD_values = sorted({float(v) for v in kD_values})
        if not kD_values or len(kD_values) > MAX_GRID_KD or any(
                not (math.isfinite(v) and v > 0) for v in kD_values):
            raise AnalysisError(f"give 1 to {MAX_GRID_KD} positive kD values")
        D_grid = tuple(D_grid or YANG_GRID_D)
        if not (0 < D_grid[0] < D_grid[1]) or not 5 <= int(D_grid[2]) <= 400:
            raise AnalysisError("the D axis needs 0 < D from < D to and 5 to 400 points")
        D_bounds, kD_range = (float(D_grid[0]), float(D_grid[1])), None
    calibration = load_session_calibration(folder)
    if progress:
        progress(0, 0, 'Building the model (geometry, mesh, FEM) from the drop outline')
    model = build_model(folder, frame_index)
    if is_cancelled is not None and is_cancelled():
        raise FrameSeriesCancelled()
    sims = [0]

    def should_cancel():
        sims[0] += 1
        if progress and sims[0] % 5 == 0:
            progress(sims[0], 0, f"{sims[0]} forward simulations")
        return bool(is_cancelled and is_cancelled())

    try:
        r = estimate_D_k_yang(model['fem'], model['meta'], t, g, calibration=calibration,
                              kD_range=kD_range, D_bounds=D_bounds,
                              auto_widen=not grid and kD_range is None and D_bounds is None,
                              t_offset=t_offset, should_cancel=should_cancel,
                              kD_values=kD_values, scan_D=False,
                              curve_points=int(D_grid[2]) if grid else CURVE_POINTS)
    except EstimationCancelled:
        raise FrameSeriesCancelled() from None
    except ValueError as e:
        raise AnalysisError(str(e)) from None
    keys = ('D', 'k', 'kD', 'E_percent', 'residual_rms', 'success', 'message',
            'n_evaluations', 'n_scan_evaluations', 'kD_range', 'D_bounds', 'widen_steps',
            'D_near', 'kD_near', 'near_tol', 'near_open', 'D_well_determined', 't_offset')
    data = dict(settings=dict(kD_range=None if grid else kD_range,
                              D_bounds=None if grid else D_bounds, t_offset=float(t_offset),
                              auto=not grid and kD_range is None and D_bounds is None,
                              search='grid' if grid else 'auto',
                              kD_values=kD_values, D_grid=list(D_grid) if grid else None,
                              kD_spec=kD_spec if grid else None,
                              curve=curve,
                              curve_sha256=curve_sha,
                              curve_file=IFT_CURVE_FILE if curve == 'file' else None,
                              needle_id_mm=folder.session.needle.get('id_mm'),
                              frame_index=None if frame_index is None else int(frame_index)),
                result={k: r[k] for k in keys},
                curve=dict(t=t, gamma_exp=g, gamma_fit=r['gamma_fit']),
                sweep=r['sweep_kD'],
                curves=r['curves'],
                model=model_summary(model),
                calibration=dict(c_sat=calibration.c_sat, gamma0=calibration.gamma0,
                                 gamma_inf=calibration.gamma_inf,
                                 n_points=len(calibration.c_points)),
                n_skipped=n_skip)
    folder.save_analysis('yang', data)
    # The concentration field and gamma(t) at the best fit, for the
    # Simulation tab (the Pendant window's Step 7 at the fitted D, k).
    try:
        run_simulation(folder, D=r['D'], k=r['k'], duration_s=float(np.max(t)) + t_offset,
                       frame_index=frame_index, model=model, source='yang',
                       progress=progress, is_cancelled=is_cancelled)
    except (FrameSeriesCancelled, AnalysisError):
        pass                                    # the fit itself is stored
    return data


# ---------------------------------------------------------------------------
# Forward simulation (the Pendant window's Step 7)
# ---------------------------------------------------------------------------
def simulation_defaults(folder):
    """(D, k, duration_s, frame_index): the stored Yang fit's values if
    there is one, else the Pendant window's fixed D, k and 10 min."""
    from core.session import SessionError
    try:
        doc = folder.load_analysis('yang')
        r, st, c = doc['data']['result'], doc['data']['settings'], doc['data']['curve']
        dur = float(np.max(np.asarray(c['t'], float))) + float(st.get('t_offset') or 0.0)
        return float(r['D']), float(r['k']), dur, doc['data']['model'].get('frame_index')
    except (SessionError, TypeError, KeyError, ValueError):
        return SIM_DEFAULT_D, SIM_DEFAULT_K, SIM_DEFAULT_DURATION_S, None


def _snapshot_steps(step_times, fractions=SIM_SNAPSHOT_FRACTIONS):
    """1-based step numbers closest to each fraction of the run."""
    t = np.asarray(step_times, float)
    return sorted({int(np.argmin(np.abs(t - f * t[-1]))) + 1 for f in fractions})


def run_simulation(folder, D=None, k=None, duration_s=None, dt_s=None, frame_index=None,
                   model=None, source='manual', progress=None, is_cancelled=None):
    """Forward diffusion simulation for (D, k) on the outline of one frame.

    ``dt_s`` None = graded steps (fine at first, as the Yang fit uses);
    a number = uniform steps of that size with theta = 0.7, as the Pendant
    window's Step 7.  gamma(t) needs the session's gamma(c) calibration;
    without one only the interface concentration Cs(t) is given.
    """
    from core.estimator import _rescale_fem_system
    from core.time_solver import graded_schedule, solve_diffusion
    if not folder.measured_frames():
        raise AnalysisError("no stored frame with a successful fit for the drop geometry")
    D0, k0, dur0, _ = simulation_defaults(folder)
    D = D0 if D is None else float(D)
    k = k0 if k is None else float(k)
    duration_s = dur0 if duration_s is None else float(duration_s)
    for v, name in ((D, 'D'), (k, 'k'), (duration_s, 'the duration')):
        if not (math.isfinite(v) and v > 0):
            raise AnalysisError(f"{name} must be a positive number")
    if dt_s is not None:
        dt_s = float(dt_s)
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise AnalysisError("the time step must be a positive number")
        if duration_s / dt_s > 2_000_000:
            raise AnalysisError(f"{duration_s / dt_s:.0f} steps of {dt_s:g} s is too many; "
                                f"use a larger time step or automatic steps")
    calibration = None
    if folder.calibration_path and folder.calibration_path.is_file():
        calibration = load_session_calibration(folder)
    if model is None:
        if progress:
            progress(0, 0, 'Building the model (geometry, mesh, FEM) from the drop outline')
        model = build_model(folder, frame_index)
    r_n = model['meta']['r_inner_m']
    if dt_s is None:
        dts = graded_schedule(duration_s, D, r_n)[0]
    else:
        dts = np.full(max(1, round(duration_s / dt_s)), dt_s)
    step_t = np.cumsum(dts)
    every = max(1, len(dts) // 200)

    def on_step(step, n, C, Cs):
        if is_cancelled is not None and is_cancelled():
            raise FrameSeriesCancelled()
        if progress and (step % every == 0 or step == n):
            progress(step, n, f"Simulating: step {step} of {n}")

    try:
        sim = solve_diffusion(_rescale_fem_system(model['fem'], D, k), model['meta'], D=D, k=k,
                              theta=0.7, dtau=dt_s if dt_s is not None else 0.01,
                              total_time=duration_s,
                              stepping='uniform' if dt_s is not None else 'graded',
                              snapshot_steps=_snapshot_steps(step_t), progress_cb=on_step,
                              calibration=calibration, verbose=False)
    except FrameSeriesCancelled:
        raise
    except Exception as e:                         # reported to the user, with its type
        raise AnalysisError(f"the simulation failed: {type(e).__name__}: {e}") from None
    t = np.asarray(sim['time_history'], float)
    Cs = np.asarray(sim['Cs_history'], float)
    gamma = np.asarray(sim['gamma_history'], float) if calibration is not None else None
    C_final = np.asarray(sim['C_final'], float)
    diag = sim['diagnostics']
    if len(diag) > SIM_LOG_LINES:
        head, tail = diag[:20], diag[-(SIM_LOG_LINES - 20):]
        diag = head + [f"... {len(diag) - len(head) - len(tail)} steps not shown ..."] + tail
    data = dict(settings=dict(D=D, k=k, duration_s=duration_s, dt_s=dt_s,
                              stepping='uniform' if dt_s is not None else 'graded',
                              frame_index=int(model['frame']['frame_index']), source=source,
                              needle_id_mm=folder.session.needle.get('id_mm')),
                result=dict(kD=float(sim['kD']), n_steps=int(sim['n_steps']),
                            simulated_s=float(sim['actual_time']), Cs_final=float(Cs[-1]),
                            gamma_final=None if gamma is None else float(gamma[-1]),
                            gamma_range=None if gamma is None else [float(gamma.min()),
                                                                    float(gamma.max())],
                            C_min=float(sim['C_min']), C_max=float(sim['C_max']),
                            C_final_min=float(C_final.min()), C_final_max=float(C_final.max()),
                            has_nan=bool(np.any(np.isnan(C_final))),
                            has_negative=bool(np.any(C_final < 0)),
                            calibrated=calibration is not None),
                curve=dict(t=t, Cs=Cs, gamma=gamma),
                snapshots=[dict(step=int(n), t=float(step_t[n - 1]),
                                C=np.round(sim['snapshots'][n], 6))
                           for n in sorted(sim['snapshots'])],
                model=model_summary(model),
                measured=_measured_ift(folder),
                diagnostics=diag)
    folder.save_analysis('simulation', data)
    return data


def _measured_ift(folder):
    """The IFT curve the stored Yang fit used, on the model's time axis, or None."""
    from core.session import SessionError
    try:
        doc = folder.load_analysis('yang')
        c, st = doc['data']['curve'], doc['data']['settings']
        return dict(t=np.asarray(c['t'], float) + float(st.get('t_offset') or 0.0),
                    gamma=c['gamma_exp'])
    except (SessionError, TypeError, KeyError, ValueError):
        return None


RUNNERS = dict(swelling=run_swelling, bagalkot=run_bagalkot, yang=run_yang,
               simulation=run_simulation)
