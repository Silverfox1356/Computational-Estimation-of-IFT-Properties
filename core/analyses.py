"""
core.analyses
=============

The analyses that run on a session folder (core.session.SessionFolder),
each reading only what the folder holds and storing its settings and
result in ``analyses/<name>.json`` (stamped with the log it used, so a
rebuilt log marks it out of date):

    swelling   core.swelling.swelling_check on V(t) from the log; recommends
               Yang (drop static enough) or Bagalkot (it swells)
    bagalkot   core.bagalkot_estimator.estimate_D_bagalkot on V(t): D only
    yang       Yang et al. 2006 on IFT(t): the drop geometry from the first
               measured frame -> domain -> mesh -> FEM, the gamma(c)
               calibration stored in the session, then estimate_D_k_yang

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
              yang='Diffusivity (Yang)')


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
def run_bagalkot(folder, D_range=None, moving=False, progress=None, is_cancelled=None):
    from core.bagalkot_estimator import (DEFAULT_D_RANGE, BagalkotCancelled,
                                         estimate_D_bagalkot)
    _check(folder, 'bagalkot')
    D_range = tuple(D_range or DEFAULT_D_RANGE)
    if not 0 < D_range[0] < D_range[1]:
        raise AnalysisError("the D range needs 0 < minimum < maximum")
    t, V, n_skip = _series(folder, 'volume_uL')
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
    data = dict(settings=dict(D_range=D_range, moving=bool(moving)),
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


def build_model(folder):
    """Domain, mesh and FEM system from the first measured frame."""
    from core.domain_builder import contour_to_domain
    from core.fem_assembly import assemble_fem_system
    from core.mesh_generator import generate_mesh
    from core.pendant_calcs import extract_pendant_edges
    row, img = folder.first_measured()
    if row is None:
        raise AnalysisError("no stored frame with a successful fit for the drop geometry")
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


def run_yang(folder, kD_range=None, D_bounds=None, t_offset=0.0, progress=None,
             is_cancelled=None):
    from core.estimator import EstimationCancelled, estimate_D_k_yang
    _check(folder, 'yang')
    t, g, n_skip = _series(folder, 'ift_mNm', qc_only=True)
    if len(t) < YANG_MIN_POINTS:
        raise AnalysisError(f"the fit needs at least {YANG_MIN_POINTS} frames that passed "
                            f"QC; this session has {len(t)}")
    if np.any(t + t_offset < 0):
        raise AnalysisError(f"a time offset of {t_offset:g} s puts measurements before the "
                            f"drop formed; use a larger offset")
    for rng, name in ((kD_range, 'kD'), (D_bounds, 'D')):
        if rng is not None and not 0 < rng[0] < rng[1]:
            raise AnalysisError(f"the {name} range needs 0 < minimum < maximum")
    calibration = load_session_calibration(folder)
    if progress:
        progress(0, 0, 'Building the model from the first measured frame')
    model = build_model(folder)
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
                              auto_widen=kD_range is None and D_bounds is None,
                              t_offset=t_offset, should_cancel=should_cancel)
    except EstimationCancelled:
        raise FrameSeriesCancelled() from None
    except ValueError as e:
        raise AnalysisError(str(e)) from None
    keys = ('D', 'k', 'kD', 'E_percent', 'residual_rms', 'success', 'message',
            'n_evaluations', 'n_scan_evaluations', 'kD_range', 'D_bounds', 'widen_steps',
            'D_near', 'kD_near', 'near_tol', 'near_open', 'D_well_determined', 't_offset')
    q = model['quality']
    data = dict(settings=dict(kD_range=kD_range, D_bounds=D_bounds, t_offset=float(t_offset),
                              auto=kD_range is None and D_bounds is None),
                result={k: r[k] for k in keys},
                curve=dict(t=t, gamma_exp=g, gamma_fit=r['gamma_fit']),
                sweep=r['sweep_kD'],
                model=dict(frame_index=int(model['frame']['frame_index']),
                           frame_time_s=float(model['frame']['time_s']),
                           n_nodes=int(q['n_nodes']), n_elements=int(q['n_elements']),
                           min_angle=float(q['min_angle']),
                           points=model['points'], triangles=model['triangles'],
                           polygon=model['polygon']),
                calibration=dict(c_sat=calibration.c_sat, gamma0=calibration.gamma0,
                                 gamma_inf=calibration.gamma_inf,
                                 n_points=len(calibration.c_points)),
                n_skipped=n_skip)
    folder.save_analysis('yang', data)
    return data


RUNNERS = dict(swelling=run_swelling, bagalkot=run_bagalkot, yang=run_yang)
