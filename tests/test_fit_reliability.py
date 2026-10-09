"""
tests.test_fit_reliability
==========================

Gates for the fit-reliability changes:

1. Graded time stepping: schedule is well-formed, matches a very fine
   reference run, is converged (refining it changes little), and is more
   accurate at early times than the old uniform default.
2. No clipping: concentrations are reported (C_min, C_max), not altered.
3. Time offset: data recorded on a clock that starts after drop formation
   are fitted correctly when the offset is given; a negative model time is
   rejected.
4. "Equally good fits" range: contains the best answer and the truth on a
   well-posed synthetic case, and is reported consistently.

Run:  conda run -n ift python tests/test_fit_reliability.py
"""

import time

import numpy as np

from _synthetic_domain import get_fem_system
from core.estimator import (_rescale_fem_system, estimate_D_k_yang,
                            simulate_ift)
from core.time_solver import graded_schedule, solve_diffusion

D_TRUE, K_TRUE = 1.0e-9, 1.0e-5
T = 150.0


def _cs(fem, meta, D, k, **kw):
    """Interface concentration history including the known start (0, 0)."""
    s = solve_diffusion(_rescale_fem_system(fem, D, k), meta, D=D, k=k,
                        total_time=T, snapshot_steps=[], verbose=False, **kw)
    return (np.r_[0.0, s['time_history']], np.r_[0.0, s['Cs_history']], s)


def test_schedule_shape():
    fem, meta = get_fem_system()
    r_n = meta['r_inner_m']
    for D in (1e-11, 1e-9, 1e-7):
        dt, th = graded_schedule(T, D, r_n)
        tau0 = dt[0] * D / r_n ** 2
        print(f"  D={D:.0e}: {len(dt)} steps, {len(set(zip(dt, th)))} distinct (dt, theta), "
              f"first dtau={tau0:.4g}, theta {th[0]:.3f}..{th[-1]:.3f}")
        assert abs(dt.sum() - T) < 1e-9 * T
        assert np.all(dt > 0) and np.all(np.diff(th) >= -1e-15)
        assert abs(th[0] - 2.0 / 3.0) < 1e-12 and th.max() <= 1.0
        assert len(set(zip(dt, th))) < 45           # few factorisations
        assert len(dt) >= 200


def test_graded_accuracy_and_convergence():
    fem, meta = get_fem_system()
    t_q = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 150])
    tr, cr, _ = _cs(fem, meta, D_TRUE, K_TRUE, stepping='uniform', dtau=0.005, theta=0.5)
    ref = np.interp(t_q, tr, cr)
    tg, cg, sg = _cs(fem, meta, D_TRUE, K_TRUE, stepping='graded')
    tu, cu, _ = _cs(fem, meta, D_TRUE, K_TRUE, stepping='uniform', dtau=T / 200, theta=0.7)
    err_g = np.abs(np.interp(t_q, tg, cg) - ref).max()
    err_u = np.abs(np.interp(t_q, tu, cu) - ref).max()
    print(f"  max |Cs - reference|: graded {err_g:.2e}  (old uniform default {err_u:.2e}); "
          f"graded steps {sg['n_steps']}")
    assert err_g < 2e-3
    assert err_g < err_u

    # Convergence: half the first step and twice the step count.
    import core.time_solver as ts
    r_n = meta['r_inner_m']
    dt, th = graded_schedule(T, D_TRUE, r_n, dtau0=ts.GRADED_DTAU0 / 2,
                             min_steps=2 * ts.GRADED_MIN_STEPS)
    fem_s = _rescale_fem_system(fem, D_TRUE, K_TRUE)
    orig = ts.graded_schedule
    ts.graded_schedule = lambda *a, **k: (dt, th)
    try:
        s = solve_diffusion(fem_s, meta, D=D_TRUE, k=K_TRUE, total_time=T, stepping='graded',
                            snapshot_steps=[], verbose=False)
    finally:
        ts.graded_schedule = orig
    d = np.abs(np.interp(t_q, np.r_[0.0, s['time_history']], np.r_[0.0, s['Cs_history']])
               - np.interp(t_q, tg, cg)).max()
    print(f"  refining the graded schedule changes Cs by {d:.2e}")
    assert d < 1e-3


def test_no_clipping():
    fem, meta = get_fem_system()
    _, _, s = _cs(fem, meta, D_TRUE, K_TRUE, stepping='graded')
    print(f"  C range over the run: [{s['C_min']:.3e}, {s['C_max']:.6f}]")
    assert 'C_min' in s and 'C_max' in s and s['stepping'] == 'graded'
    assert s['C_max'] < 1.0 + 1e-3 and s['C_min'] > -0.05


def test_time_offset():
    fem, meta = get_fem_system()
    offset = 3.0
    t_clock = np.linspace(2.0, T - offset, 30)          # clock starts 3 s late
    g = simulate_ift(fem, meta, D_TRUE, K_TRUE, t_clock + offset, total_time=T)
    kw = dict(n_kD=12, scan_D=False, refine=True)
    r = estimate_D_k_yang(fem, meta, t_clock, g, t_offset=offset, **kw)
    r0 = estimate_D_k_yang(fem, meta, t_clock, g, **kw)
    kD_true = K_TRUE * meta['r_inner_m'] / D_TRUE
    print(f"  with offset: D {r['D'] / D_TRUE - 1:+.1%}, kD {r['kD'] / kD_true - 1:+.1%}, "
          f"E {r['E_percent']:.3f}%;  without: D {r0['D'] / D_TRUE - 1:+.1%}, "
          f"kD {r0['kD'] / kD_true - 1:+.1%}, E {r0['E_percent']:.3f}%")
    assert abs(r['D'] / D_TRUE - 1) < 0.10 and abs(r['kD'] / kD_true - 1) < 0.10
    assert r['E_percent'] < r0['E_percent'] and r['t_offset'] == offset
    try:
        estimate_D_k_yang(fem, meta, t_clock, g, t_offset=-5.0, **kw)
    except ValueError as e:
        print(f"  negative model time rejected: {e}")
    else:
        raise AssertionError("negative model time was accepted")


def test_one_sim_per_kD_matches_direct():
    """The τ-rescaling fast path gives the same γ as a direct simulation
    at that (D, k), and the fit uses one simulation per kD."""
    from core.estimator import simulate_ift as sim_direct
    fem, meta = get_fem_system()
    r_n = meta['r_inner_m']
    t_exp = np.linspace(2.0, T, 40)
    g = sim_direct(fem, meta, D_TRUE, K_TRUE, t_exp, total_time=T)
    t0 = time.perf_counter()
    r = estimate_D_k_yang(fem, meta, t_exp, g, n_kD=12, scan_D=False, refine=True)
    t_fast = time.perf_counter() - t0
    n_kD_tried = len(r['sweep_kD']['kD'])
    print(f"  fit: {r['n_evaluations']} simulations for {n_kD_tried} kD values, {t_fast:.1f} s; "
          f"D {r['D'] / D_TRUE - 1:+.2%}, kD {r['kD'] / (K_TRUE * r_n / D_TRUE) - 1:+.2%}")
    assert r['n_evaluations'] <= n_kD_tried + 2
    # Fitted curve vs a direct simulation at the fitted (D, k).
    g_direct = sim_direct(fem, meta, r['D'], r['k'], t_exp, total_time=T)
    d = np.abs(np.asarray(r['gamma_fit']) - g_direct).max()
    print(f"  fast-path curve vs direct simulation: max |dgamma| = {d:.4f} mN/m")
    # Both paths carry ~0.01–0.04 mN/m discretisation error (the fast one
    # the smaller); the data here are generated by the direct path.
    assert d < 0.05
    assert abs(r['D'] / D_TRUE - 1) < 0.05


def test_near_range():
    fem, meta = get_fem_system()
    t_exp = np.linspace(2.0, T, 40)
    g = simulate_ift(fem, meta, D_TRUE, K_TRUE, t_exp, total_time=T)
    g = g + np.random.default_rng(7).normal(0.0, 0.05, t_exp.size)
    t0 = time.perf_counter()
    r = estimate_D_k_yang(fem, meta, t_exp, g, n_kD=12, scan_D=False, refine=True)
    lo, hi = r['D_near']
    print(f"  D = {r['D']:.3e}, equally good fits D {lo:.3e}..{hi:.3e} ({hi / lo:.2f}x), "
          f"kD {r['kD_near'][0]:.2f}..{r['kD_near'][1]:.2f}, open={r['near_open']}, "
          f"well determined={r['D_well_determined']}  ({time.perf_counter() - t0:.0f} s)")
    assert lo <= r['D'] <= hi and r['kD_near'][0] <= r['kD'] <= r['kD_near'][1]
    assert lo <= D_TRUE * 1.15 and hi >= D_TRUE * 0.85
    assert r['D_well_determined']


if __name__ == "__main__":
    for name, fn in [("graded schedule shape", test_schedule_shape),
                     ("graded accuracy + convergence", test_graded_accuracy_and_convergence),
                     ("no clipping", test_no_clipping),
                     ("time offset", test_time_offset),
                     ("one simulation per kD", test_one_sim_per_kD_matches_direct),
                     ("equally-good-fits range", test_near_range)]:
        print(f"[{name}]")
        fn()
        print("  PASS")
    print("All checks passed.")
