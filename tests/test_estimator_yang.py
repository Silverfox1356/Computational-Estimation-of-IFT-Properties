"""
tests.test_estimator_yang
=========================

Acceptance gate for the Yang et al. (2006) nested one-parameter search
(``estimate_D_k_yang``).  Generate a synthetic γ_exp(t) from the forward
model with known (D*, k*), then confirm the nested kD-sweep recovers both
the diffusion coefficient D and the Biot number kD = k·rₙ/D.

Also checks that the E-vs-kD / E-vs-D sweep arrays (the paper's Figure 7)
come back well-formed with a genuine interior minimum.

Run:  conda run -n ift python tests/test_estimator_yang.py
"""

import time

import numpy as np

from _synthetic_domain import get_fem_system
from core.estimator import (estimate_D_k_yang, estimate_D_k, simulate_ift,
                            EstimationCancelled)

D_TRUE = 1.0e-9        # m²/s
K_TRUE = 1.0e-5        # m/s
TOTAL_TIME = 150.0     # s
NOISE_MN_M = 0.05


def test_yang_recovery():
    fem_results, metadata = get_fem_system()
    r_n = float(metadata['r_inner_m'])
    kD_true = K_TRUE * r_n / D_TRUE

    t_exp = np.linspace(2.0, TOTAL_TIME, 40)
    dtau = TOTAL_TIME / 200.0
    gamma_true = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                              t_exp, dtau=dtau, total_time=TOTAL_TIME)
    rng = np.random.default_rng(7)
    gamma_exp = gamma_true + rng.normal(0.0, NOISE_MN_M, t_exp.size)

    t0 = time.perf_counter()
    res = estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                            n_kD=12, n_D_scan=8, refine=True)
    elapsed = time.perf_counter() - t0

    D_err = abs(res['D'] - D_TRUE) / D_TRUE
    kD_err = abs(res['kD'] - kD_true) / kD_true

    print(f"  true    : D = {D_TRUE:.3e}  k = {K_TRUE:.3e}  kD = {kD_true:.3f}")
    print(f"  found   : D = {res['D']:.3e}  k = {res['k']:.3e}  kD = {res['kD']:.3f}")
    print(f"  error   : D {100*D_err:.2f} %   kD {100*kD_err:.2f} %")
    print(f"  E_min   : {res['E_percent']:.3f} %   "
          f"rms {res['residual_rms']:.4f} mN/m")
    print(f"  effort  : {res['n_evaluations']} forward sims, {elapsed:.1f} s")
    print(f"  status  : {res['message']}")

    # Sweep arrays present and consistent with the reported optimum.
    swk, swd = res['sweep_kD'], res['sweep_D']
    assert len(swk['kD']) == len(swk['E']) == len(swk['D']) and len(swk['kD']) >= 12
    assert len(swd['D']) == len(swd['E']) == 8
    assert np.isclose(swk['E'].min(), res['E_percent'], rtol=1e-6), \
        "reported E_percent should equal the sweep minimum"

    assert res['success'], f"minimum not interior: {res['message']}"
    # kD (the shape parameter) is the better-constrained of the two; D is
    # weakly identifiable, so its tolerance is looser.  The coarse general
    # grid (n_kD=12 over a wide range) is what a hands-off user gets.
    assert kD_err < 0.15, f"kD off by {100*kD_err:.1f} %"
    assert D_err < 0.15, f"D off by {100*D_err:.1f} %"


def test_auto_widen():
    """A too-narrow starting kD range (true kD=5 sits above hi=0.5) must
    be auto-widened until the interior minimum is found."""
    fem_results, metadata = get_fem_system()
    r_n = float(metadata['r_inner_m'])
    kD_true = K_TRUE * r_n / D_TRUE

    t_exp = np.linspace(2.0, TOTAL_TIME, 30)
    dtau = TOTAL_TIME / 200.0
    gamma_exp = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                             t_exp, dtau=dtau, total_time=TOTAL_TIME)

    res = estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                            kD_range=(0.01, 0.5), n_kD=6, n_D_scan=4,
                            refine=False)
    print(f"  widen   : {res['widen_steps']} step(s) → "
          f"kD_range {tuple(round(x,3) for x in res['kD_range'])}, "
          f"kD={res['kD']:.3f} (true {kD_true:.3f}), success={res['success']}")

    # This test verifies the widening *mechanism* (trigger, bound growth,
    # reaching an interior minimum) — not precision: it uses a tiny 6-point
    # grid with refine off, so the recovered kD is only ballpark.
    assert res['widen_steps'] >= 1, "should have auto-widened the kD range"
    assert res['kD_range'][1] > 0.5, "hi kD bound should have expanded"
    assert res['success'], f"still on edge: {res['message']}"
    assert res['kD'] > 0.5, "recovered kD should be past the original edge"


def test_D_edge_is_detected():
    """A D box that EXCLUDES the true D must be reported as an edge hit.

    Regression guard.  The edge test used to be `|log10 D − bound| < 1e-3`,
    ten times tighter than the inner search's own xatol — and scipy's
    bounded Brent never returns a point exactly on a bound, so the test
    could never fire.  A D pinned to the search box was reported as
    "interior minimum" and never auto-widened, which is precisely the
    Yang-validation failure mode.  With auto_widen off the run must now
    admit the edge; with it on, widening must rescue the fit.

    The kD range is pinned around the true value so this isolates D.
    """
    fem_results, metadata = get_fem_system()
    r_n = float(metadata['r_inner_m'])
    kD_true = K_TRUE * r_n / D_TRUE

    t_exp = np.linspace(2.0, TOTAL_TIME, 30)
    dtau = TOTAL_TIME / 200.0
    gamma_exp = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                             t_exp, dtau=dtau, total_time=TOTAL_TIME)

    # True D = 1e-9 sits an order of magnitude ABOVE this box, so the inner
    # search can only pin to its upper bound.
    pinned = dict(kD_range=(0.5 * kD_true, 2.0 * kD_true),
                  D_bounds=(1e-12, 1e-10), n_kD=5, n_D_scan=4, refine=False)

    res = estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                            auto_widen=False, **pinned)
    print(f"  pinned  : D={res['D']:.3e} (box {res['D_bounds'][0]:.0e}–"
          f"{res['D_bounds'][1]:.0e}), success={res['success']}")
    assert not res['success'], \
        "D pinned to the box edge must NOT be reported as an interior minimum"
    assert 'D bounds' in res['message'], \
        f"message should name the D bounds, got: {res['message']}"

    # Same box, widening allowed: the D bound must actually grow.
    res_w = estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                              auto_widen=True, **pinned)
    print(f"  widened : D={res_w['D']:.3e} (box {res_w['D_bounds'][0]:.0e}–"
          f"{res_w['D_bounds'][1]:.0e}), {res_w['widen_steps']} step(s), "
          f"success={res_w['success']}")
    assert res_w['widen_steps'] >= 1, "should have auto-widened the D bounds"
    assert res_w['D_bounds'][1] > 1e-10, "upper D bound should have expanded"

    # An interior minimum must still read as interior (no false positives).
    res_ok = estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                               kD_range=(0.5 * kD_true, 2.0 * kD_true),
                               D_bounds=(1e-11, 1e-8), n_kD=5, n_D_scan=4,
                               refine=False, auto_widen=False)
    print(f"  interior: D={res_ok['D']:.3e}, success={res_ok['success']}")
    assert res_ok['success'], \
        f"interior minimum wrongly flagged as an edge: {res_ok['message']}"


def test_cancellation():
    """``should_cancel`` must unwind the sweep instead of running it out."""
    fem_results, metadata = get_fem_system()
    t_exp = np.linspace(2.0, TOTAL_TIME, 20)
    dtau = TOTAL_TIME / 200.0
    gamma_exp = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                             t_exp, dtau=dtau, total_time=TOTAL_TIME)

    calls = [0]

    def cancel_after_3():
        calls[0] += 1
        return calls[0] > 3

    t0 = time.perf_counter()
    try:
        estimate_D_k_yang(fem_results, metadata, t_exp, gamma_exp,
                          n_kD=25, n_D_scan=25,
                          should_cancel=cancel_after_3)
    except EstimationCancelled:
        print(f"  cancel  : unwound after {calls[0]} objective call(s), "
              f"{time.perf_counter() - t0:.2f} s")
    else:
        raise AssertionError("should_cancel was ignored — the sweep ran on")
    assert calls[0] <= 5, f"kept simulating after cancel ({calls[0]} calls)"


def test_rejects_unusable_gamma():
    """Non-positive / non-finite IFT must be refused, not silently fit.

    The default objective divides by γ_exp, so a failed drop-shape fit
    reporting γ ≤ 0 would poison every residual.
    """
    fem_results, metadata = get_fem_system()
    t_exp = np.linspace(2.0, TOTAL_TIME, 20)
    dtau = TOTAL_TIME / 200.0
    gamma_ok = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                            t_exp, dtau=dtau, total_time=TOTAL_TIME)

    for label, bad_value in (("zero", 0.0), ("negative", -3.0),
                             ("NaN", np.nan)):
        gamma_bad = gamma_ok.copy()
        gamma_bad[5] = bad_value
        for fn, kwargs in ((estimate_D_k_yang, dict(n_kD=3, n_D_scan=3)),
                           (estimate_D_k, {})):
            try:
                fn(fem_results, metadata, t_exp, gamma_bad, **kwargs)
            except ValueError as e:
                msg = str(e)
            else:
                raise AssertionError(
                    f"{fn.__name__} accepted a {label} IFT value")
            assert 'IFT' in msg, f"unhelpful message: {msg}"
        print(f"  reject  : {label} IFT -> {msg[:58]}…")


def test_initial_guess_outside_bounds():
    """estimate_D_k must clip x0 into the box, not crash.

    least_squares raises "Initial guess is outside of provided bounds" for
    an infeasible x0 — which the default guess is whenever a caller
    narrows `bounds` around it.
    """
    fem_results, metadata = get_fem_system()
    t_exp = np.linspace(2.0, TOTAL_TIME, 20)
    dtau = TOTAL_TIME / 200.0
    gamma_exp = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                             t_exp, dtau=dtau, total_time=TOTAL_TIME)

    # Default initial_guess (1e-9, 1e-5) lies outside both these ranges.
    res = estimate_D_k(fem_results, metadata, t_exp, gamma_exp,
                       bounds=((1e-12, 1e-10), (1e-7, 1e-6)))
    print(f"  clipped : D={res['D']:.3e} k={res['k']:.3e} "
          f"(ran {res['n_evaluations']} sims)")
    assert 1e-12 <= res['D'] <= 1e-10
    assert 1e-7 <= res['k'] <= 1e-6


if __name__ == '__main__':
    print("── Yang nested (D, kD) search test ──")
    test_yang_recovery()
    print("── auto-widen test ──")
    test_auto_widen()
    print("── D edge-detection test ──")
    test_D_edge_is_detected()
    print("── cancellation test ──")
    test_cancellation()
    print("── unusable-gamma rejection test ──")
    test_rejects_unusable_gamma()
    print("── initial-guess clipping test ──")
    test_initial_guess_outside_bounds()
    print("PASS")
