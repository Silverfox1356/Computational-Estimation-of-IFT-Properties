"""
tests.test_estimator_synthetic
==============================

Phase-A acceptance gate: synthetic parameter recovery.

Generate a "measured" γ_exp(t) from the forward model itself with known
(D*, k*), add small noise, and confirm ``estimate_D_k`` recovers the
true values from a deliberately wrong starting point.  If this fails,
the inverse machinery is wrong — independent of any physics or
calibration question.

Run:  python tests/test_estimator_synthetic.py
(Plain script on purpose — the ift env has no pytest.)
"""

import time

import numpy as np

from _synthetic_domain import get_fem_system
from core.estimator import estimate_D_k, simulate_ift

# Ground truth (Yang-2006-plausible magnitudes) and experiment window.
D_TRUE = 1.0e-9        # m²/s
K_TRUE = 1.0e-5        # m/s
TOTAL_TIME = 200.0     # s
NOISE_MN_M = 0.05      # γ noise std-dev, mN/m (~digitisation scale)
TOLERANCE = 0.05       # recovered value within 5 % of truth


def test_synthetic_recovery():
    fem_results, metadata = get_fem_system()

    # Synthetic experiment: 60 samples, small Gaussian noise.
    t_exp = np.linspace(2.0, TOTAL_TIME, 60)
    dtau = TOTAL_TIME / 200.0
    gamma_true = simulate_ift(fem_results, metadata, D_TRUE, K_TRUE,
                              t_exp, dtau=dtau, total_time=TOTAL_TIME)
    rng = np.random.default_rng(42)
    gamma_exp = gamma_true + rng.normal(0.0, NOISE_MN_M, t_exp.size)

    # Start ~3× off in both parameters.
    t0 = time.perf_counter()
    result = estimate_D_k(fem_results, metadata, t_exp, gamma_exp,
                          initial_guess=(3e-9, 3e-6))
    elapsed = time.perf_counter() - t0

    D_err = abs(result['D'] - D_TRUE) / D_TRUE
    k_err = abs(result['k'] - K_TRUE) / K_TRUE

    print(f"  true    : D = {D_TRUE:.3e}  k = {K_TRUE:.3e}")
    print("  start   : D = 3.000e-09  k = 3.000e-06")
    print(f"  found   : D = {result['D']:.3e}  k = {result['k']:.3e}")
    print(f"  error   : D {100 * D_err:.2f} %   k {100 * k_err:.2f} %")
    print(f"  rms     : {result['residual_rms']:.4f} mN/m "
          f"({result['n_evaluations']} forward sims, {elapsed:.1f} s)")
    print(f"  status  : {result['message']}")

    assert result['success'], f"optimiser failed: {result['message']}"
    assert D_err < TOLERANCE, f"D off by {100 * D_err:.1f} % (> 5 %)"
    assert k_err < TOLERANCE, f"k off by {100 * k_err:.1f} % (> 5 %)"


if __name__ == '__main__':
    print("── Phase A synthetic-recovery test ──")
    test_synthetic_recovery()
    print("PASS")
