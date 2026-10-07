"""
Regression gate for the Step-3 drop-shape fit (ADSA-P).

Fits synthetic drops with known β and R₀ (tests/_synthetic_drop_image.py)
through the app's own pipeline and checks accuracy, speed and cancellation.
Thresholds sit ~3–5x above what the fit achieved when they were set
(σ error ≤ 0.12 %, ≤ 0.8 s per fit), so they catch regressions, not noise.

Run from the repo root:
    conda run -n ift python tests/test_adsa_fit.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.image_processing import attempt_auto_baseline
from core.pendant_calcs import extract_pendant_edges
from core.physics_calcs import calculate_physics
from _synthetic_drop_image import render_drop

P2M, RHO_IN, RHO_OUT, G = 1e-5, 1045.0, 160.8, 9.80665
MAX_SIGMA_ERR_PCT = 0.5
MAX_TIME_S = 5.0
MAX_RMSE_PX = 0.2

failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def fit(beta, R0_px, noise, **kw):
    img, truth = render_drop(beta, R0_px, noise_std=noise)
    edges = extract_pendant_edges(img)
    base = attempt_auto_baseline(edges, truth['needle_width_px'])
    t0 = time.perf_counter()
    r = calculate_physics(edges, base, P2M, density_inner=RHO_IN,
                          density_outer=RHO_OUT, g=G, **kw)
    return r, time.perf_counter() - t0


print("Accuracy and speed on synthetic drops")
for beta, R0_px, noise in [(0.20, 150, 0.0), (0.30, 400, 4.0), (0.40, 150, 4.0)]:
    r, dt = fit(beta, R0_px, noise)
    tag = f"beta={beta} R0={R0_px}px noise={noise}"
    if not r.get('fit_success'):
        check(False, f"{tag}: fit failed ({r.get('error')})")
        continue
    sigma_true = G * (RHO_IN - RHO_OUT) * (R0_px * P2M) ** 2 / beta * 1e3
    err = 100 * abs(r['ift_mNm'] / sigma_true - 1)
    check(err <= MAX_SIGMA_ERR_PCT, f"{tag}: sigma error {err:.3f} % (<= {MAX_SIGMA_ERR_PCT})")
    check(dt <= MAX_TIME_S, f"{tag}: fit time {dt:.2f} s (<= {MAX_TIME_S})")
    check(r['rmse_px'] <= MAX_RMSE_PX, f"{tag}: RMSE {r['rmse_px']:.3f} px (<= {MAX_RMSE_PX})")
    check(r['qc_pass'], f"{tag}: QC pass")

print("Stopping")
r, dt = fit(0.30, 150, 0.0, should_cancel=lambda: True)
check(not r['fit_success'] and r['error'] == "Fit stopped: cancelled",
      f"cancel request stops the fit ({r.get('error')}, {dt:.3f} s)")
r, dt = fit(0.30, 150, 0.0, time_limit_s=0.0)
check(not r['fit_success'] and 'time limit' in r['error'],
      f"time limit stops the fit ({r.get('error')})")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All checks passed.")
