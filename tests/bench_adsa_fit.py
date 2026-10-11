"""
Benchmark the Step-3 drop-shape fit (ADSA-P) against synthetic drops with
known β and R₀.  Not a pass/fail test — prints an accuracy/speed table.

Run from the repo root:
    conda run -n ift python tests/bench_adsa_fit.py [--quick] [--csv out.csv]

The pipeline under test is exactly the app's: extract_pendant_edges →
attempt_auto_baseline → calculate_physics.  σ error equals the combined
R₀²/β error, since g, Δρ and the pixel scale are held exact.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.image_processing import attempt_auto_baseline
from core.pendant_calcs import extract_pendant_edges
from core.physics_calcs import calculate_physics
from _synthetic_drop_image import render_drop

PIXEL_TO_M = 1e-5                      # exact by construction
RHO_IN, RHO_OUT, G = 1045.0, 160.8, 9.80665

FULL = dict(beta=[0.20, 0.30, 0.40], R0_px=[150, 400],
            tilt_deg=[0.0, 1.0], noise_std=[0.0, 4.0])
QUICK = dict(beta=[0.20, 0.40], R0_px=[150, 400],
             tilt_deg=[0.0], noise_std=[0.0, 4.0])


def run_case(beta, R0_px, tilt_deg, noise_std, seed=0):
    img, truth = render_drop(beta, R0_px, tilt_deg=tilt_deg,
                             noise_std=noise_std, seed=seed)
    row = dict(beta=beta, R0_px=R0_px, tilt_deg=tilt_deg, noise=noise_std)
    edges = extract_pendant_edges(img)
    base = attempt_auto_baseline(edges, truth['needle_width_px'])
    if base is None:
        row.update(status='no baseline')
        return row
    t0 = time.perf_counter()
    r = calculate_physics(edges, base, PIXEL_TO_M,
                          density_inner=RHO_IN, density_outer=RHO_OUT, g=G)
    dt = time.perf_counter() - t0
    if not r.get('fit_success'):
        row.update(status='fail: ' + str(r.get('error')), time_s=dt)
        return row
    sigma_true = G * (RHO_IN - RHO_OUT) * (R0_px * PIXEL_TO_M) ** 2 / beta * 1e3
    row.update(status='ok', time_s=dt,
               nfev=r.get('fit_nfev', np.nan),
               beta_err_pct=100 * (r['beta'] / beta - 1),
               R0_err_pct=100 * (r['R0_px'] / R0_px - 1),
               sigma_err_pct=100 * (r['ift_mNm'] / sigma_true - 1),
               rmse_px=r['rmse_px'], qc_pass=r['qc_pass'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--csv')
    args = ap.parse_args()
    grid = QUICK if args.quick else FULL

    rows = []
    hdr = ("  beta   R0px tilt noise | status      time_s  nfev | "
           "beta_err%  R0_err%  sigma_err% | rmse_px  qc")
    print(hdr)
    print("-" * len(hdr))
    for combo in itertools.product(*grid.values()):
        row = run_case(*combo)
        rows.append(row)
        if row['status'] == 'ok':
            print(f"  {row['beta']:.2f} {row['R0_px']:6.0f} {row['tilt_deg']:4.1f} "
                  f"{row['noise']:5.1f} | ok        {row['time_s']:8.2f} "
                  f"{row['nfev']:5.0f} | {row['beta_err_pct']:+9.3f} "
                  f"{row['R0_err_pct']:+8.3f} {row['sigma_err_pct']:+10.3f} | "
                  f"{row['rmse_px']:7.3f}  {'pass' if row['qc_pass'] else 'FAIL'}",
                  flush=True)
        else:
            print(f"  {row['beta']:.2f} {row['R0_px']:6.0f} {row['tilt_deg']:4.1f} "
                  f"{row['noise']:5.1f} | {row['status']}", flush=True)

    ok = [r for r in rows if r['status'] == 'ok']
    if ok:
        for label, sel in (("untilted", [r for r in ok if r['tilt_deg'] == 0]),
                           ("tilted", [r for r in ok if r['tilt_deg'] != 0])):
            if not sel:
                continue
            se = np.array([abs(r['sigma_err_pct']) for r in sel])
            tt = np.array([r['time_s'] for r in sel])
            print(f"{label:9s}: |sigma err| mean {se.mean():.3f} %  max {se.max():.3f} %  "
                  f"| time mean {tt.mean():.2f} s  max {tt.max():.2f} s  (n={len(sel)})")
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
