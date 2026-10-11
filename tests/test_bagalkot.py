"""
Bagalkot (2019) volume method: sphere model and D search.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_bagalkot.py      (~30 s)

Checks
  1. static sphere vs Crank eq 6.20 (exact) to 0.1 % after τ = 1e-3
  2. eq 4.6 recovers SF exactly for a given C_avg; F = 0 for a perfect model
  3. growing sphere with q → 0 equals the static sphere
  4. growing sphere ends at V/V0 = SF (mass-consistent volume law)
  5. static fit recovers D and SF from noise-free static data (< 0.1 %)
  6. growing fit recovers D and SF from growing-sphere data (< 0.5 %)
  7. static fit on growing data is biased low, by about Bagalkot's
     Table 4.3 amount at V_f/V0 = 1.155 (he reports 14.8 %)
  8. noisy data: D within a few %, and the equally-good range contains it
  9. cancel and bad input
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.bagalkot_estimator import (BagalkotCancelled, estimate_D_bagalkot)
from core.sphere_model import (analytic_uptake, objective_F, simulate_sphere,
                               sphere_radius_m, swelling_factor)

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def growing_data(D, SF, V0, t):
    a0 = sphere_radius_m(V0)
    q = (SF - 1) / SF
    run = simulate_sphere(D * t.max() / a0 ** 2 * 1.001, growth_q=q)
    return V0 + q * V0 * np.interp(D * t / a0 ** 2, run['tau'], run['m'])


def main():
    print("── sphere model ──")
    r = simulate_sphere(2.0)
    ex = analytic_uptake(r['tau'])
    sel = r['tau'] > 1e-3
    err = np.max(np.abs(r['m'][sel] - ex[sel]) / ex[sel])
    check("static sphere matches Crank eq 6.20 within 0.1 %", err < 1e-3,
          f"max rel err {err:.2e}")

    t = np.linspace(0, 600, 61)
    C = 3.0 * analytic_uptake(1e-3 * t)
    V = 3.0 + 0.2 * C
    check("eq 4.6 recovers SF exactly", abs(swelling_factor(t, V, 3.0, C) - 1.2) < 1e-12)
    check("eq 4.8 is zero for a perfect model", objective_F(t, V, V) == 0.0)

    rq = simulate_sphere(1.0, growth_q=1e-12)
    rs = simulate_sphere(1.0)
    check("growing sphere with q -> 0 equals the static sphere",
          np.allclose(rq['m'], rs['m'], atol=1e-12))
    rg = simulate_sphere(30.0, growth_q=(1.155 - 1) / 1.155)
    V_ratio = 1 + (0.155 / 1.155) * rg['m'][-1]
    check("growing sphere saturates at V/V0 = SF", abs(V_ratio - 1.155) < 2e-3,
          f"V/V0 = {V_ratio:.4f}, radius {rg['radius'][-1]:.4f}")

    print("── D search ──")
    V0, D, SF = 12.5, 1.5e-9, 1.08
    a0 = sphere_radius_m(V0)
    t = np.linspace(0, 600, 121)
    V = V0 + (SF - 1) * V0 * analytic_uptake(D * t / a0 ** 2)
    r = estimate_D_bagalkot(t, V)
    check("static: D recovered < 0.1 %", abs(r['D'] / D - 1) < 1e-3,
          f"D = {r['D']:.5e}")
    check("static: SF recovered", abs(r['SF'] - SF) < 1e-4, f"SF = {r['SF']:.5f}")
    check("static: D in m2/min", abs(r['D_per_min'] - 60 * r['D']) < 1e-20)

    Dm, SFm = 8e-10, 1.155
    tm = np.linspace(0, 1800, 61)
    Vm = growing_data(Dm, SFm, V0, tm)
    rm = estimate_D_bagalkot(tm, Vm, moving=True, n_grid=11, n_refine=3)
    check("growing: D recovered < 0.5 %", abs(rm['D'] / Dm - 1) < 5e-3,
          f"D = {rm['D']:.5e}")
    check("growing: SF recovered", abs(rm['SF'] - SFm) < 2e-3, f"SF = {rm['SF']:.4f}")
    rs = estimate_D_bagalkot(tm, Vm, moving=False)
    change = (Dm / rs['D'] - 1) * 100
    check("static fit on 15.5 % swelling: D low by about 15 % (his 14.8 %)",
          8 < change < 22, f"static -> growing change {change:+.1f} %")

    rng = np.random.default_rng(1)
    Vn = V + rng.normal(0, 0.002 * V0, size=V.shape)
    rn = estimate_D_bagalkot(t, Vn)
    lo, hi = rn['D_near']
    check("noisy static data: D within 5 %", abs(rn['D'] / D - 1) < 0.05,
          f"D = {rn['D']:.4e}")
    check("noisy static data: equally good range contains the true D",
          lo <= D <= hi, f"{lo:.3e} to {hi:.3e}")

    print("── cancel / bad input ──")
    try:
        estimate_D_bagalkot(t, V, is_cancelled=lambda: True)
        check("cancel raises BagalkotCancelled", False)
    except BagalkotCancelled:
        check("cancel raises BagalkotCancelled", True)
    for name, kw in (("too few points", dict(t=t[:3], V_exp=V[:3])),
                     ("bad D range", dict(t=t, V_exp=V, D_range=(1e-9, 1e-10)))):
        try:
            estimate_D_bagalkot(**kw)
            check(f"{name} rejected", False)
        except ValueError:
            check(f"{name} rejected", True)

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
