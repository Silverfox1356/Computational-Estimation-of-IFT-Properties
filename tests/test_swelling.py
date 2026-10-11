"""
Swelling check (core.swelling): verdicts on synthetic volume curves.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_swelling.py      (< 1 s)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.swelling import static_error_indication, swelling_check, volume_noise

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


def curve(S, noise_rel, n=120, seed=0, tau_frac=0.25):
    """V0 = 10, swelling S approached exponentially, white noise."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([[0.0], np.geomspace(0.6, 600, n - 1)])
    V = 10.0 * (1 + S * (1 - np.exp(-t / (tau_frac * 600))))
    return t, V + rng.normal(0, noise_rel * 10.0, n)


def main():
    print("── noise estimate ──")
    t, V = curve(0.05, 0.002)
    sig = volume_noise(V)
    check("noise estimate within 30 % of the true 0.02", 0.014 < sig < 0.026,
          f"{sig:.4f}")

    print("── verdicts (threshold 3 %) ──")
    for S, noise, want in ((0.0, 0.002, 'not_measurable'),
                           (0.002, 0.003, 'not_measurable'),
                           (0.02, 0.001, 'negligible'),
                           (0.05, 0.002, 'significant'),
                           (0.20, 0.002, 'significant'),
                           (1.1, 0.002, 'significant'),
                           (-0.06, 0.002, 'shrinking')):
        r = swelling_check(*curve(S, noise))
        check(f"S = {S:+.3f}, noise {noise:.3f} -> {want}", r['verdict'] == want,
              f"got {r['verdict']}, S = {r['S']:+.4f} ± {r['S_err']:.4f}")
    r = swelling_check(*curve(0.05, 0.002))
    check("significant recommends Bagalkot", r['recommend'] == 'bagalkot')
    r = swelling_check(*curve(0.02, 0.001))
    check("negligible recommends Yang", r['recommend'] == 'yang')
    r = swelling_check(*curve(0.05, 0.002), threshold=0.08)
    check("threshold is honoured (5 % < 8 % -> negligible)", r['verdict'] == 'negligible')

    print("── details ──")
    t, V = curve(0.05, 0.001, tau_frac=4.0)   # still rising at the end
    r = swelling_check(t, V)
    check("still-rising flag", r['still_rising'] and 'lower bound' in r['message'])
    t, V = curve(0.05, 0.001)
    V[5] = np.nan
    r = swelling_check(t, V)
    check("NaN frames are skipped", r['n_valid'] == len(V) - 1)
    r = swelling_check(t[:3], V[:3])
    check("too few frames -> insufficient", r['verdict'] == 'insufficient')
    check("Table 4.3 indication at 3.1 % is 2.0 %",
          abs(static_error_indication(0.031) - 2.015) < 1e-9)
    check("Table 4.3 indication beyond his range is None",
          static_error_indication(1.5) is None)

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
