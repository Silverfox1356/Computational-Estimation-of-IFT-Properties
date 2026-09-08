"""
tests.test_calibration
======================

Phase-B acceptance: the plug-and-play calibration.

    1. Table lookups + units bridge behave (endpoints, clamping,
       pressure → c_sat).
    2. A forward run through the REAL Yang calibration reaches the
       digitized equilibrium IFT for its pressure, decreasing
       monotonically on the way.
    3. A dummy calibration CSV loads and runs with no code edit —
       plug-and-play proven.
    4. Synthetic (D, k) recovery still works with the real calibration
       in the loop (the whole inverse chain, end to end).

Run:  python tests/test_calibration.py
"""

import tempfile
from pathlib import Path

import numpy as np

from _synthetic_domain import get_fem_system
from core.calibration import Calibration, c_sat_for_pressure
from core.estimator import estimate_D_k, simulate_ift
from config import make_calibration

REPO = Path(__file__).resolve().parents[1]
YANG_CSV = REPO / 'data' / 'calibration_co2_brine.csv'

# Yang run used throughout: P = 3.67 MPa → c_sat = 776.1 mol/m³,
# equilibrium IFT 43.16 mN/m (row 5 of the CSV).
P_TEST = 3.67
C_SAT_TEST = 776.1
GAMMA_EQ_TEST = 43.16


def test_table_and_units_bridge():
    cal = make_calibration(pressure_MPa=P_TEST)
    assert abs(cal.c_sat - C_SAT_TEST) < 1e-9

    # Cs = 0 → clean surface;  Cs = 1 → the row's equilibrium point
    # (c_sat lands exactly on a table node, so no interpolation error).
    assert abs(cal.gamma(0.0) - 70.55) < 1e-9
    assert abs(cal.gamma(1.0) - GAMMA_EQ_TEST) < 1e-9

    # Clamping: beyond either table end, hold the endpoint.
    assert abs(cal.gamma(-0.1) - 70.55) < 1e-9
    assert abs(cal.gamma(1.5) - 36.64) < 1e-9   # 1.5·c_sat > max c_eq

    # Monotone decreasing over the physical range.
    g = cal.gamma(np.linspace(0, 1, 50))
    assert np.all(np.diff(g) <= 1e-12)

    # Pressure lookup errors out helpfully off-table.
    assert abs(c_sat_for_pressure(YANG_CSV, 0.49) - 155.1) < 1e-9
    try:
        c_sat_for_pressure(YANG_CSV, 9.0)
        raise AssertionError("expected ValueError for off-table pressure")
    except ValueError:
        pass
    print("  table + units bridge: OK")


def test_forward_reaches_equilibrium():
    fem, meta = get_fem_system()
    cal = make_calibration(pressure_MPa=P_TEST)

    t = np.linspace(5.0, 1500.0, 100)
    gamma = simulate_ift(fem, meta, 1e-9, 1e-5, t,
                         dtau=5.0, total_time=1500.0, calibration=cal)

    assert np.all(np.diff(gamma) <= 1e-9), "dynamic IFT must decrease"
    assert gamma[0] < 70.55, "must start below clean-surface value"
    err = gamma[-1] - GAMMA_EQ_TEST
    print(f"  gamma(t→∞) = {gamma[-1]:.2f} mN/m vs table {GAMMA_EQ_TEST} "
          f"(diff {err:+.2f})")
    assert abs(err) < 0.5, "equilibrium IFT not reproduced"


def test_dummy_csv_plug_and_play():
    # A made-up second system, dropped in as data only.
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / 'dummy_system.csv'
        csv.write_text("# c_eq, gamma_eq\n0,50\n100,40\n200,20\n")
        cal = Calibration(csv, gamma0=50.0, gamma_inf=20.0, c_sat=200.0)

        assert abs(cal.gamma(0.0) - 50.0) < 1e-9
        assert abs(cal.gamma(0.25) - 45.0) < 1e-9   # midpoint, 1st segment
        assert abs(cal.gamma(1.0) - 20.0) < 1e-9

        fem, meta = get_fem_system()
        t = np.linspace(2.0, 100.0, 20)
        gamma = simulate_ift(fem, meta, 1e-9, 1e-5, t,
                             dtau=1.0, calibration=cal)
        assert np.all(gamma <= 50.0) and np.all(gamma >= 20.0)
    print("  dummy CSV ran with zero code change: OK")


def test_recovery_with_real_calibration():
    fem, meta = get_fem_system()
    cal = make_calibration(pressure_MPa=P_TEST)

    D_true, k_true, T = 0.77e-9, 2.69e-5, 200.0   # Yang Table 2 @ 3.67 MPa
    t_exp = np.linspace(2.0, T, 60)
    gamma_true = simulate_ift(fem, meta, D_true, k_true, t_exp,
                              dtau=1.0, total_time=T, calibration=cal)
    rng = np.random.default_rng(7)
    gamma_exp = gamma_true + rng.normal(0.0, 0.05, t_exp.size)

    result = estimate_D_k(fem, meta, t_exp, gamma_exp,
                          initial_guess=(3e-9, 8e-6), calibration=cal)
    D_err = abs(result['D'] - D_true) / D_true
    k_err = abs(result['k'] - k_true) / k_true
    print(f"  true D={D_true:.3e} k={k_true:.3e} | "
          f"found D={result['D']:.3e} k={result['k']:.3e} | "
          f"err {100 * D_err:.2f} % / {100 * k_err:.2f} % "
          f"({result['n_evaluations']} sims)")
    assert result['success']
    assert D_err < 0.05 and k_err < 0.05


if __name__ == '__main__':
    print("── Phase B calibration tests ──")
    test_table_and_units_bridge()
    test_forward_reaches_equilibrium()
    test_dummy_csv_plug_and_play()
    test_recovery_with_real_calibration()
    print("PASS")
