"""
tests.test_fem_conservation
===========================

Regression tests for the physics of the FEM forward model.

``assemble_stiffness_matrix`` once added an "axisymmetric correction"
(1/r)∂c/∂r on top of the r-weighted weak form, which already contains
that term.  Counted twice, it made K non-symmetric with non-zero column
sums — the operator itself created CO2 (~25 % of the drop's content at
t = 180 s in the Yang-2006 case) — and the drop filled faster than an
exact analytical bound.  Mesh-convergence studies cannot catch this (it
is an operator error, not a discretisation error), so these tests pin
the physics down directly:

  1. K is symmetric, and every row AND column sums to zero.
  2. Mass balance: CO2 in the domain = CO2 that crossed the interface.
  3. The drop never fills faster than the exact solution for a sphere of
     the same radius (Crank, *The Mathematics of Diffusion*, eq 6.40).
  4. The built-in sanity check flags a stiffness matrix that creates mass.

Run:  python tests/test_fem_conservation.py
(Plain script on purpose — the ift env has no pytest.)
"""

from itertools import pairwise

import numpy as np
from _synthetic_domain import R_DROP, get_fem_system
from scipy.optimize import brentq

from core.fem_assembly import run_sanity_checks
from core.time_solver import solve_diffusion

# The fixture is assembled at assemble_fem_system's default coefficients,
# so forward runs use exactly those (no rescaling of K, Kb, F needed).
D = 1e-9          # m²/s
K_COEF = 1e-5     # m/s
THETA = 0.7       # solve_diffusion default
DT = 0.5          # s
T_END = 180.0     # s


def _run(fem, meta):
    """Forward run recording the (clipped) field after every step."""
    states = [np.zeros(fem['H'].shape[0])]
    solve_diffusion(fem, meta, D=D, k=K_COEF, theta=THETA, dtau=DT,
                    total_time=T_END, snapshot_steps=[],
                    progress_cb=lambda step, n, C, Cs: states.append(C.copy()),
                    verbose=False)
    return states


def _drop_average(fem, meta, C):
    """Volume average of C over the drop (below the needle tip)."""
    pts, tri = np.asarray(fem['points']), np.asarray(fem['triangles'])
    P = pts[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    w = area * np.abs(P[:, :, 0]).mean(axis=1)          # ∝ 2π|r| dA
    in_drop = P[:, :, 1].mean(axis=1) > meta['neck_height_m']
    c_tri = C[tri].mean(axis=1)
    return float(np.sum(w[in_drop] * c_tri[in_drop]) / np.sum(w[in_drop]))


def _robin_sphere_uptake(a, t, n_terms=300):
    """Crank eq 6.40: fractional uptake of a sphere of radius a with
    surface condition −D ∂c/∂r = k(c − 1).  L = a·k/D."""
    L = a * K_COEF / D
    def root_fn(b):                     # β cot β + L − 1 = 0, multiplied out
        return b * np.cos(b) + (L - 1.0) * np.sin(b)

    s = 0.0
    for n in range(1, n_terms + 1):
        b = brentq(root_fn, 1e-12 if n == 1 else (n - 1) * np.pi, n * np.pi)
        s += 6 * L ** 2 * np.exp(-b ** 2 * D * t / a ** 2) / (
            b ** 2 * (b ** 2 + L * (L - 1.0)))
    return 1.0 - s


def test_stiffness_symmetric_and_conservative():
    fem, _ = get_fem_system()
    K = fem['K']
    one = np.ones(K.shape[0])
    scale = np.abs(K.diagonal()).max()
    asym = float(np.sqrt((K - K.T).power(2).sum() / K.power(2).sum()))
    row = np.abs(K @ one).max() / scale
    col = np.abs(K.T @ one).max() / scale
    print(f"  K: rel. asymmetry {asym:.1e}, max|row sum| {row:.1e}, "
          f"max|column sum| {col:.1e}  (relative to max|diag|)")
    assert asym < 1e-12, f"K is not symmetric (rel. error {asym:.1e})"
    assert row < 1e-10, f"K rows do not sum to zero ({row:.1e})"
    assert col < 1e-10, (f"K columns do not sum to zero ({col:.1e}): the "
                         f"diffusion operator creates or destroys mass")


def test_mass_balance():
    fem, meta = get_fem_system()
    assert fem['assembly_D'] == D and fem['assembly_k'] == K_COEF
    states = _run(fem, meta)
    one = np.ones(fem['H'].shape[0])
    oH, oKb, oF = fem['H'].T @ one, fem['Kb'].T @ one, float(one @ fem['F'])

    def influx(C):                      # CO2 crossing the interface per s
        return oF - oKb @ C

    crossed = sum(DT * (THETA * influx(c1) + (1 - THETA) * influx(c0))
                  for c0, c1 in pairwise(states))
    content = float(oH @ states[-1])
    err = abs(content - crossed) / content
    print(f"  mass balance at t = {T_END:.0f} s: content {content:.4e}, "
          f"crossed the interface {crossed:.4e}  (mismatch {100 * err:.2f} %)")
    # The solver clips C to [0, 1] (a small, bounded mass change, ~1 %);
    # the double-counted term this guards against created ~25 %.
    assert err < 0.03, (f"domain content and interface influx disagree by "
                        f"{100 * err:.1f} % — the operator creates mass")


def test_sphere_bound():
    fem, meta = get_fem_system()
    states = _run(fem, meta)
    c_avg = _drop_average(fem, meta, states[-1])
    bound = _robin_sphere_uptake(R_DROP, T_END)
    print(f"  drop average at t = {T_END:.0f} s: {c_avg:.3f}   exact sphere of "
          f"the same radius (upper bound): {bound:.3f}")
    # Less interface (the top is the needle tip) and more volume (the
    # needle column) than the sphere: the drop must fill SLOWER.
    assert c_avg <= bound, (f"drop fills faster than the exact sphere bound "
                            f"({c_avg:.3f} > {bound:.3f})")
    # ...but not absurdly slower — a floor that catches a dead solver.
    assert c_avg >= 0.8 * bound, f"drop barely fills ({c_avg:.3f})"


def test_sanity_check_flags_mass_creation():
    fem, _ = get_fem_system()
    args = (fem['H'], fem['K'], fem['Kb'], fem['F'], fem['points'],
            fem['interface_edges'], fem['wall_edges'])
    good = run_sanity_checks(*args)
    assert good['K_symmetry_status'] == 'ok', good['K_symmetry_error']
    assert good['K_conservative'], good['K_colsum_rel']

    # Perturb one off-diagonal entry by 4 % of the diagonal scale: the
    # matrix is now asymmetric and one column no longer sums to zero.
    # In absolute terms the asymmetry is ~1e-14, which the old absolute
    # 1e-10 tolerance reported as 'ok'.
    K_bad = fem['K'].tolil()
    K_bad[0, 1] += 0.04 * np.abs(fem['K'].diagonal()).max()
    bad = run_sanity_checks(args[0], K_bad.tocsr(), *args[2:])
    print(f"  sanity check on a mass-creating K: symmetry "
          f"'{bad['K_symmetry_status']}', conservative {bad['K_conservative']}")
    assert bad['K_symmetry_status'] == 'fail'
    assert not bad['K_conservative']
    assert not bad['all_pass']


if __name__ == '__main__':
    print("── FEM conservation regression tests ──")
    for test in (test_stiffness_symmetric_and_conservative, test_mass_balance,
                 test_sphere_bound, test_sanity_check_flags_mass_creation):
        print(f"{test.__name__}")
        test()
    print("PASS")
