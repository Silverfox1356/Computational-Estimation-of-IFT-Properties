"""
core.sphere_model
=================

Bagalkot's (2019, thesis §4.2) forward model: CO₂ diffusing into an
equivalent **sphere** with the drop's starting volume V0.

    ∂C/∂t = D ∇²C                       (eq 4.1, spherical symmetry)
    C(r, 0) = 0                         (eq 4.2)
    C(R, t) = C0                        (eq 4.3, interface at equilibrium)
    ∂C/∂r = 0 at r = 0                  (eq 4.4)
    C_avg(t) = ∫ C/C0 dV                (eq 4.5, a volume, here in µL)
    SF − 1 = ∫(Vexp−V0)C_avg/Vexp² dt / ∫C_avg²/Vexp² dt        (eq 4.6)
    V(t) = V0 + (SF − 1)·C_avg(t)       (eq 4.7)
    F = (1/T)∫ ((Vexp − V)/Vexp)² dt · 100 %                    (eq 4.8)

Numerics
--------
He solves eq 4.1 on an axisymmetric (r, z) finite-difference grid around
the sphere.  A sphere is spherically symmetric, so a 1-D radial
finite-volume grid solves the same problem exactly (to discretisation
error) at far lower cost; solute is conserved cell by cell.  Time
stepping is Crank–Nicolson (as his) after a few fully implicit start-up
steps that damp the oscillation CN gives with a step boundary condition;
steps are fine at first and grow geometrically.

Static sphere: in τ = D·t/a0² the solution does not depend on D, so one
run gives C_avg/V0 as a function of τ for every D (cached).

Growing sphere (§4.2.4, the "adaptable interface")
--------------------------------------------------
Steps i–vii of §4.2.4: start with N cells; at every time step get V(t)
from the model volume, turn it into a radius, and add cells of the same
spacing when the radius has grown.  The thesis leaves three things open;
our choices:

1. **Concentration of new cells:** C = C0 (the new liquid forms at the
   interface, which is at equilibrium).  ``new_cell='neighbour'`` copies
   the outermost cell instead, for a sensitivity check.
2. **Volume law on a growing grid.** Taken literally, eq 4.7 counts the
   new liquid twice: at saturation C_avg equals the *grown* volume, so
   V = V0 + (SF−1)·V gives V0/(2−SF) instead of SF·V0 (and diverges for
   SF ≥ 2, e.g. his n-decane at 60 bar).  We use the mass-consistent form
   V = V0 + (SF−1)/SF · C_avg (volume grows in proportion to the CO₂
   dissolved), which equals eq 4.7 when the sphere does not grow.
3. **SF is needed to move the boundary, but SF comes from the fit** (the
   circularity in §4.3 step vi).  :mod:`core.bagalkot_estimator` iterates:
   solve, get SF from the closed form, move the boundary with it, solve
   again, until SF stops changing.
Only cells are added, never removed (SF < 1 keeps the starting grid).
"""

import numpy as np
from scipy.linalg import solve_banded

N_CELLS = 400
DTAU0 = 1e-6              # first step in τ = D t / a0²
GROWTH = 1.06             # step growth factor per step
DTAU_MAX_FRAC = 1 / 400   # longest step as a fraction of the run
N_IMPLICIT_START = 4      # fully implicit steps before Crank–Nicolson


def sphere_radius_m(V0_uL):
    """Radius (m) of the sphere with volume V0 (µL = mm³)."""
    return (3.0 * V0_uL * 1e-9 / (4.0 * np.pi)) ** (1.0 / 3.0)


def tau_steps(tau_end, dtau0=DTAU0, growth=GROWTH, max_frac=DTAU_MAX_FRAC):
    """Geometrically growing steps in τ that end exactly at ``tau_end``."""
    if tau_end <= 0:
        raise ValueError("simulation time must be > 0")
    cap = max(tau_end * max_frac, dtau0)
    steps, tot, d = [], 0.0, min(dtau0, tau_end)
    while tot < tau_end * (1 - 1e-12):
        d = min(d, cap, tau_end - tot)
        steps.append(d)
        tot += d
        d *= growth
    return np.array(steps)


def analytic_uptake(tau, n_terms=200):
    """Crank (1975) eq 6.20: M_t/M∞ for a sphere with C = C0 at the
    surface, τ = D t / a²."""
    tau = np.atleast_1d(np.asarray(tau, float))
    n = np.arange(1, n_terms + 1)[:, None]
    s = np.sum(np.exp(-(n * np.pi) ** 2 * tau[None, :]) / n ** 2, axis=0)
    out = 1.0 - 6.0 / np.pi ** 2 * s
    out[tau <= 0] = 0.0
    # short-time series (better conditioned for tiny τ)
    small = (tau > 0) & (tau < 1e-3)
    ts = tau[small]
    out[small] = 6.0 * np.sqrt(ts / np.pi) - 3.0 * ts
    return out


def _geometry(n, dr):
    """Cell volumes and face areas (dimensionless units, 4π kept)."""
    rf = np.arange(n + 1) * dr                         # faces 0 .. n
    vol = 4.0 / 3.0 * np.pi * (rf[1:] ** 3 - rf[:-1] ** 3)
    area = 4.0 * np.pi * rf ** 2
    return vol, area


def _operator(n, dr):
    """Banded K (3 × n, for solve_banded) and boundary vector b for
    v dC/dt = −K C + b, with C = 1 on the outer face (half-cell distance)."""
    vol, area = _geometry(n, dr)
    g = area[1:n] / dr                                 # interior faces 1..n-1
    g_out = area[n] / (0.5 * dr)                       # outer face to C = 1
    diag = np.zeros(n)
    diag[:-1] += g
    diag[1:] += g
    diag[-1] += g_out
    K = np.zeros((3, n))
    K[0, 1:] = -g                                      # upper
    K[1, :] = diag
    K[2, :-1] = -g                                     # lower
    b = np.zeros(n)
    b[-1] = g_out
    return vol, K, b


def simulate_sphere(tau_end, n_cells=N_CELLS, growth_q=None,
                    new_cell='interface', record_tau=None):
    """Solve the dimensionless problem (a0 = 1, D = 1, C0 = 1).

    growth_q : None for a static sphere; otherwise q = (SF−1)/SF and the
               sphere grows with V/V0 = 1 + q·M/V0 (adaptable interface).
    Returns dict(tau, m, radius, n_cells): ``m`` = C_avg / V0 at each step
    end (plus τ = 0), ``radius`` = R/a0.
    """
    if new_cell not in ('interface', 'neighbour'):
        raise ValueError("new_cell must be 'interface' or 'neighbour'")
    dr = 1.0 / n_cells
    V0 = 4.0 / 3.0 * np.pi
    n = n_cells
    vol, K, b = _operator(n, dr)
    C = np.zeros(n)
    taus, ms, radii, ncs = [0.0], [0.0], [1.0], [n]
    t = 0.0
    for k, dtau in enumerate(tau_steps(tau_end)):
        th = 1.0 if k < N_IMPLICIT_START else 0.5
        A = th * K
        A[1] += vol / dtau
        rhs = vol / dtau * C - (1 - th) * _apply(K, C) + b
        C = solve_banded((1, 1), A, rhs)
        t += dtau
        M = float(vol @ C)
        if growth_q is not None and growth_q > 0:
            # grow until the grid holds the model volume (cells never removed)
            while True:
                V = V0 + growth_q * M
                R = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
                n_new = max(n, int(np.floor(R / dr + 0.5)))
                if n_new == n:
                    break
                fill = 1.0 if new_cell == 'interface' else C[-1]
                C = np.concatenate([C, np.full(n_new - n, fill)])
                n = n_new
                vol, K, b = _operator(n, dr)
                M = float(vol @ C)
        taus.append(t)
        ms.append(M / V0)
        radii.append(n * dr)
        ncs.append(n)
    return dict(tau=np.array(taus), m=np.array(ms), radius=np.array(radii),
                n_cells=np.array(ncs))


def _apply(K, C):
    """K @ C for the banded K."""
    out = K[1] * C
    out[:-1] += K[0, 1:] * C[1:]
    out[1:] += K[2, :-1] * C[:-1]
    return out


# ---------------------------------------------------------------------------
# Eq 4.6 – 4.8
# ---------------------------------------------------------------------------
def _trapz(y, t):
    return float(np.trapezoid(y, t))


def swelling_factor(t, V_exp, V0, C_avg):
    """Eq 4.6: closed-form SF for a given C_avg(t) (same volume units)."""
    w = 1.0 / V_exp ** 2
    num = _trapz((V_exp - V0) * C_avg * w, t)
    den = _trapz(C_avg ** 2 * w, t)
    if den <= 0:
        return 1.0
    return 1.0 + num / den


def growth_q_closed_form(t, V_exp, V0, M):
    """Mass-consistent analogue of eq 4.6 for the growing sphere:
    least-squares q in V = V0 + q·M, with q = (SF−1)/SF."""
    w = 1.0 / V_exp ** 2
    den = _trapz(M ** 2 * w, t)
    return 0.0 if den <= 0 else _trapz((V_exp - V0) * M * w, t) / den


def objective_F(t, V_exp, V_model):
    """Eq 4.8, in %: time-averaged squared relative volume error."""
    T = float(t[-1] - t[0])
    if T <= 0:
        raise ValueError("need at least two distinct times")
    return _trapz(((V_exp - V_model) / V_exp) ** 2, t) / T * 100.0
