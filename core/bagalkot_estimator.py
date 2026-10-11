"""
core.bagalkot_estimator
=======================

Bagalkot's (2019, thesis §4.3) procedure for the effective CO₂ diffusion
coefficient D from a measured drop-volume curve V_exp(t):

  i–ii.  Discretise eq 4.1; input V0 and V_exp(t).
  iii.   Set a range of D and divide it into **equally spaced** values.
  iv.    For every D: solve, get C_avg(t) (eq 4.5), SF (eq 4.6, closed
         form) and V(t) (eq 4.7).
  v.     Compare V(t) with V_exp(t) through F (eq 4.8).
  vi.    Update the interface position from V(t) (adaptable interface,
         §4.2.4) — only when the growing-sphere option is on.
  vii–viii. Plot F against D; D at F_min is the result (with its SF).
  ix.    "The smaller the subintervals of D the higher the accuracy":
         the grid is refined around the minimum ``n_refine`` times.

D only: his interface condition C = C0 means no interfacial resistance,
so this D is an *effective* D (his word).  Nothing beyond his method is
added here.

The equivalent sphere has the starting volume V0 (radius from V0).  The
model and the choices the thesis leaves open are documented in
:mod:`core.sphere_model`.
"""

import numpy as np

from core.sphere_model import (N_CELLS, growth_q_closed_form, objective_F,
                               simulate_sphere, sphere_radius_m,
                               swelling_factor)

DEFAULT_D_RANGE = (1e-11, 1e-8)       # m²/s
DEFAULT_N_GRID = 31                   # equally spaced values per pass
DEFAULT_N_REFINE = 3                  # refinement passes around the minimum
NEAR_FIT_TOL = 0.10                   # "equally good": F within 10 % of F_min
SF_TOL = 1e-6                         # growing sphere: SF iteration tolerance
SF_MAX_ITER = 30


class BagalkotCancelled(Exception):
    """Raised when the cancel check fires during the D sweep."""


def _static_curve_factory(n_cells):
    """C_avg/V0 vs τ for the static sphere, computed once and extended on
    demand (the curve does not depend on D)."""
    cache = {}

    def m_of_tau(tau):
        tau = np.asarray(tau, float)
        need = float(tau.max()) if tau.size else 0.0
        if 'run' not in cache or cache['run']['tau'][-1] < need:
            cache['run'] = simulate_sphere(max(need, 1e-3) * 1.05,
                                           n_cells=n_cells)
        run = cache['run']
        return np.interp(tau, run['tau'], run['m'])
    return m_of_tau


def evaluate_D(D, t, V_exp, V0, a0, moving=False, n_cells=N_CELLS,
               new_cell='interface', static_curve=None, q_start=None):
    """F, SF and the model curve for one D (steps iv–vi).

    ``q_start`` (growing sphere): starting guess for q = (SF−1)/SF, e.g.
    from the neighbouring D; default the static-sphere SF.

    Returns dict(D, F, SF, V_model, C_avg, iterations, converged)."""
    tau = D * t / a0 ** 2
    if not moving:
        m = (static_curve or _static_curve_factory(n_cells))(tau)
        C_avg = V0 * m                                        # eq 4.5, µL
        SF = swelling_factor(t, V_exp, V0, C_avg)             # eq 4.6
        V_model = V0 + (SF - 1.0) * C_avg                     # eq 4.7
        return dict(D=D, F=objective_F(t, V_exp, V_model), SF=SF,
                    V_model=V_model, C_avg=C_avg, iterations=1,
                    converged=True)

    # Growing sphere: SF moves the boundary and the boundary changes
    # C_avg, so iterate (thesis §4.3 step vi).  Start from the static SF.
    m0 = _static_curve_factory(n_cells)(tau) if static_curve is None \
        else static_curve(tau)
    SF = swelling_factor(t, V_exp, V0, V0 * m0)
    q = max(0.0, (SF - 1.0) / SF) if SF > 0 else 0.0
    if q_start is not None:
        q = float(q_start)
    tau_end = float(tau.max()) * 1.0001

    def step(q_in):
        """One pass: grow the sphere with q_in, return (q_out, run, M)."""
        run_ = simulate_sphere(tau_end, n_cells=n_cells, growth_q=q_in,
                               new_cell=new_cell)
        M_ = V0 * np.interp(tau, run_['tau'], run_['m'])
        return max(0.0, growth_q_closed_form(t, V_exp, V0, M_)), run_, M_

    # Fixed point q = step(q), accelerated with the secant method on
    # g(q) = step(q) − q (plain repetition converges linearly).
    converged = False
    q_prev, g_prev = None, None
    for it in range(1, SF_MAX_ITER + 1):
        q_out, run, M = step(q)
        g = q_out - q
        if abs(g) < SF_TOL * (1.0 - min(q, 0.999)) ** 2:
            q = q_out
            converged = True
            break
        if g_prev is not None and g != g_prev:
            q_next = q - g * (q - q_prev) / (g - g_prev)
        else:
            q_next = q_out
        q_prev, g_prev = q, g
        q = float(np.clip(q_next, 0.0, 0.999))
    SF = 1.0 / (1.0 - q)
    V_model = V0 + q * M
    return dict(D=D, F=objective_F(t, V_exp, V_model), SF=float(SF), q=q,
                V_model=V_model, C_avg=M, iterations=it, converged=converged,
                radius_m=a0 * np.interp(tau, run['tau'], run['radius']))


def estimate_D_bagalkot(t, V_exp, V0=None, D_range=DEFAULT_D_RANGE,
                        n_grid=DEFAULT_N_GRID, n_refine=DEFAULT_N_REFINE,
                        moving=False, n_cells=N_CELLS, new_cell='interface',
                        progress_cb=None, is_cancelled=None):
    """Bagalkot's D sweep on a volume curve.

    t       : times (s) after drop formation, ascending
    V_exp   : measured volumes (µL); NaN entries are dropped
    V0      : starting volume (µL); default the first measured volume
    D_range : (D_min, D_max) in m²/s, swept on equal steps
    moving  : use the adaptable (growing) interface, §4.2.4

    Returns a dict with D (m²/s), D_per_min (m²/min, his tables' unit),
    SF, F_min, V0, radius_m, the fitted curve, the full sweep, the equally
    good D range and an at-edge flag.
    """
    t = np.asarray(t, float)
    V_exp = np.asarray(V_exp, float)
    ok = np.isfinite(t) & np.isfinite(V_exp) & (V_exp > 0)
    t, V_exp = t[ok], V_exp[ok]
    order = np.argsort(t, kind='stable')
    t, V_exp = t[order], V_exp[order]
    if len(t) < 5:
        raise ValueError(f"need at least 5 volume points, got {len(t)}")
    if t[-1] <= t[0]:
        raise ValueError("volume curve needs at least two distinct times")
    if V0 is None:
        V0 = float(V_exp[0])
    D_lo, D_hi = float(D_range[0]), float(D_range[1])
    if not 0 < D_lo < D_hi:
        raise ValueError("D range must satisfy 0 < D_min < D_max")
    if n_grid < 3:
        raise ValueError("need at least 3 D values per pass")
    a0 = sphere_radius_m(V0)
    static_curve = _static_curve_factory(n_cells)

    tried = {}
    total = n_grid * (1 + n_refine)
    done = [0]

    def run(D):
        if D in tried:
            return tried[D]
        if is_cancelled is not None and is_cancelled():
            raise BagalkotCancelled()
        q0 = None
        if moving and tried:
            # warm start from the nearest D already solved
            near = min(tried, key=lambda d: abs(d - D))
            if tried[near]['converged']:
                q0 = tried[near]['q']
        r = evaluate_D(D, t, V_exp, V0, a0, moving=moving, n_cells=n_cells,
                       new_cell=new_cell, static_curve=static_curve,
                       q_start=q0)
        tried[D] = r
        done[0] += 1
        if progress_cb is not None:
            progress_cb(min(done[0], total), total, D, r['F'])
        return r

    lo, hi = D_lo, D_hi
    passes = []
    for p in range(1 + n_refine):
        grid = np.linspace(lo, hi, n_grid)
        Fs = np.array([run(D)['F'] for D in grid])
        i = int(np.argmin(Fs))
        passes.append(dict(D=grid, F=Fs))
        if p == 0:
            at_edge = i in (0, n_grid - 1)
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
        if hi - lo <= 1e-6 * grid[i]:
            break

    Ds = np.array(sorted(tried))
    Fs = np.array([tried[D]['F'] for D in Ds])
    SFs = np.array([tried[D]['SF'] for D in Ds])
    best = tried[Ds[int(np.argmin(Fs))]]
    near = Fs <= best['F'] * (1.0 + NEAR_FIT_TOL) + 1e-15
    D_near = (float(Ds[near].min()), float(Ds[near].max()))
    near_open = bool(near[0] or near[-1])
    return dict(
        D=float(best['D']), D_per_min=float(best['D']) * 60.0,
        SF=float(best['SF']), F_min=float(best['F']), V0=float(V0),
        radius_m=float(a0), moving=bool(moving), new_cell=new_cell,
        t=t, V_exp=V_exp, V_model=best['V_model'], C_avg=best['C_avg'],
        sweep_D=Ds, sweep_F=Fs, sweep_SF=SFs, passes=passes,
        D_near=D_near, near_open=near_open, near_tol=NEAR_FIT_TOL,
        at_edge=bool(at_edge), D_range=(D_lo, D_hi),
        sf_converged=bool(best['converged']),
        n_evaluations=len(tried))
