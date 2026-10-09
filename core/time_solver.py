"""
core.time_solver
================

Forward diffusion simulation using θ-scheme time stepping on the
axisymmetric pendant-drop FEM domain.

Physical parameters
-------------------
    D  = 1e-9  m²/s   (diffusion coefficient, aqueous species)
    k  = 1e-5  m/s    (mass-transfer coefficient at interface)
    kD = k·r_n / D    (dimensionless Biot number)

Time stepping
-------------
    θ-scheme, unconditionally stable for θ ≥ 0.5.

    At each step n:
        A · C^{n+1} = B · C^n + F
    where
        A = H/Δτ + θ · K_total
        B = H/Δτ − (1−θ) · K_total

    Two schedules (``stepping``):

    * ``'uniform'`` — constant step ``dtau`` (seconds), constant θ.
    * ``'graded'``  — after D. Yang's FEM code (2005 PhD thesis,
      Appendix D.2): tiny steps at the start, where the interface
      concentration changes fastest, growing as the solution smooths, and
      θ rising from 2/3 to 1 over the first dimensionless time unit.  See
      ``graded_schedule``.

All coordinates and matrices must be in **metres** (SI units).
"""

import numpy as np
from scipy.sparse.linalg import splu


# ======================================================================
# Surface-tension model
# ======================================================================
def compute_gamma(Cs_history, gamma0=72.0, gamma_inf=35.0):
    """
    Linear equation of state:  γ = γ∞ + (γ₀ − γ∞)(1 − Cs)

    Parameters
    ----------
    Cs_history : array-like   – interface concentration at each step
    gamma0     : float        – clean surface tension (mN/m)
    gamma_inf  : float        – fully saturated surface tension (mN/m)

    Returns
    -------
    gamma : ndarray (mN/m)
    """
    Cs = np.asarray(Cs_history, dtype=float)
    return gamma_inf + (gamma0 - gamma_inf) * (1.0 - Cs)


# ======================================================================
# Main solver
# ======================================================================
# Graded schedule (dimensionless time τ = D·t/rₙ²).  The θ ramp follows
# D. Yang's code (2005 thesis, Appendix D.2: THETA = 0.6667 + 0.3333·τ,
# capped at 1).  Instead of his per-step growth (which would need a new LU
# factorisation every step) the step doubles every GRADED_STEPS_PER_BLOCK
# steps, so one run needs only ~20 factorisations.
# First step: Yang used Δτ = 0.0025.  The interface concentration rises like
# √t at the start, and with a larger needle radius (rₙ = 0.5 mm in the test
# domain → Δτ 0.0025 = 0.6 s) that step was measurably too coarse (Cs off by
# 0.005 at 2 s).  1e-4 makes the start converged (tests/test_fit_reliability)
# for ~50 extra steps.
GRADED_DTAU0 = 1e-4
GRADED_STEPS_PER_BLOCK = 10
# Late-time step cap: at least this many steps over the whole run.
GRADED_MIN_STEPS = 200


def graded_schedule(total_time, D, r_n,
                    dtau0=GRADED_DTAU0,
                    steps_per_block=GRADED_STEPS_PER_BLOCK,
                    min_steps=GRADED_MIN_STEPS):
    """Step sizes (s) and θ per step for a graded run of ``total_time`` s.

    Steps start at ``dtau0`` in dimensionless time (τ = D·t/rₙ²) and double
    every ``steps_per_block`` steps until they reach total/``min_steps``,
    then stay there.  θ is constant within a block: min(1, 2/3 + τ/3) at the
    block's start.  The last step is trimmed to land exactly on
    ``total_time``.

    Returns (dt, theta): two equal-length float arrays.
    """
    if total_time <= 0 or D <= 0 or r_n <= 0:
        raise ValueError("total_time, D and r_n must be positive")
    to_s = r_n ** 2 / D                       # seconds per unit of τ
    tau_end = total_time / to_s
    dtau_max = tau_end / float(min_steps)
    dts, thetas = [], []
    tau, dtau = 0.0, min(dtau0, dtau_max)
    while tau < tau_end * (1.0 - 1e-12):
        theta = min(1.0, 2.0 / 3.0 + tau / 3.0)
        for _ in range(steps_per_block):
            step = min(dtau, tau_end - tau)
            if step <= tau_end * 1e-12:
                break
            dts.append(step * to_s)
            thetas.append(theta)
            tau += step
        dtau = min(2.0 * dtau, dtau_max)
    return np.asarray(dts), np.asarray(thetas)


def solve_diffusion(fem_results, domain_metadata,
                    D=1e-9, k=1e-5,
                    theta=0.7, dtau=0.01, n_steps=100,
                    total_time=None,
                    snapshot_steps=None,
                    progress_cb=None,
                    calibration=None,
                    stepping='uniform',
                    schedule_kw=None,
                    verbose=True):
    """
    Run a forward diffusion simulation.

    Parameters
    ----------
    fem_results : dict
        Output of ``assemble_fem_system``  (keys: H, K, Kb, F,
        interface_edges, wall_edges, points, triangles).
    domain_metadata : dict
        From ``build_domain_polygon`` (needs 'r_inner_m').
    D : float
        Diffusion coefficient (m²/s).
    k : float
        Mass-transfer coefficient (m/s).
    theta : float
        Implicitness parameter (0 = explicit, 0.5 = Crank-Nicolson,
        1 = fully implicit).  Uniform stepping only.
    dtau : float
        Time step size, in seconds.  Uniform stepping only.
    n_steps : int
        Number of time steps. Ignored if ``total_time`` is given.
    total_time : float or None
        Total simulated time in seconds. If given, overrides ``n_steps``
        as ``n_steps = round(total_time / dtau)`` (e.g. 600 for 10 minutes).
        Required for graded stepping.
    stepping : {'uniform', 'graded'}
        'uniform' = constant ``dtau`` and ``theta``; 'graded' = the
        ``graded_schedule`` (fine early steps, θ from 2/3 to 1).
    schedule_kw : dict or None
        Extra keyword arguments for ``graded_schedule`` (e.g. a finer
        ``min_steps``).  Graded stepping only.
    snapshot_steps : list of int or None
        Steps at which to save full concentration fields for
        contour plotting.  Default: [1, 10, 50, n_steps].
    progress_cb : callable(step, n_steps, C, Cs) or None
        Optional callback for GUI progress updates.
    calibration : object or None
        Optional concentration→IFT calibration with a ``gamma(Cs)``
        method (see ``core.calibration.Calibration``).  If None, the
        placeholder linear ``compute_gamma`` is used.
    verbose : bool
        Print per-step diagnostics to the console.  Set False for
        batch runs (e.g. inside the parameter-estimation loop).

    Returns
    -------
    result : dict
        'C_final'       – (N,) final concentration
        'Cs_history'    – list of mean interface concentration per step
        'time_history'  – list of times
        'gamma_history' – surface tension vs time
        'snapshots'     – dict {step: (N,) concentration}
        'points'        – (M, 2) node coords (metres)
        'triangles'     – (T, 3) connectivity
        'kD'            – dimensionless Biot number
        'diagnostics'   – per-step diagnostics string
        'n_steps'       – actual number of steps run (post total_time override)
        'actual_time'   – simulated duration actually reached, in seconds
                          (n_steps * dtau; may differ slightly from a
                          requested total_time due to rounding)
        'C_min', 'C_max' – lowest / highest nodal concentration over the
                          run.  Not clipped: the FEM solution can dip
                          slightly below 0 next to the interface at early
                          times, which is a discretisation feature, not
                          physics; clipping it would add solute.
        'stepping'      – the schedule used
    """
    if stepping not in ('uniform', 'graded'):
        raise ValueError("stepping must be 'uniform' or 'graded'")
    # ---- Unpack FEM data ----
    H  = fem_results['H']
    K  = fem_results['K']
    Kb = fem_results['Kb']
    F  = fem_results['F']
    points    = fem_results['points']
    triangles = fem_results['triangles']
    interface_edges = fem_results['interface_edges']

    N = H.shape[0]

    r_n = domain_metadata.get('r_inner_m', 1e-3)

    # ---- Step sizes (s) and θ for every step ----
    if stepping == 'graded':
        if total_time is None:
            raise ValueError("graded stepping needs total_time")
        dts, thetas = graded_schedule(total_time, D, r_n, **(schedule_kw or {}))
    else:
        if total_time is not None:
            n_steps = max(1, round(total_time / dtau))
        dts = np.full(n_steps, float(dtau))
        thetas = np.full(n_steps, float(theta))
    n_steps = len(dts)

    # ---- Compute dimensionless Biot number ----
    kD = (k * r_n) / D
    if verbose:
        print("\n── Simulation parameters ──")
        print(f"  D  = {D:.2e} m²/s")
        print(f"  k  = {k:.2e} m/s")
        print(f"  r_n= {r_n:.4e} m")
        print(f"  kD = {kD:.4f}  (Biot number)")
        print(f"  stepping = {stepping},  θ = {thetas[0]:.3f}..{thetas[-1]:.3f},  "
              f"Δt = {dts.min():.3g}..{dts.max():.3g} s,  steps = {n_steps}")
        print(f"  N  = {N} nodes\n")

    # ---- Total system matrices ----
    # K already includes D inside it from assembly
    K_total = K + Kb
    F_total = F.copy()

    # ---- Time discretisation ----
    # A = H/Δt + θ·K depends only on (Δt, θ).  Uniform runs factorise once;
    # graded runs change (Δt, θ) once per block, so the factorisations are
    # cached per pair rather than redone every step.
    lu_cache = {}

    def factor(dt, th):
        key = (dt, th)
        if key not in lu_cache:
            lu_cache[key] = splu((H / dt + th * K_total).tocsc())
        return lu_cache[key]

    # ---- Initial condition ----
    C = np.zeros(N)

    # ---- Interface nodes (from classified edges) ----
    intf_nodes = set()
    for e in interface_edges:
        intf_nodes.update(e)
    intf_nodes = np.array(sorted(intf_nodes))

    # ---- Snapshot schedule ----
    if snapshot_steps is None:
        snapshot_steps = sorted(set([1, 10, 50, n_steps]))
    snapshots = {}

    # ---- Storage ----
    Cs_history = []
    time_history = []
    diagnostics = []

    if verbose:
        print("Starting simulation...\n")

    t_now = 0.0
    C_min_run, C_max_run = 0.0, 0.0
    for step in range(1, n_steps + 1):
        dt, th = float(dts[step - 1]), float(thetas[step - 1])
        # RHS, then solve.  No clipping: see 'C_min' in the docstring.
        B = (H / dt - (1.0 - th) * K_total) @ C + F_total
        C = factor(dt, th).solve(B)

        # Interface concentration
        Cs = float(np.mean(C[intf_nodes]))

        t_now += dt
        Cs_history.append(Cs)
        time_history.append(t_now)

        # Diagnostics
        c_min, c_max = float(C.min()), float(C.max())
        C_min_run, C_max_run = min(C_min_run, c_min), max(C_max_run, c_max)
        has_nan = bool(np.any(np.isnan(C)))
        diag = (f"Step {step:>4d}/{n_steps} | "
                f"Cs={Cs:.6f} | "
                f"min={c_min:.6e} max={c_max:.6e}"
                + (" NaN!" if has_nan else ""))
        diagnostics.append(diag)

        if verbose and (step <= 5 or step % 20 == 0 or step == n_steps):
            print(diag)

        # Snapshot
        if step in snapshot_steps:
            snapshots[step] = C.copy()

        # GUI callback
        if progress_cb is not None:
            progress_cb(step, n_steps, C, Cs)

    # ---- Surface tension ----
    if calibration is not None:
        gamma = calibration.gamma(np.asarray(Cs_history))
    else:
        gamma = compute_gamma(Cs_history)

    if verbose:
        print("\n── Simulation complete ──")
        print(f"  Final Cs  = {Cs_history[-1]:.6f}")
        print(f"  Final C range: [{C.min():.6e}, {C.max():.6e}]")
        print(f"  γ range: [{gamma.min():.2f}, {gamma.max():.2f}] mN/m")

    return dict(
        C_final=C,
        Cs_history=Cs_history,
        time_history=time_history,
        gamma_history=gamma.tolist(),
        snapshots=snapshots,
        points=points,
        triangles=triangles,
        kD=kD,
        diagnostics=diagnostics,
        n_steps=n_steps,
        actual_time=time_history[-1],
        C_min=C_min_run,
        C_max=C_max_run,
        stepping=stepping,
    )
