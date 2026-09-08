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
    θ-scheme (θ = 0.7 → Crank-Nicolson-like, unconditionally stable)

    At each step n:
        A · C^{n+1} = B · C^n + F
    where
        A = H/Δτ + θ · K_total
        B = H/Δτ − (1−θ) · K_total

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
def solve_diffusion(fem_results, domain_metadata,
                    D=1e-9, k=1e-5,
                    theta=0.7, dtau=0.01, n_steps=100,
                    total_time=None,
                    snapshot_steps=None,
                    progress_cb=None,
                    calibration=None,
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
        1 = fully implicit).
    dtau : float
        Time step size, in seconds.
    n_steps : int
        Number of time steps. Ignored if ``total_time`` is given.
    total_time : float or None
        Total simulated time in seconds. If given, overrides ``n_steps``
        as ``n_steps = round(total_time / dtau)`` (e.g. 600 for 10 minutes).
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
    """
    # ---- Unpack FEM data ----
    H  = fem_results['H']
    K  = fem_results['K']
    Kb = fem_results['Kb']
    F  = fem_results['F']
    points    = fem_results['points']
    triangles = fem_results['triangles']
    interface_edges = fem_results['interface_edges']

    N = H.shape[0]

    # ---- Convert requested wall-clock duration into a step count ----
    if total_time is not None:
        n_steps = max(1, round(total_time / dtau))

    # ---- Compute dimensionless Biot number ----
    r_n = domain_metadata.get('r_inner_m', 1e-3)
    kD = (k * r_n) / D
    if verbose:
        print("\n── Simulation parameters ──")
        print(f"  D  = {D:.2e} m²/s")
        print(f"  k  = {k:.2e} m/s")
        print(f"  r_n= {r_n:.4e} m")
        print(f"  kD = {kD:.4f}  (Biot number)")
        print(f"  θ  = {theta},  Δτ = {dtau},  steps = {n_steps}")
        print(f"  N  = {N} nodes\n")

    # ---- Total system matrices ----
    # K already includes D inside it from assembly
    K_total = K + Kb
    F_total = F.copy()

    # ---- Time discretisation ----
    H_dt = H / dtau
    A = H_dt + theta * K_total         # LHS (constant)
    # A never changes between steps, so factorise once instead of
    # re-solving from scratch every step.
    A_lu = splu(A.tocsc())

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

    for step in range(1, n_steps + 1):
        # RHS
        B = (H_dt - (1.0 - theta) * K_total) @ C + F_total

        # Solve
        C = A_lu.solve(B)

        # Stability clip (physically: concentration ∈ [0, 1])
        C = np.clip(C, 0.0, 1.0)

        # Interface concentration
        Cs = float(np.mean(C[intf_nodes]))

        Cs_history.append(Cs)
        time_history.append(step * dtau)

        # Diagnostics
        c_min, c_max = float(C.min()), float(C.max())
        has_nan = bool(np.any(np.isnan(C)))
        diag = (f"Step {step:>4d}/{n_steps} | "
                f"Cs={Cs:.6f} | "
                f"min={c_min:.6e} max={c_max:.6e}"
                + (" ⚠ NaN!" if has_nan else ""))
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
    )
