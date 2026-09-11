"""
core.estimator
==============

Inverse problem: recover the diffusion coefficient **D** and the
interfacial mass-transfer coefficient **k** from a measured dynamic
interfacial-tension curve γ_exp(t).

    find (D, k) that minimise  E(D, k) = Σ [ γ_sim(tᵢ; D, k) − γ_exp(tᵢ) ]²

Three layers, each a plain function:

    simulate_ift    – forward model: (D, k) → γ_sim on the experimental
                      time grid
    ift_residuals   – residual vector for scipy.optimize.least_squares
    estimate_D_k    – the fit itself; returns recovered D, k + fit info

Why the matrices are rescaled instead of reassembled
----------------------------------------------------
``assemble_fem_system`` bakes D into the stiffness matrix K and k into
the boundary terms Kb, F.  Both are *linear* in their coefficient, so a
run at a different (D, k) only needs a scalar multiply of the assembled
matrices — no reassembly.  The reference coefficients are stored in
``fem_results['assembly_D']`` / ``['assembly_k']``.

Why the fit is in log10 space
-----------------------------
D (~1e-9) and k (~1e-5) span orders of magnitude; fitting log10(D),
log10(k) gives the optimiser a well-scaled, dimension-comparable
parameter space.  Conversion back to physical units happens inside
``ift_residuals``.
"""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from core.time_solver import solve_diffusion

# Yang et al. (2006) search ranges for the reservoir brine–CO2 system
# (Section 2.3): D ∈ 0.01–10 ×10⁻⁹ m²/s, Biot number kD ∈ 0.1–15.
# These are SYSTEM-SPECIFIC — pass them explicitly to reproduce the paper.
YANG_KD_RANGE = (0.1, 15.0)
YANG_D_BOUNDS = (0.01e-9, 10.0e-9)

# General, system-agnostic defaults: wide enough to bracket the true values
# for most gas–liquid systems (gas diffusion in liquids is ~1e-11–1e-8 m²/s).
# The optimiser auto-widens these if the minimum lands on an edge, so the
# user need not know their system's magnitudes in advance.
GENERAL_KD_RANGE = (0.01, 100.0)
GENERAL_D_BOUNDS = (1e-11, 1e-8)

# Default search box for the fit (physical units).
DEFAULT_BOUNDS = ((1e-11, 1e-8),   # D  (m²/s)
                  (1e-7, 1e-3))    # k  (m/s)

# Forward runs inside the fit use a coarse time grid: the θ-scheme is
# unconditionally stable, so ~N_FIT_STEPS steps over the experiment
# duration trade resolution for speed (one optimiser run = dozens to
# hundreds of forward simulations).
N_FIT_STEPS = 200

# Convergence tolerance of the inner 1-D search over log10(D) (scipy's
# bounded Brent).  Brent keeps its iterate strictly INSIDE the bracket, so
# a D that has genuinely pinned to a bound still stops roughly half this
# distance away from it.  The edge test must therefore be LOOSER than the
# search's own resolution — a tighter one can never fire, silently turning
# "D pinned to the search box" into a reported interior minimum.
D_SEARCH_XATOL = 1e-2
EDGE_TOL_LOG10 = 2.0 * D_SEARCH_XATOL


class EstimationCancelled(Exception):
    """Raised out of the objective to unwind a cancelled fit.

    ``estimate_D_k_yang`` runs hundreds of forward simulations, so a caller
    (the GUI) must be able to stop one; it passes ``should_cancel``.
    """


def _validate_gamma_exp(gamma_exp):
    """Return ``gamma_exp`` as a float array, or explain why it can't be fit.

    The default (Yang eq 14) objective divides by γ_exp, so a non-positive
    or non-finite measurement poisons every residual and the optimiser
    silently returns nonsense.  In practice such points are failed
    drop-shape fits and should be removed by the caller (QC flags) — this
    is the last line of defence, not the filter.
    """
    gamma_exp = np.asarray(gamma_exp, dtype=float)
    if not np.all(np.isfinite(gamma_exp)):
        n = int(np.sum(~np.isfinite(gamma_exp)))
        raise ValueError(
            f"measured IFT contains {n} NaN/Inf value(s) — drop the failed "
            f"(QC-flagged) measurements before fitting")
    if np.any(gamma_exp <= 0.0):
        n = int(np.sum(gamma_exp <= 0.0))
        raise ValueError(
            f"measured IFT contains {n} non-positive value(s); interfacial "
            f"tension must be > 0 (the relative objective divides by it). "
            f"These are almost always failed drop-shape fits — drop them "
            f"before fitting")
    return gamma_exp


def _rescale_fem_system(fem_results, D, k):
    """Shallow-copied ``fem_results`` with K, Kb, F rescaled to (D, k).

    Falls back to the ``assemble_fem_system`` defaults for systems
    assembled before the reference coefficients were stored.
    """
    D_ref = fem_results.get('assembly_D', 1e-9)
    k_ref = fem_results.get('assembly_k', 1e-5)

    scaled = dict(fem_results)
    scaled['K'] = fem_results['K'] * (D / D_ref)
    scaled['Kb'] = fem_results['Kb'] * (k / k_ref)
    scaled['F'] = fem_results['F'] * (k / k_ref)
    scaled['assembly_D'] = D
    scaled['assembly_k'] = k
    return scaled


def simulate_ift(fem_results, domain_metadata, D, k, t_eval,
                 dtau, total_time=None, calibration=None):
    """
    Forward model as a clean callable.

    Runs ``solve_diffusion`` for the given (D, k) and interpolates the
    resulting γ(t) onto the requested time points.

    Parameters
    ----------
    fem_results : dict         – from ``assemble_fem_system``
    domain_metadata : dict     – from ``build_domain_polygon``
    D : float                  – diffusion coefficient (m²/s)
    k : float                  – mass-transfer coefficient (m/s)
    t_eval : array-like        – times (s) to evaluate γ_sim at
    dtau : float               – forward-run time step (s)
    total_time : float or None – simulated duration; default t_eval[-1]
    calibration : object or None
        Concentration→IFT calibration passed through to
        ``solve_diffusion`` (None = placeholder linear model).

    Returns
    -------
    gamma_sim : ndarray, same length as t_eval  (mN/m)
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if total_time is None:
        total_time = float(t_eval.max())

    fem_scaled = _rescale_fem_system(fem_results, D, k)
    sim = solve_diffusion(
        fem_scaled, domain_metadata,
        D=D, k=k,
        dtau=dtau, total_time=total_time,
        snapshot_steps=[],          # no snapshots inside the fit loop
        progress_cb=None,
        calibration=calibration,
        verbose=False,
    )
    return np.interp(t_eval, sim['time_history'], sim['gamma_history'])


def ift_residuals(params, fem_results, domain_metadata, t_exp, gamma_exp,
                  dtau, total_time, calibration=None, relative=True):
    """
    Residual vector for ``scipy.optimize.least_squares``.

    ``params = (log10 D, log10 k)``.  One forward simulation per call.

    Two objective flavours (``relative`` flag):

    * ``relative=True`` (Yang et al. 2006, eq 14) — the residual is the
      *relative* error ``(γ_sim − γ_exp) / γ_exp``.  Minimising Σ of its
      squares is equivalent to minimising Yang's E, the root-mean-squared
      relative error (the √, 1/N and ×100 % are monotinic and don't move
      the optimum).  This weights every point ∝ 1/γ_exp, so the low-IFT
      near-saturation tail is not swamped by the high-IFT early points.
    * ``relative=False`` — the original absolute residual γ_sim − γ_exp
      (mN/m), kept for comparison.
    """
    D = 10.0 ** params[0]
    k = 10.0 ** params[1]
    gamma_sim = simulate_ift(fem_results, domain_metadata, D, k, t_exp,
                             dtau=dtau, total_time=total_time,
                             calibration=calibration)
    gamma_exp = np.asarray(gamma_exp, dtype=float)
    resid = gamma_sim - gamma_exp
    if relative:
        # The 1/γ_m weighting of Yang's eq 14.  γ_exp > 0 is guaranteed by
        # _validate_gamma_exp in the callers — a failed drop-shape fit can
        # report γ ≤ 0, which would silently poison every residual.
        return resid / gamma_exp
    return resid


def estimate_D_k(fem_results, domain_metadata, t_exp, gamma_exp,
                 initial_guess=(1e-9, 1e-5),
                 bounds=DEFAULT_BOUNDS,
                 dtau=None, total_time=None,
                 calibration=None,
                 objective='relative',
                 verbose=False):
    """
    Least-squares fit for (D, k) against a measured γ_exp(t).

    Parameters
    ----------
    t_exp, gamma_exp : array-like
        Experimental dynamic-IFT curve (s, mN/m), same length.
    initial_guess : (D0, k0)
        Physical-unit starting point.
    bounds : ((D_lo, D_hi), (k_lo, k_hi))
        Physical-unit search box.
    dtau : float or None
        Forward-run step for fit iterations.  Default: total_time /
        N_FIT_STEPS — coarse on purpose, see module docstring.
    total_time : float or None
        Default: t_exp[-1].
    calibration : object or None
        Passed through to the forward model.
    objective : {'relative', 'absolute'}
        'relative' (default) uses Yang et al. 2006 eq 14 — the RMS
        *relative* error (residual weighted by 1/γ_exp).  'absolute' uses
        the plain γ_sim − γ_exp misfit.  See ``ift_residuals``.
    verbose : bool
        Print optimiser progress.

    Returns
    -------
    dict with keys
        'D', 'k'          – recovered values (m²/s, m/s)
        'objective'       – which error function was minimised
        'success'         – optimiser convergence flag
        'cost'            – 0.5 · Σ residual²  (least_squares convention)
        'E_percent'       – Yang eq 14 value: RMS relative error × 100 %
        'residual_rms'    – RMS misfit in mN/m (absolute, easy to read)
        'n_evaluations'   – number of forward simulations used
        'message'         – optimiser status message
        'gamma_fit'       – γ_sim at the optimum, on t_exp (for plotting)
    """
    if objective not in ('relative', 'absolute'):
        raise ValueError("objective must be 'relative' or 'absolute'")
    relative = (objective == 'relative')

    t_exp = np.asarray(t_exp, dtype=float)
    gamma_exp = _validate_gamma_exp(gamma_exp)

    if total_time is None:
        total_time = float(t_exp.max())
    if dtau is None:
        dtau = total_time / N_FIT_STEPS

    lb = np.log10([bounds[0][0], bounds[1][0]])
    ub = np.log10([bounds[0][1], bounds[1][1]])
    # Clip the start into the box: least_squares rejects an infeasible x0
    # outright ("Initial guess is outside of provided bounds"), which the
    # default guess is whenever a caller narrows `bounds` around it.
    x0 = np.clip(np.log10(np.asarray(initial_guess, dtype=float)), lb, ub)

    result = least_squares(
        ift_residuals, x0,
        bounds=(lb, ub),
        args=(fem_results, domain_metadata, t_exp, gamma_exp,
              dtau, total_time),
        kwargs=dict(calibration=calibration, relative=relative),
        # Finite-difference step in log10 space.  The scipy default
        # (~1e-8) would change D by a factor of 1 + 2e-8 — below the
        # forward model's numerical resolution.  0.01 ≈ a 2.3 % change.
        diff_step=0.01,
        x_scale='jac',
        verbose=2 if verbose else 0,
    )

    D_fit = 10.0 ** result.x[0]
    k_fit = 10.0 ** result.x[1]

    # Recover γ_sim at the optimum from the residual (mode-dependent), then
    # derive both error measures from it so they're consistent regardless
    # of which objective was minimised.
    if relative:
        rel_resid = result.fun                       # (γ_sim − γ_exp)/γ_exp
        gamma_fit = gamma_exp * (1.0 + rel_resid)
    else:
        gamma_fit = gamma_exp + result.fun           # γ_sim − γ_exp
        rel_resid = (gamma_fit - gamma_exp) / gamma_exp
    abs_resid = gamma_fit - gamma_exp

    return dict(
        D=D_fit,
        k=k_fit,
        objective=objective,
        success=bool(result.success),
        cost=float(result.cost),
        # Yang et al. 2006, eq 14 — RMS relative error as a percentage.
        E_percent=float(np.sqrt(np.mean(rel_resid ** 2)) * 100.0),
        residual_rms=float(np.sqrt(np.mean(abs_resid ** 2))),
        n_evaluations=int(result.nfev),
        message=result.message,
        gamma_fit=gamma_fit,
    )


def _relative_E(gamma_sim, gamma_exp):
    """Yang eq 14 without the ×100 %: sqrt(mean(((sim−exp)/exp)²))."""
    rel = (gamma_sim - gamma_exp) / gamma_exp
    return float(np.sqrt(np.mean(rel ** 2)))


def estimate_D_k_yang(fem_results, domain_metadata, t_exp, gamma_exp,
                      calibration=None,
                      kD_range=None, D_bounds=None, n_kD=25,
                      dtau=None, total_time=None,
                      n_D_scan=25, refine=True, scan_D=True,
                      auto_widen=True, max_widen=4, widen_factor=10.0,
                      should_cancel=None,
                      verbose=False):
    """
    Yang et al. (2006) nested one-parameter search for (D, k).

    Faithful to Section 2.3 of the paper:

    * the parameters are **D** and the mass-transfer Biot number
      ``kD = k·rₙ/D`` (rₙ = needle inner radius — the model's
      characteristic length, ``domain_metadata['r_inner_m']``, the same
      rₙ ``time_solver`` uses);
    * ``kD`` is the outer *loading* parameter, swept across ``kD_range``;
    * at each fixed ``kD`` a 1-D search over ``D`` finds the local-minimum
      objective E(D; kD) (Yang eq 14, the RMS relative error);
    * the ``kD`` whose local minimum is smallest is the global minimum
      ``E_min``, and its (D, kD) pair is the answer.

    Because ``kD`` is the parameter the transient's *shape* actually
    constrains, the returned E-vs-kD sweep doubles as an identifiability
    check — a sharp minimum means kD is well-determined, a flat one means
    it is not.  The two sweeps reproduce the paper's Figure 7.

    Search range — general vs system-specific
    -----------------------------------------
    The paper's ranges are tuned to CO2–brine.  This routine is
    system-agnostic: it defaults to wide ``GENERAL_*`` ranges and, if
    ``auto_widen`` is on, expands whichever bound the minimum lands on and
    re-sweeps (up to ``max_widen`` times).  So the user need not know their
    system's magnitudes.  To reproduce the paper exactly, or to hand-pin a
    system you know, pass ``kD_range``/``D_bounds`` explicitly.

    Parameters
    ----------
    kD_range : (lo, hi) or None  – Biot-number sweep range (None → general).
    D_bounds : (lo, hi) or None  – physical D search box, m²/s (None → general).
    n_kD : int                   – coarse (log-spaced) grid points across kD.
    n_D_scan : int               – points in the E-vs-D scan at the best kD.
    refine : bool                – finer kD sweep around the coarse best.
    scan_D : bool                – run the Figure-7a E-vs-D scan.  It costs
                                   ``n_D_scan`` extra forward simulations and
                                   is only needed for the plot, so pass False
                                   when you just want the numbers.
    auto_widen : bool            – expand an edge-hit bound and re-sweep.
    max_widen : int              – max auto-widen attempts.
    widen_factor : float         – multiplicative bound expansion per attempt.
    should_cancel : callable() -> bool or None
                                 – polled before every forward simulation; if
                                   it returns True the fit unwinds with
                                   :class:`EstimationCancelled`.  Lets a GUI
                                   abort a sweep that would otherwise run for
                                   minutes.

    Returns
    -------
    dict with keys
        'D', 'k', 'kD'    – recovered values (m²/s, m/s, dimensionless)
        'objective'       – 'relative' (Yang eq 14)
        'method'          – 'yang-nested'
        'success'         – True iff the minimum is interior (not pinned
                            to a search-box edge — a real quality signal)
        'message'         – human-readable status / how to fix an edge hit
        'E_percent'       – global-minimum E (RMS relative error × 100 %)
        'residual_rms'    – absolute RMS misfit at the optimum (mN/m)
        'n_evaluations'   – forward simulations used by the search itself
        'n_scan_evaluations' – extra simulations spent on the Fig-7a D scan
        'kD_range','D_bounds' – the final (possibly widened) search box
        'widen_steps'     – how many times a bound was auto-widened
        'gamma_fit'       – γ_sim at the optimum, on t_exp (for plotting)
        'sweep_kD'        – dict(kD, E[%], D): objective vs kD, already
                            minimised over D at each point.  Not a figure in
                            the paper; it is the kD identifiability view.
        'sweep_D'         – dict(D, E[%]): objective vs D at kD_best — this
                            is the paper's Figure 7b (Fig 7a is E vs D at
                            several kD, for one pressure).  None when
                            ``scan_D`` is False

    Raises
    ------
    EstimationCancelled
        If ``should_cancel`` returns True while the sweep is running.
    """
    kD_range = list(GENERAL_KD_RANGE if kD_range is None else kD_range)
    D_bounds = list(GENERAL_D_BOUNDS if D_bounds is None else D_bounds)

    r_n = float(domain_metadata.get('r_inner_m', 1e-3))
    t_exp = np.asarray(t_exp, dtype=float)
    gamma_exp = _validate_gamma_exp(gamma_exp)
    if total_time is None:
        total_time = float(t_exp.max())
    if dtau is None:
        dtau = total_time / N_FIT_STEPS

    n_calls = [0]

    def E_at(D, kD):
        """Objective for a (D, kD) pair: convert to physical k = kD·D/rₙ,
        run the forward model, return Yang's relative-error E."""
        if should_cancel is not None and should_cancel():
            raise EstimationCancelled()
        k = kD * D / r_n
        n_calls[0] += 1
        gamma_sim = simulate_ift(fem_results, domain_metadata, D, k, t_exp,
                                 dtau=dtau, total_time=total_time,
                                 calibration=calibration)
        return _relative_E(gamma_sim, gamma_exp)

    def best_D_for(kD, logD_lo, logD_hi):
        """Inner 1-D search: minimise E over D (in log10 space) at fixed
        kD.  Golden-section to ~2 % in D (D_SEARCH_XATOL, in log10)."""
        res = minimize_scalar(
            lambda logD: E_at(10.0 ** logD, kD),
            bounds=(logD_lo, logD_hi), method='bounded',
            options={'xatol': D_SEARCH_XATOL})
        return 10.0 ** res.x, float(res.fun)

    def sweep(kD_range, D_bounds):
        """One full outer sweep + refinement for the given search box.
        Returns sorted (kD, E, D) arrays and the index of the minimum."""
        logD_lo, logD_hi = np.log10(D_bounds[0]), np.log10(D_bounds[1])
        # Log-spaced coarse grid: an order-of-magnitude search resolves kD
        # proportionally everywhere, unlike a linear grid over a wide range.
        kD_list = list(np.geomspace(kD_range[0], kD_range[1], n_kD))
        E_list, D_list = [], []
        for kD in kD_list:
            D_loc, E_loc = best_D_for(kD, logD_lo, logD_hi)
            D_list.append(D_loc)
            E_list.append(E_loc)
            if verbose:
                print(f"  kD={kD:9.4f}  D={D_loc:.4e}  E={E_loc*100:.3f}%")
        if refine and n_kD >= 3:
            m = int(np.argmin(E_list))
            lo = kD_list[max(0, m - 1)]
            hi = kD_list[min(len(kD_list) - 1, m + 1)]
            for kD in np.linspace(lo, hi, max(5, n_kD // 2)):
                if any(abs(kD - k0) < 1e-12 for k0 in kD_list):
                    continue
                D_loc, E_loc = best_D_for(kD, logD_lo, logD_hi)
                kD_list.append(float(kD))
                D_list.append(D_loc)
                E_list.append(E_loc)
        order = np.argsort(kD_list)
        return (np.asarray(kD_list)[order], np.asarray(E_list)[order],
                np.asarray(D_list)[order])

    # --- sweep, auto-widening any edge-hit bound and re-sweeping -----
    widen_steps = 0
    while True:
        kD_arr, E_arr, D_arr = sweep(kD_range, D_bounds)
        j = int(np.argmin(E_arr))
        kD_best, D_best, E_best = float(kD_arr[j]), float(D_arr[j]), float(E_arr[j])
        logD_lo, logD_hi = np.log10(D_bounds[0]), np.log10(D_bounds[1])

        at_kD_lo = (j == 0)
        at_kD_hi = (j == len(kD_arr) - 1)
        # EDGE_TOL_LOG10, not a tighter number: best_D_for's bounded Brent
        # search stops ~D_SEARCH_XATOL/2 short of a bound it is converging
        # onto, so a test tighter than that tolerance can never fire and a
        # D pinned to the search box would be reported as an interior
        # minimum (and never auto-widened).
        at_D_lo = abs(np.log10(D_best) - logD_lo) < EDGE_TOL_LOG10
        at_D_hi = abs(np.log10(D_best) - logD_hi) < EDGE_TOL_LOG10
        on_edge = at_kD_lo or at_kD_hi or at_D_lo or at_D_hi

        if not on_edge or not auto_widen or widen_steps >= max_widen:
            break
        # Expand only the bound(s) actually hit, then re-sweep.
        if at_kD_lo:
            kD_range[0] /= widen_factor
        if at_kD_hi:
            kD_range[1] *= widen_factor
        if at_D_lo:
            D_bounds[0] /= widen_factor
        if at_D_hi:
            D_bounds[1] *= widen_factor
        widen_steps += 1

    k_best = kD_best * D_best / r_n
    n_search_calls = n_calls[0]

    # --- E-vs-D scan at the global-min kD (paper Figure 7a) ----------
    # Purely for the plot, and it costs n_D_scan more forward simulations,
    # so it is skippable and counted separately from the search itself.
    if scan_D:
        D_scan = np.logspace(np.log10(D_bounds[0]), np.log10(D_bounds[1]),
                             n_D_scan)
        E_scan = np.array([E_at(D, kD_best) for D in D_scan])
        sweep_D = dict(D=D_scan, E=E_scan * 100.0)
    else:
        sweep_D = None

    # --- final curve + absolute misfit at the optimum ----------------
    gamma_fit = simulate_ift(fem_results, domain_metadata, D_best, k_best,
                             t_exp, dtau=dtau, total_time=total_time,
                             calibration=calibration)
    abs_resid = gamma_fit - gamma_exp

    success = not on_edge
    if success:
        message = "global minimum found in the interior of the search box"
        if widen_steps:
            message += f" (after auto-widening {widen_steps}×)"
    else:
        # Name EVERY bound that was hit, not just the first: when both kD
        # and D are pinned, reporting only one sends the user to widen the
        # wrong parameter.
        hits = []
        if at_kD_lo or at_kD_hi:
            hits.append("kD range ({} edge)".format(
                "lower" if at_kD_lo else "upper"))
        if at_D_lo or at_D_hi:
            hits.append("D bounds ({} edge)".format(
                "lower" if at_D_lo else "upper"))
        message = (f"minimum still on the {' and the '.join(hits)} after "
                   f"{widen_steps} auto-widen step(s) — result is unreliable; "
                   f"widen the search range manually")

    return dict(
        D=D_best, k=k_best, kD=kD_best,
        objective='relative', method='yang-nested',
        success=success, message=message,
        E_percent=E_best * 100.0,
        residual_rms=float(np.sqrt(np.mean(abs_resid ** 2))),
        n_evaluations=n_search_calls,
        n_scan_evaluations=n_calls[0] - n_search_calls,
        kD_range=tuple(kD_range), D_bounds=tuple(D_bounds),
        widen_steps=widen_steps,
        gamma_fit=gamma_fit,
        sweep_kD=dict(kD=kD_arr, E=E_arr * 100.0, D=D_arr),
        sweep_D=sweep_D,
    )
