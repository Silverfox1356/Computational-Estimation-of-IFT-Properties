"""
core.physics_calcs
==================

Pendant-drop physics:
    · Young–Laplace ODE integration (Bashforth–Adams form)
    · Simultaneous (x_axis, y_apex, R₀, β) least-squares fit (ADSA-P) with
      soft-L1 robust loss, signed point-to-curve normal distance and a
      mostly analytic Jacobian
    · Parameter-uncertainty propagation to σ including calibration and
      density tolerances
    · Axisymmetry diagnostic
    · DS/DE method kept as an independent cross-check
"""

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter


FIT_LOSS             = 'soft_l1'
FIT_F_SCALE_PX       = 0.5
FIT_N_THEO_POINTS    = 1200
FIT_S_MAX            = 6.0      # arc length integrated; cut at the neck
FIT_TOL              = 1e-8     # x/f/g tolerances — far below pixel noise
FIT_MAX_NFEV         = 300
FIT_TIME_LIMIT_S     = 60.0
FIT_BETA_FD_STEP     = 1e-5     # relative step for the ∂r/∂β column
FIT_TILT_MAX_DEG     = 10.0     # bound on the fitted camera/needle tilt
QC_TILT_MAX_DEG      = 3.0      # larger fitted tilt is flagged in QC
FIT_TRIM_TOP_PX      = 3
SAVGOL_WINDOW        = 15
SAVGOL_POLY          = 3
G_DEFAULT            = 9.80665


# ===========================================================================
# Young–Laplace profile
# ===========================================================================
def _rhs(s, y, beta):
    phi, x, z = y
    if x < 1e-10:
        return [2.0 - beta * z - 1.0, np.cos(phi), np.sin(phi)]
    return [2.0 - beta * z - np.sin(phi) / x,
            np.cos(phi), np.sin(phi)]


def yl_profile(beta, s_max=4.5, n=FIT_N_THEO_POINTS):
    """Non-dim YL integration from apex. L'Hôpital at s=0 → φ≈s, x≈s,
    z≈s²/2; we seed the solver at s=ε with exactly that."""
    eps = 1e-5
    y0 = [eps, eps, 0.5 * eps * eps]
    sol = solve_ivp(_rhs, [0.0, s_max], y0, args=(beta,),
                    dense_output=True,
                    rtol=1e-9, atol=1e-11, max_step=0.02)
    s = np.linspace(0, sol.t[-1], n)
    phi, x, z = sol.sol(s)
    return s, x, z, phi


# ===========================================================================
# Geometric helpers
# ===========================================================================
def fit_circle(x, y):
    def calc_R(cx, cy):
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    def resid(c):
        return calc_R(*c) - calc_R(*c).mean()
    res = least_squares(resid, [np.mean(x), np.mean(y)])
    return res.x, float(calc_R(*res.x).mean())


def fit_apex_circle(ys, lxs, rxs, apex_fraction=0.12):
    n = len(ys)
    start = int(n * (1 - apex_fraction))
    xf = np.concatenate([rxs[start:], lxs[start:]])
    yf = np.concatenate([ys[start:],  ys[start:]])
    (cx, cy), R = fit_circle(xf, yf)
    return (float(cx), float(cy)), float(R)


def dsde_beta(ys, lxs, rxs):
    widths = rxs - lxs
    DE = float(widths.max())
    apex_y = float(ys.max())
    idx = int(np.argmin(np.abs(ys - (apex_y - DE))))
    DS = float(widths[idx])
    r = DS / DE
    beta = 0.12836 - 0.7577 * r + 1.7713 * r ** 2 - 0.5426 * r ** 3
    return beta, DS, DE, r


def check_axisymmetry(ys, lxs, rxs):
    mids = 0.5 * (lxs + rxs)
    x_axis = float(np.median(mids))
    trans_rms = float(np.sqrt(np.mean((mids - x_axis) ** 2)))
    A = np.column_stack([ys, np.ones_like(ys)])
    coef, *_ = np.linalg.lstsq(A, mids, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = mids - (slope * ys + intercept)
    shape_rms = float(np.sqrt(np.mean(resid ** 2)))
    return dict(x_axis=x_axis,
                trans_rms_px=trans_rms,
                tilt_slope=slope,
                shape_rms_px=shape_rms)


# ===========================================================================
# Simultaneous robust fit
# ===========================================================================
class FitStopped(Exception):
    """Raised inside the fit when it is cancelled or exceeds its time limit."""


def _yl_drop_branch(beta, s_max=FIT_S_MAX, n=FIT_N_THEO_POINTS):
    """Physical part of the non-dim YL profile: apex → equator → neck.

    Past the neck (first minimum of x after the equator) the YL solution
    swells into a second, unphysical bulge — the real drop is attached to
    the needle before that, so the branch is cut there.  Returns
    (x, z, phi, cKDTree of (x, z)).
    """
    eps = 1e-5

    def hit_axis(s, y, b):                 # x → 0: stop before sin(φ)/x blows up
        return y[1] - 1e-3 if s > 0.5 else 1.0
    hit_axis.terminal = True

    sol = solve_ivp(_rhs, [0.0, s_max], [eps, eps, 0.5 * eps * eps],
                    args=(beta,), dense_output=True, events=hit_axis,
                    rtol=1e-9, atol=1e-11, max_step=0.02)
    s = np.linspace(0.0, sol.t[-1], n)
    phi, x, z = sol.sol(s)
    dx = np.diff(x)
    if np.any(dx < 0):
        i_eq = int(np.argmax(dx < 0))
        grow = np.where(dx[i_eq:] > 0)[0]
        if len(grow):
            i_neck = i_eq + int(grow[0])
            x, z, phi = x[:i_neck + 1], z[:i_neck + 1], phi[:i_neck + 1]
    return x, z, phi, cKDTree(np.column_stack([x, z]))


def fit_profile_simultaneous(x_exp, y_exp,
                             x_axis0, y_apex0, R0_px0, beta0=0.2,
                             bounds=None, should_cancel=None,
                             time_limit_s=FIT_TIME_LIMIT_S):
    """Fit (x_axis, y_apex, log R₀, β) of the Young–Laplace profile (ADSA-P).

    Residual = SIGNED normal distance (px) from each edge point to the
    theoretical profile (positive outside the drop), found by projecting
    onto the two curve segments next to the nearest vertex — exact for the
    polyline, so it does not depend on how densely the curve is sampled.
    Both halves fold onto one branch via |x − x_axis| (axisymmetry).

    Jacobian: the x_axis, y_apex and R₀ columns are analytic (the curve is
    only translated / scaled, so ∂r/∂p = −N·∂F/∂p at the foot point F with
    unit normal N); only the β column needs a finite difference, i.e. one
    extra profile integration per iteration.

    ``should_cancel()`` → True, or exceeding ``time_limit_s``, raises
    FitStopped.  Stopping criteria are set well below pixel noise but
    nowhere near machine precision, so a noisy contour cannot make the
    optimiser crawl for minutes.
    """
    x_exp = np.asarray(x_exp, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)
    t_start = time.perf_counter()

    branch_cache = {}

    def branch(beta):
        key = float(beta)
        if key not in branch_cache:
            if len(branch_cache) >= 32:
                branch_cache.clear()
            branch_cache[key] = _yl_drop_branch(key)
        return branch_cache[key]

    def check_stop():
        if should_cancel is not None and should_cancel():
            raise FitStopped("cancelled")
        if time_limit_s is not None and time.perf_counter() - t_start > time_limit_s:
            raise FitStopped(f"time limit ({time_limit_s:g} s) exceeded")

    def geometry(p):
        """Signed residuals plus what the analytic Jacobian needs."""
        x_axis, y_apex, logR, beta, theta = p
        R0 = np.exp(logR)
        xn, zn, phin, tree = branch(beta)
        # Rotate edge points into the drop frame (axis vertical) about the
        # apex: u across the axis, v along it (image-down positive).
        c, s = np.cos(theta), np.sin(theta)
        dx, dy = x_exp - x_axis, y_exp - y_apex
        u = c * dx + s * dy
        v = -s * dx + c * dy
        sgn = np.where(u >= 0, 1.0, -1.0)
        q = np.column_stack([sgn * u / R0, -v / R0])
        _, i = tree.query(q)
        best_d = np.full(len(q), np.inf)
        foot = np.zeros_like(q)
        phi_f = np.zeros(len(q))
        last = len(xn) - 1
        for j in (i - 1, i):                     # segments [j, j+1] around i
            a_i = np.clip(j, 0, last - 1)
            a = np.column_stack([xn[a_i], zn[a_i]])
            ab = np.column_stack([xn[a_i + 1], zn[a_i + 1]]) - a
            t = np.clip(np.einsum('ij,ij->i', q - a, ab) /
                        np.einsum('ij,ij->i', ab, ab), 0.0, 1.0)
            f = a + t[:, None] * ab
            d = np.hypot(*(q - f).T)
            better = d < best_d
            best_d[better] = d[better]
            foot[better] = f[better]
            phi_f[better] = (phin[a_i] + t * (phin[a_i + 1] - phin[a_i]))[better]
        N = np.column_stack([np.sin(phi_f), -np.cos(phi_f)])   # outward normal
        r = R0 * np.einsum('ij,ij->i', q - foot, N)
        return r, R0, sgn, foot, N, u, v, c, s

    def residuals(p):
        check_stop()
        try:
            return geometry(p)[0]
        except FitStopped:
            raise
        except Exception:
            return np.full(len(x_exp), 1e3)

    def jacobian(p):
        # r = sgn·u·Nx − v·Nz − R0·(F·N), with the foot F held fixed
        # (envelope theorem), u, v the rotated coordinates above.
        check_stop()
        r, R0, sgn, foot, N, u, v, c, s = geometry(p)
        Nx, Nz = N[:, 0], N[:, 1]
        J = np.empty((len(r), 5))
        J[:, 0] = -sgn * c * Nx - s * Nz                        # x_axis
        J[:, 1] = -sgn * s * Nx + c * Nz                        # y_apex
        J[:, 2] = -R0 * np.einsum('ij,ij->i', foot, N)          # log R0
        J[:, 4] = sgn * v * Nx + u * Nz                         # theta
        h = FIT_BETA_FD_STEP * max(1.0, abs(p[3]))
        hi = min(p[3] + h, bounds[1][3])
        lo = hi - h
        p_hi, p_lo = np.array(p, float), np.array(p, float)
        p_hi[3], p_lo[3] = hi, lo
        r_hi = geometry(p_hi)[0]
        r_lo = r if lo == p[3] else geometry(p_lo)[0]
        J[:, 3] = (r_hi - r_lo) / h                             # beta
        return J

    if bounds is None:
        x_lo, x_hi = float(np.min(x_exp)) - 200, float(np.max(x_exp)) + 200
        y_lo, y_hi = float(np.min(y_exp)) - 200, float(np.max(y_exp)) + 200
        t_max = np.deg2rad(FIT_TILT_MAX_DEG)
        bounds = ([x_lo, y_lo, 2.0, 0.03, -t_max],
                  [x_hi, y_hi, 10.0, 2.0, t_max])

    p0 = [np.clip(x_axis0, bounds[0][0] + 1, bounds[1][0] - 1),
          np.clip(y_apex0, bounds[0][1] + 1, bounds[1][1] - 1),
          float(np.log(R0_px0)),
          float(np.clip(beta0, bounds[0][3] + 1e-3, bounds[1][3] - 1e-3)),
          0.0]

    result = least_squares(
        residuals, p0, jac=jacobian,
        method='trf', loss=FIT_LOSS, f_scale=FIT_F_SCALE_PX,
        bounds=bounds, x_scale='jac',
        xtol=FIT_TOL, ftol=FIT_TOL, gtol=FIT_TOL, max_nfev=FIT_MAX_NFEV,
    )
    x_axis, y_apex, logR, beta, theta = result.x
    R0_px = float(np.exp(logR))
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return dict(x_axis=float(x_axis), y_apex=float(y_apex),
                R0_px=R0_px, beta=float(beta),
                tilt_rad=float(theta), tilt_deg=float(np.rad2deg(theta)),
                rmse_px=rmse, result=result, logR=float(logR))


def fitted_profile_px(results):
    """Fitted drop outline in image coordinates, as two polylines.

    Same geometry the fit used: the physical (neck-cut) YL branch, scaled
    by R₀, rotated by the fitted tilt about the fitted apex.  Returns
    ((x_right, y_right), (x_left, y_left)).
    """
    xn, zn, _, _ = _yl_drop_branch(results['beta'])
    R0 = results['R0_px']
    th = results.get('tilt_rad', 0.0)
    c, s = np.cos(th), np.sin(th)
    ax, ay = results['x_axis_fit'], results['y_apex_fit']
    out = []
    for side in (1.0, -1.0):
        u, v = side * xn * R0, -zn * R0          # drop frame (v image-down)
        out.append((ax + c * u - s * v, ay + s * u + c * v))
    return tuple(out)


# ===========================================================================
# Uncertainty propagation
# ===========================================================================
def compute_uncertainty(fit, sigma_Npm,
                        density_inner, density_outer, delta_rho,
                        needle_tol_rel=0.01,
                        density_inner_tol_rel=0.0005,
                        density_outer_tol_rel=0.02,
                        g_tol_rel=1e-5):
    res = fit['result']
    J   = res.jac
    dof = max(1, J.shape[0] - J.shape[1])
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        JTJ_inv = np.linalg.pinv(J.T @ J)
    s2 = float(np.sum(res.fun ** 2) / dof)
    cov = JTJ_inv * s2
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    d_logR, d_beta = float(perr[2]), float(perr[3])

    beta = fit['beta']
    rel_R0_fit   = d_logR
    rel_R0_calib = float(needle_tol_rel)
    rel_R0_total = float(np.hypot(rel_R0_fit, rel_R0_calib))
    rel_beta = d_beta / beta

    rho_var = ((density_inner * density_inner_tol_rel) ** 2 +
               (density_outer * density_outer_tol_rel) ** 2)
    rel_rho = float(np.sqrt(rho_var) / delta_rho)

    sigma_mNm = sigma_Npm * 1000.0
    c_R0_fit    = 2 * rel_R0_fit   * sigma_mNm
    c_R0_calib  = 2 * rel_R0_calib * sigma_mNm
    c_beta      = rel_beta         * sigma_mNm
    c_rho       = rel_rho          * sigma_mNm
    c_g         = g_tol_rel        * sigma_mNm

    total_rel = float(np.sqrt((2 * rel_R0_total) ** 2 +
                              rel_beta ** 2 +
                              rel_rho ** 2 +
                              g_tol_rel ** 2))
    total_abs = sigma_mNm * total_rel

    return dict(sigma_mNm=sigma_mNm,
                total_abs_mNm=total_abs, total_rel=total_rel,
                d_beta=d_beta, d_R0_rel=rel_R0_fit,
                c_R0_fit=c_R0_fit, c_R0_calib=c_R0_calib,
                c_beta=c_beta, c_rho=c_rho, c_g=c_g,
                rel_R0_fit=rel_R0_fit, rel_R0_calib=rel_R0_calib,
                rel_beta=rel_beta, rel_rho=rel_rho,
                cov=cov)


# ===========================================================================
# Main entry point
# ===========================================================================
def calculate_physics(edges_df, baseline_y, pixel_to_m,
                      density_inner=998.2, density_outer=1.204,
                      g=G_DEFAULT,
                      needle_tol_rel=0.01,
                      density_inner_tol_rel=0.0005,
                      density_outer_tol_rel=0.02,
                      qc_wo_min=0.30,
                      qc_rmse_max_px=1.5,
                      qc_beta_range=(0.05, 1.50),
                      qc_asym_trans_px=1.0,
                      qc_tilt_max_deg=QC_TILT_MAX_DEG,
                      should_cancel=None,
                      time_limit_s=FIT_TIME_LIMIT_S,
                      warm_start=None):
    """Fit one pendant-drop contour below ``baseline_y``.

    ``warm_start``: optional dict with the previous frame's fitted
    ``beta`` and ``R0_px`` (video analysis), used as the starting point of
    the shape fit instead of the DS/DE guess; the result is the same
    minimum, reached in fewer iterations.  ``should_cancel`` (callable)
    and ``time_limit_s`` stop a fit that runs too long.
    """
    sub = edges_df[edges_df['y'] >= baseline_y].reset_index(drop=True)
    if len(sub) < 30:
        return dict(fit_success=False,
                    error="Too few contour rows below baseline.")

    ys = sub['y'].to_numpy(dtype=float)
    lx = sub['left_x'].to_numpy(dtype=float)
    rx = sub['right_x'].to_numpy(dtype=float)

    if len(ys) > SAVGOL_WINDOW:
        lxs = savgol_filter(lx, SAVGOL_WINDOW, SAVGOL_POLY)
        rxs = savgol_filter(rx, SAVGOL_WINDOW, SAVGOL_POLY)
    else:
        lxs, rxs = lx, rx

    radii_px = 0.5 * (rxs - lxs)
    V_m3 = float(np.pi * np.sum(radii_px ** 2)) * pixel_to_m ** 3
    V_uL = V_m3 * 1e9

    pre_asym = check_axisymmetry(ys, lxs, rxs)
    try:
        (cx0, cy0), R0_px_init = fit_apex_circle(ys, lxs, rxs)
    except Exception as e:
        return dict(fit_success=False, error=f"Apex circle fit failed: {e}", volume_ul=V_uL)
    y_apex0 = float(ys.max())
    x_axis0 = pre_asym['x_axis']

    beta_dsde, DS, DE, ratio = dsde_beta(ys, lxs, rxs)

    fit_mask = ys >= (baseline_y + FIT_TRIM_TOP_PX)
    ys_f, lx_f, rx_f = ys[fit_mask], lx[fit_mask], rx[fit_mask]
    if len(ys_f) < 20:
        ys_f, lx_f, rx_f = ys, lx, rx

    x_exp = np.concatenate([rx_f, lx_f])
    y_exp = np.concatenate([ys_f, ys_f])

    # Start β from the DS/DE estimate the image already gives us; fall back
    # to a typical pendant-drop value if the DS/DE polynomial is off-range.
    beta0 = beta_dsde if np.isfinite(beta_dsde) and 0.05 <= beta_dsde <= 1.0 else 0.2

    try:
        if warm_start:
            R0_guess = float(warm_start.get('R0_px', R0_px_init))
            beta_guess = float(warm_start.get('beta', beta0))
        else:
            R0_guess, beta_guess = R0_px_init, beta0
        fit = fit_profile_simultaneous(x_exp, y_exp,
                                       x_axis0, y_apex0, R0_guess,
                                       beta0=beta_guess,
                                       should_cancel=should_cancel,
                                       time_limit_s=time_limit_s)
    except FitStopped as e:
        return dict(fit_success=False, error=f"Fit stopped: {e}", volume_ul=V_uL)
    except Exception as e:
        return dict(fit_success=False, error=f"Profile fit failed: {e}", volume_ul=V_uL)

    # status 0 = evaluation cap reached: keep the (usually fine) result and
    # let the "Fit converged" QC check flag it; status < 0 is a real failure.
    if fit['result'].status < 0:
        return dict(fit_success=False,
                    error=f"Fit did not converge (status={fit['result'].status})",
                    volume_ul=V_uL)

    R0_px_fit = fit['R0_px']
    R0_m_fit  = R0_px_fit * pixel_to_m
    beta_fit  = fit['beta']
    rmse_px   = fit['rmse_px']

    mids_f = 0.5 * (lx_f + rx_f)
    A = np.column_stack([ys_f, np.ones_like(ys_f)])
    coef_p, *_ = np.linalg.lstsq(A, mids_f, rcond=None)
    post_slope = float(coef_p[0])
    post_shape = float(np.sqrt(np.mean((mids_f - (coef_p[0] * ys_f + coef_p[1])) ** 2)))
    post_trans = float(np.sqrt(np.mean((mids_f - fit['x_axis']) ** 2)))
    # Left/right asymmetry in the fitted (tilt-corrected) frame: half the
    # per-row difference between right- and left-edge residuals.  Pure
    # noise gives ~noise/√2; a drop that bulges on one side, or an axis
    # that is off, gives a systematic difference.  Unlike the pre-fit
    # midpoint checks, a straight-but-tilted drop does not trip it.
    n_f = len(ys_f)
    lr_diff = 0.5 * (fit['result'].fun[:n_f] - fit['result'].fun[n_f:])
    lr_asym = float(np.sqrt(np.mean(lr_diff ** 2)))
    post_asym = dict(x_axis=fit['x_axis'], trans_rms_px=post_trans,
                     tilt_slope=post_slope, shape_rms_px=post_shape,
                     lr_asym_px=lr_asym)

    delta_rho  = abs(density_inner - density_outer)
    R0_m_init  = R0_px_init * pixel_to_m
    sigma_fit  = g * delta_rho * R0_m_fit  ** 2 / beta_fit
    sigma_dsde = g * delta_rho * R0_m_init ** 2 / beta_dsde

    unc = compute_uncertainty(fit, sigma_fit,
                              density_inner, density_outer, delta_rho,
                              needle_tol_rel=needle_tol_rel,
                              density_inner_tol_rel=density_inner_tol_rel,
                              density_outer_tol_rel=density_outer_tol_rel)

    qc_checks = [
        ("Fit RMSE",
         rmse_px <= qc_rmse_max_px,
         f"{rmse_px:.3f} px  (≤ {qc_rmse_max_px})"),
        ("β in plausible range",
         qc_beta_range[0] <= beta_fit <= qc_beta_range[1],
         f"{beta_fit:.4f}  ∈ {qc_beta_range}"),
        ("Fit converged",
         fit['result'].status > 0,
         f"status = {fit['result'].status}"),
        ("Tilt (fitted, corrected)",
         abs(fit['tilt_deg']) <= qc_tilt_max_deg,
         f"{fit['tilt_deg']:+.2f}°  (|·| ≤ {qc_tilt_max_deg}°)"),
        ("Left/right asymmetry",
         lr_asym <= qc_asym_trans_px,
         f"{lr_asym:.3f} px  (≤ {qc_asym_trans_px})"),
    ]
    qc_pass_all = all(ok for _, ok, _ in qc_checks)

    return dict(
        fit_success=True,
        ift_mNm=unc['sigma_mNm'],
        beta=beta_fit,
        R0_mm=R0_m_fit * 1000.0,
        volume_ul=V_uL,
        center=(fit['x_axis'], fit['y_apex']),
        R0_px=R0_px_fit,
        ift_uncertainty_mNm=unc['total_abs_mNm'],
        ift_rel_uncertainty=unc['total_rel'],
        beta_uncertainty=unc['d_beta'],
        R0_px_init=R0_px_init,
        x_axis_fit=fit['x_axis'], y_apex_fit=fit['y_apex'],
        tilt_deg=fit['tilt_deg'], tilt_rad=fit['tilt_rad'],
        rmse_px=rmse_px, rmse_um=rmse_px * pixel_to_m * 1e6,
        fit_nfev=int(fit['result'].nfev),
        ift_dsde_mNm=sigma_dsde * 1000.0,
        beta_dsde=beta_dsde, ds_de_ratio=ratio,
        ys_use=ys, lxs=lxs, rxs=rxs,
        ys_fit=ys_f, lxs_fit=lx_f, rxs_fit=rx_f,
        uncertainty=unc,
        pre_asym=pre_asym,
        post_asym=post_asym,
        qc_checks=qc_checks,
        qc_pass=qc_pass_all,
        pixel_to_m=pixel_to_m,
        delta_rho=delta_rho,
    )


def compute_worthington(results, needle_diameter_m, g=G_DEFAULT):
    if not results.get('fit_success'):
        return results
    V_m3 = results['volume_ul'] * 1e-9
    sigma_Npm = results['ift_mNm'] / 1000.0
    Wo = V_m3 * results['delta_rho'] * g / (np.pi * needle_diameter_m * sigma_Npm)
    results['Wo'] = float(Wo)
    results['qc_checks'].append(
        ("Worthington number", Wo >= 0.30, f"{Wo:.3f}  (≥ 0.30)"))
    results['qc_pass'] = all(ok for _, ok, _ in results['qc_checks'])
    return results
