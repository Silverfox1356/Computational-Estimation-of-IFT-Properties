"""
core.physics_calcs
==================

Pendant-drop physics:
    · Young–Laplace ODE integration (Bashforth–Adams form)
    · Simultaneous (x_axis, y_apex, R₀, β) least-squares fit with
      soft-L1 robust loss and nearest-point (KDTree) distance metric
    · Parameter-uncertainty propagation to σ including calibration and
      density tolerances
    · Axisymmetry diagnostic
    · DS/DE method kept as an independent cross-check
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter


FIT_LOSS             = 'soft_l1'
FIT_F_SCALE_PX       = 0.5
FIT_N_THEO_POINTS    = 1200
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
def fit_profile_simultaneous(x_exp, y_exp,
                             x_axis0, y_apex0, R0_px0, beta0=0.2,
                             bounds=None):
    exp_pts = np.column_stack([x_exp, y_exp])

    def theoretical(R0_px, beta, x_axis, y_apex):
        _, xn, zn, _ = yl_profile(beta)
        xr = x_axis + xn * R0_px
        xl = x_axis - xn * R0_px
        yt = y_apex - zn * R0_px
        return np.column_stack([np.concatenate([xr, xl]),
                                np.concatenate([yt, yt])])

    def residuals(p):
        x_axis, y_apex, logR, beta = p
        try:
            theo = theoretical(np.exp(logR), beta, x_axis, y_apex)
        except Exception:
            return np.full(len(x_exp), 1e3)
        tree = cKDTree(theo)
        d, _ = tree.query(exp_pts, k=1)
        return d

    if bounds is None:
        x_lo, x_hi = float(np.min(x_exp)) - 200, float(np.max(x_exp)) + 200
        y_lo, y_hi = float(np.min(y_exp)) - 200, float(np.max(y_exp)) + 200
        bounds = ([x_lo, y_lo, 2.0, 0.03],
                  [x_hi, y_hi, 10.0, 2.0])

    p0 = [np.clip(x_axis0, bounds[0][0] + 1, bounds[1][0] - 1),
          np.clip(y_apex0, bounds[0][1] + 1, bounds[1][1] - 1),
          float(np.log(R0_px0)),
          float(beta0)]

    result = least_squares(
        residuals, p0,
        method='trf', loss=FIT_LOSS, f_scale=FIT_F_SCALE_PX,
        bounds=bounds,
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=8000,
    )
    x_axis, y_apex, logR, beta = result.x
    R0_px = float(np.exp(logR))
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return dict(x_axis=float(x_axis), y_apex=float(y_apex),
                R0_px=R0_px, beta=float(beta),
                rmse_px=rmse, result=result, logR=float(logR))


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
                      qc_asym_tilt=0.005):
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
        return dict(fit_success=False, error=f"Apex circle fit failed: {e}")
    y_apex0 = float(ys.max())
    x_axis0 = pre_asym['x_axis']

    beta_dsde, DS, DE, ratio = dsde_beta(ys, lxs, rxs)

    fit_mask = ys >= (baseline_y + FIT_TRIM_TOP_PX)
    ys_f, lx_f, rx_f = ys[fit_mask], lx[fit_mask], rx[fit_mask]
    if len(ys_f) < 20:
        ys_f, lx_f, rx_f = ys, lx, rx

    x_exp = np.concatenate([rx_f, lx_f])
    y_exp = np.concatenate([ys_f, ys_f])

    try:
        fit = fit_profile_simultaneous(x_exp, y_exp,
                                       x_axis0, y_apex0, R0_px_init,
                                       beta0=0.2)
    except Exception as e:
        return dict(fit_success=False, error=f"Profile fit failed: {e}")

    if not fit['result'].success and fit['result'].status <= 0:
        return dict(fit_success=False,
                    error=f"Fit did not converge (status={fit['result'].status})")

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
    post_asym = dict(x_axis=fit['x_axis'], trans_rms_px=post_trans,
                     tilt_slope=post_slope, shape_rms_px=post_shape)

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
        ("Pre-fit translation asymmetry",
         pre_asym['trans_rms_px'] <= qc_asym_trans_px,
         f"{pre_asym['trans_rms_px']:.3f} px  (≤ {qc_asym_trans_px})"),
        ("Pre-fit tilt",
         abs(pre_asym['tilt_slope']) <= qc_asym_tilt,
         f"slope = {pre_asym['tilt_slope']:+.4f}  (|·| ≤ {qc_asym_tilt})"),
        ("Post-fit shape asymmetry",
         post_asym['shape_rms_px'] <= qc_asym_trans_px,
         f"{post_asym['shape_rms_px']:.3f} px  (≤ {qc_asym_trans_px})"),
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
        rmse_px=rmse_px, rmse_um=rmse_px * pixel_to_m * 1e6,
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
