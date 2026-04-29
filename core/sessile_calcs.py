"""
core.sessile_calcs
==================

Sessile-drop contact-angle analysis assuming user-provided baseline and ROI.

Two independent angle measurements are returned per contact point:

    CONTOUR method  (primary, physically meaningful):
        A local polynomial is fit to the detected contour points in a
        neighborhood of each contact point, and its tangent at the
        contact is used to compute the contact angle. Reflects the
        ACTUAL drop shape, including any local deformation that a
        global circle cannot capture.

    CIRCLE method  (secondary reference):
        A weighted LSQ circle is fit to the whole contour — with the
        apex and the two detected contact points heavily weighted — so
        the circle is constrained to pass through the three anchor
        points and cannot wander outside a sensibly-drawn ROI.
        Tangent at each contact is perpendicular to the radial vector.

Pipeline
--------
    1. Binary segmentation inside the user's ROI, above the baseline
    2. Outer contour extraction with sub-pixel refinement
    3. TRUE contact-point detection: leftmost/rightmost contour points
       within a small band along the baseline, inside the ROI
    4. Apex = highest contour point (perpendicular distance) above baseline
    5. Weighted LSQ circle fit anchored at (left, apex, right)
    6. Contour-tangent angle via local polynomial fit at each contact
       (primary result)
    7. Circle-tangent angle at each contact (secondary result)
    8. Spherical-cap volume, Bond number, RMS residuals
"""

import cv2
import numpy as np

from core.image_processing import subpixel_refine_edges


# ===========================================================================
# Geometry primitives
# ===========================================================================
def _baseline_line(baseline_data):
    """Return (m, c) such that y = m·x + c for the baseline."""
    if baseline_data[0] == '1pt':
        return 0.0, float(baseline_data[1])
    x1, y1, x2, y2 = baseline_data[1]
    if x1 == x2:
        x2 = x2 + 1e-5
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return float(m), float(c)


def _perp_distance_to_baseline(px, py, m, c):
    """Signed distance: positive when point is ABOVE the baseline (smaller y)."""
    return (m * px + c - py) / np.sqrt(m * m + 1.0)


# ===========================================================================
# Weighted circle fit  (anchored — cannot bulge outside the ROI)
# ===========================================================================
def fit_circle_weighted_lsq(points, weights=None):
    """Algebraic Kasa circle fit with optional per-point weights.

    Solves  |p_i − C|² = R²  in linearised form  A·θ = b  where
        θ = (cx, cy, k),  k = R² − cx² − cy²
    Each row is scaled by √w_i so high-weight points dominate the residual.
    """
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        sw = np.sqrt(np.clip(w, 0, None))
        A = A * sw[:, None]
        b = b * sw
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = theta
    R2 = k + cx * cx + cy * cy
    if R2 <= 0:
        raise RuntimeError("Circle fit produced non-positive radius.")
    return (float(cx), float(cy)), float(np.sqrt(R2))


# ===========================================================================
# Contact-point & apex detection (from actual contour, not from circle)
# ===========================================================================
def _detect_contact_points(contour_pts, m, c, search_band_px=None):
    """Find the true contact points: leftmost and rightmost contour
    points within a small band above the baseline.

    Old behaviour intersected the fitted circle with the baseline, which
    could place the "contact" off the real contour entirely when the
    circle wandered outside the ROI. This version picks actual pixels
    off the refined contour.
    """
    pts = np.asarray(contour_pts, dtype=float)
    d = (m * pts[:, 0] + c - pts[:, 1]) / np.sqrt(m * m + 1.0)

    if search_band_px is None:
        span = pts[:, 0].max() - pts[:, 0].min()
        search_band_px = float(np.clip(0.05 * span, 3.0, 15.0))

    mask = (d >= 0.0) & (d <= search_band_px)
    candidates = pts[mask]
    widened = search_band_px
    while len(candidates) < 4 and widened < 3.0 * search_band_px:
        widened *= 1.5
        mask = (d >= -0.5) & (d <= widened)
        candidates = pts[mask]
    if len(candidates) < 2:
        order = np.argsort(d)
        candidates = pts[order[:max(4, len(pts) // 20)]]

    contact_left  = candidates[np.argmin(candidates[:, 0])]
    contact_right = candidates[np.argmax(candidates[:, 0])]

    # Snap y EXACTLY to the baseline — the contact sits ON it by definition
    xL = float(contact_left[0]); yL = m * xL + c
    xR = float(contact_right[0]); yR = m * xR + c
    return (xL, yL), (xR, yR), float(widened)


def _detect_apex(contour_pts, m, c):
    """Apex = point with the largest perpendicular distance above the
    baseline. Works for any baseline tilt."""
    pts = np.asarray(contour_pts, dtype=float)
    d = (m * pts[:, 0] + c - pts[:, 1]) / np.sqrt(m * m + 1.0)
    idx = int(np.argmax(d))
    return (float(pts[idx, 0]), float(pts[idx, 1]))


# ===========================================================================
# Contour-tangent angle  (PRIMARY method)
# ===========================================================================
def contour_tangent_angle(contour_pts, contact_pt, side, m_base,
                          window_px=None, min_points=6, poly_degree=2):
    """Fit a local polynomial to the contour near the contact point in
    a coordinate frame rotated to the baseline, then take its tangent
    at the contact to compute the contact angle.

    Why rotate to the baseline frame: if the baseline is at a non-trivial
    slope and the drop angle approaches 90°, a polynomial y=f(x) in IMAGE
    coordinates becomes singular. Working in the baseline-aligned frame
    (u along baseline, v perpendicular) keeps v=f(u) well-posed.

    Returns (theta_deg, tangent_unit_vec_in_image_coords, n_points_used).
    Returns NaN + zero vector when the fit can't be trusted.
    """
    pts = np.asarray(contour_pts, dtype=float)

    # Rotate into baseline frame. Baseline direction in image:
    #   (cos θ_base, sin θ_base)  where θ_base = atan(m_base).
    # The v axis is the baseline-perpendicular direction with v>0 meaning
    # ABOVE the baseline (i.e. on the drop side). In image coords, "above"
    # is smaller y, so we flip the natural perp sign.
    theta_base = np.arctan(m_base)
    cs, sn = np.cos(theta_base), np.sin(theta_base)
    dx = pts[:, 0] - contact_pt[0]
    dy = pts[:, 1] - contact_pt[1]
    u_all =  cs * dx + sn * dy
    v_all = -(-sn * dx + cs * dy)         # flip so above-baseline → v>0

    if window_px is None:
        span = float(pts[:, 0].max() - pts[:, 0].min())
        window_px = max(10.0, 0.12 * span)

    # Directional mask so we don't accidentally pick up the opposite side.
    # Left contact: interior lies in +u direction.  Right contact: −u.
    near_in_u = np.abs(u_all) <= window_px
    above_v   = v_all > -1.0
    if side == 'left':
        directional = u_all >= -0.5
    else:
        directional = u_all <= 0.5
    mask = near_in_u & above_v & directional
    u = u_all[mask]; v = v_all[mask]

    if len(u) < min_points:
        # Relax the directional cut as a fallback
        mask = near_in_u & above_v
        u = u_all[mask]; v = v_all[mask]
        if len(u) < min_points:
            return float('nan'), np.array([0.0, 0.0]), int(len(u))

    # v = a u^2 + b u + c
    try:
        coeffs = np.polyfit(u, v, poly_degree)
    except (np.linalg.LinAlgError, ValueError):
        return float('nan'), np.array([0.0, 0.0]), int(len(u))

    # Tangent slope at u=0 is the coefficient of u (for deg 2):  dv/du = b
    deriv = np.polyder(coeffs)
    dvdu_at_contact = float(np.polyval(deriv, 0.0))

    # Tangent (du, dv) in the rotated frame, oriented into the drop
    tu, tv = 1.0, dvdu_at_contact
    if side == 'right':
        tu, tv = -tu, -tv
    n = np.hypot(tu, tv)
    tu /= n; tv /= n

    # Contact angle is the angle between (tu, tv) and the baseline
    # direction pointing into the footprint, which is simply +u for
    # left contact and −u for right contact (both have bv=0 in the
    # rotated frame).
    if side == 'left':
        bu, bv = 1.0, 0.0
    else:
        bu, bv = -1.0, 0.0
    cos_th = np.clip(tu * bu + tv * bv, -1.0, 1.0)
    theta = float(np.degrees(np.arccos(cos_th)))

    # Map tangent back to image coordinates for drawing. Reverse rotation
    # and reverse the v-flip we did earlier.
    tx = cs * tu - sn * (-tv)
    ty = sn * tu + cs * (-tv)
    n2 = np.hypot(tx, ty)
    return theta, np.array([tx / n2, ty / n2]), int(len(u))


# ===========================================================================
# Circle-tangent angle  (secondary)
# ===========================================================================
def circle_tangent_angle(cx, cy, R, px, py, m):
    """Angle from the circle's tangent at point P, vs baseline pointing
    into the footprint. Tangent ⊥ radial C→P, oriented upward."""
    rx, ry = px - cx, py - cy
    tx, ty = -ry, rx
    if ty > 0:
        tx, ty = -tx, -ty
    side_sign = +1.0 if px < cx else -1.0
    bx = side_sign * 1.0
    by = side_sign * m
    n_t = np.hypot(tx, ty)
    n_b = np.hypot(bx, by)
    if n_t == 0 or n_b == 0:
        return float('nan'), np.array([0.0, 0.0])
    cos_th = np.clip((tx * bx + ty * by) / (n_t * n_b), -1.0, 1.0)
    theta = float(np.degrees(np.arccos(cos_th)))
    return theta, np.array([tx / n_t, ty / n_t])


# ===========================================================================
# Physical helpers
# ===========================================================================
def capillary_length(gamma=0.072, rho=1000.0, g=9.81):
    return float(np.sqrt(gamma / (rho * g)))


def bond_number_from_radius(radius_m, gamma=0.072, rho=1000.0, g=9.81):
    lc = capillary_length(gamma, rho, g)
    return (radius_m / lc) ** 2, lc


def spherical_cap_volume(R_m, h_m):
    return float(np.pi * h_m * h_m * (3.0 * R_m - h_m) / 3.0)


# ===========================================================================
# Contour extraction (unchanged)
# ===========================================================================
def _extract_drop_contour(img_bgr, baseline_m, baseline_c, roi_poly):
    """Outer drop contour, restricted to inside the ROI and above the
    baseline. Returns Nx2 sub-pixel-refined points."""
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blurred, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours detected.")
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2:
        raise RuntimeError("Contour degenerate.")

    keep = np.ones(len(contour), dtype=bool)
    if roi_poly is not None:
        poly = np.asarray(roi_poly, dtype=np.int32)
        H, W = bw.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        keep &= mask[contour[:, 1], contour[:, 0]] > 0

    # Above baseline (tolerance allows contacts exactly on the baseline)
    y_line_expected = baseline_m * contour[:, 0] + baseline_c
    keep &= contour[:, 1] <= y_line_expected + 1
    contour_kept = contour[keep]
    if len(contour_kept) < 12:
        raise RuntimeError("Not enough contour points above the baseline.")

    import pandas as pd
    cdf = pd.DataFrame(contour_kept, columns=['x', 'y'])
    edges = (cdf.groupby('y')['x']
                .agg(['min', 'max'])
                .rename(columns={'min': 'left_x', 'max': 'right_x'})
                .reset_index())
    edges = subpixel_refine_edges(blurred, edges)
    rx = edges['right_x'].to_numpy(dtype=float)
    lx = edges['left_x'].to_numpy(dtype=float)
    ys = edges['y'].to_numpy(dtype=float)
    refined = np.vstack([
        np.column_stack([rx, ys]),
        np.column_stack([lx, ys])
    ])
    return refined


# ===========================================================================
# Main entry point
# ===========================================================================
def analyze_sessile_drop(image_input,
                         baseline_data,
                         roi_poly=None,
                         px_to_m=None,
                         gamma_hint=0.072,
                         rho=1000.0,
                         g=9.81,
                         bo_warn_threshold=0.1,
                         bo_fail_threshold=0.5,
                         anchor_weight=80.0):
    """Run full analysis. Returns a dict with both angle sets.

    anchor_weight: weight given to apex and detected contact points in
    the LSQ circle fit, relative to unit weight for ordinary contour
    points. A value of 50–100 forces the circle to pass very close to
    those three anchors, so it cannot bulge outside the ROI.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    else:
        img = image_input.copy()

    m_base, c_base = _baseline_line(baseline_data)
    contour_pts = _extract_drop_contour(
        img, m_base, c_base,
        np.asarray(roi_poly) if roi_poly is not None else None)

    # --- 1. True contact points from contour ---
    contact_left, contact_right, used_band = _detect_contact_points(
        contour_pts, m_base, c_base)

    # --- 2. Apex from contour ---
    apex_pt = _detect_apex(contour_pts, m_base, c_base)

    # --- 3. CONTOUR-tangent angles (primary) ---
    footprint_px = float(np.hypot(contact_right[0] - contact_left[0],
                                  contact_right[1] - contact_left[1]))
    ct_window = max(10.0, 0.18 * footprint_px)

    theta_l_c, tangent_l_c, n_l_c = contour_tangent_angle(
        contour_pts, contact_left,  'left',  m_base, window_px=ct_window)
    theta_r_c, tangent_r_c, n_r_c = contour_tangent_angle(
        contour_pts, contact_right, 'right', m_base, window_px=ct_window)
    contour_valid = not (np.isnan(theta_l_c) or np.isnan(theta_r_c))

    # --- 4. Weighted circle fit anchored at apex + contacts ---
    anchor_pts = np.array([contact_left, apex_pt, contact_right], dtype=float)
    all_pts = np.vstack([contour_pts, anchor_pts])
    weights = np.ones(len(all_pts))
    weights[-3:] = float(anchor_weight)
    (cx, cy), R = fit_circle_weighted_lsq(all_pts, weights=weights)

    d = np.sqrt((contour_pts[:, 0] - cx) ** 2 +
                (contour_pts[:, 1] - cy) ** 2) - R
    rms_circle = float(np.sqrt(np.mean(d * d)))

    # --- 5. CIRCLE-tangent angles (secondary) ---
    theta_l_s, tangent_l_s = circle_tangent_angle(
        cx, cy, R, *contact_left,  m_base)
    theta_r_s, tangent_r_s = circle_tangent_angle(
        cx, cy, R, *contact_right, m_base)

    # --- 6. Physical derived ---
    cap_height_px = float(abs(
        _perp_distance_to_baseline(apex_pt[0], apex_pt[1], m_base, c_base)))

    volume_uL = None
    Bo = lc = R_m = None
    suitability = "Unknown (no scale)"
    if px_to_m is not None and px_to_m > 0:
        R_m = R * px_to_m
        h_m = cap_height_px * px_to_m
        volume_uL = spherical_cap_volume(R_m, h_m) * 1e9
        Bo, lc = bond_number_from_radius(R_m, gamma=gamma_hint, rho=rho, g=g)
        if Bo >= bo_fail_threshold:
            suitability = "Fail (Gravity)"
        elif Bo >= bo_warn_threshold:
            suitability = "Warn (Gravity)"
        else:
            suitability = "OK"

    avg_contour = (float(np.nanmean([theta_l_c, theta_r_c]))
                   if contour_valid else float('nan'))
    avg_circle  = float(np.nanmean([theta_l_s, theta_r_s]))
    asym_contour = (float(abs(theta_l_c - theta_r_c))
                    if contour_valid else float('nan'))
    asym_circle  = float(abs(theta_l_s - theta_r_s))

    return dict(
        # shared geometry
        contour_pts=contour_pts,
        baseline=(m_base, c_base),
        contact_left_px=contact_left,
        contact_right_px=contact_right,
        contact_search_band_px=used_band,
        apex_px=apex_pt,
        cap_height_px=cap_height_px,
        footprint_px=footprint_px,

        # circle fit + circle-tangent
        circle_center=(cx, cy),
        circle_radius_px=R,
        rms_residual_px=rms_circle,
        circle_left_angle_deg=theta_l_s,
        circle_right_angle_deg=theta_r_s,
        circle_avg_angle_deg=avg_circle,
        circle_angle_asymmetry_deg=asym_circle,
        circle_tangent_left=tangent_l_s,
        circle_tangent_right=tangent_r_s,

        # contour-tangent (primary)
        contour_left_angle_deg=theta_l_c,
        contour_right_angle_deg=theta_r_c,
        contour_avg_angle_deg=avg_contour,
        contour_angle_asymmetry_deg=asym_contour,
        contour_tangent_left=tangent_l_c,
        contour_tangent_right=tangent_r_c,
        contour_npts_left=n_l_c,
        contour_npts_right=n_r_c,
        contour_fit_valid=contour_valid,

        # physical
        bond_number=Bo,
        capillary_length_m=lc,
        radius_m=R_m,
        volume_uL=volume_uL,
        suitability=suitability,
    )
