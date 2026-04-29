"""
core.domain_builder
===================

Builds a **complete**, closed FEM domain polygon for a pendant drop.
The polygon represents the FULL 2-D cross-section of the liquid
(both left and right sides) — no symmetry-axis boundary.

Included regions
----------------
    · Liquid column inside the needle  (between ±r_inner, z = 0 … neck_h)
    · Tip step on both sides           (r_inner → r_outer at z = neck_h)
    · Free-surface drop contour        (both sides, rounded at the apex)

Coordinate convention
---------------------
    r : horizontal coordinate (positive = right, negative = left)
    z : axial, increases downward (gravity direction)
    z = 0 at the top of the neck
    baseline (needle tip) at z = neck_height
    apex at z = neck_height + drop_height

Polygon winding
---------------
    Counter-clockwise (standard for Gmsh / pygmsh).

Boundary layout (CCW starting top-right)
-----------------------------------------
    top_wall           : (-r_inner, 0) → (+r_inner, 0)
    right_inner_wall   : (+r_inner, 0) → (+r_inner, neck_h)
    right_tip_step     : (+r_inner, neck_h) → (+r_outer, neck_h)
    free_surface_right : drop contour from (+r_outer, neck_h) → apex
    free_surface_left  : drop contour from apex → (-r_outer, neck_h)
    left_tip_step      : (-r_outer, neck_h) → (-r_inner, neck_h)
    left_inner_wall    : (-r_inner, neck_h) → (-r_inner, 0)
"""

import numpy as np


# ======================================================================
# Main polygon builder
# ======================================================================
def build_domain_polygon(r_m, z_m,
                         r_inner_m, r_outer_m,
                         neck_height_factor=0.8,
                         neck_height_m=None,
                         apex_recon_points=8,
                         apex_smooth_points=6):
    """
    Build a complete closed FEM domain polygon for a pendant drop.

    Parameters
    ----------
    r_m : array_like
        Radius (half-width) of the drop contour (metres),
        ordered from baseline toward apex.
    z_m : array_like
        Axial position (metres), z = 0 at the baseline,
        increasing downward toward the apex.
    r_inner_m : float
        Inner radius of the needle (metres).
    r_outer_m : float
        Outer radius of the needle (metres).
    neck_height_factor : float
        Neck height as a fraction of the drop height.
    neck_height_m : float or None
        Explicit neck height (metres).  Overrides *neck_height_factor*.

    Returns
    -------
    polygon : (N, 2) ndarray
        Closed polygon ``(r, z)`` — full cross-section.
    metadata : dict
        Geometry info and segment boundary indices.
    """
    r = np.asarray(r_m, dtype=float).copy()
    z = np.asarray(z_m, dtype=float).copy()

    # Sort by z (baseline → apex, ascending z)
    idx = np.argsort(z)
    r = r[idx]
    z = z[idx]

    # --- neck height ---
    drop_height = z.max() - z.min()
    if neck_height_m is None:
        neck_height_m = neck_height_factor * drop_height

    # Shift z so z = 0 is at the top of the neck (baseline → neck_height_m)
    z_shifted = z + neck_height_m

    # Force the first contour point to match the outer radius
    # (eliminates the gap between the tip step and the contour)
    r[0] = r_outer_m

    # ---- Apex reconstruction if the contour doesn't reach r ≈ 0 ----
    if r[-1] > 1e-6:
        n_fit = min(apex_recon_points, len(r) - 1)
        r_tail = r[-n_fit:]
        z_tail = z_shifted[-n_fit:]
        z0 = z_tail[-1]
        dz = z_tail - z0
        mask = dz != 0

        if np.sum(mask) >= 2:
            # Parabolic fit: r ≈ a · (z − z0)²
            a = np.mean(r_tail[mask] / (dz[mask] ** 2 + 1e-14))
            dz_step = max(abs(z_tail[-1] - z_tail[-2]), 1e-6)
            z_new = np.linspace(z_tail[-1] + dz_step * 0.2,
                                z_tail[-1] + dz_step * 1.5,
                                apex_recon_points)
            r_new = np.maximum(a * (z_new - z0) ** 2, 0.0)
            r_new = np.minimum(r_new, r_tail[-1])   # monotone decrease
            r_new[-1] = 0.0
            r = np.concatenate([r, r_new])
            z_shifted = np.concatenate([z_shifted, z_new])

    # ---- Smooth near apex (linear taper to 0) ----
    n_smooth = min(apex_smooth_points, len(r) - 1)
    if n_smooth > 2:
        r[-n_smooth:] = np.linspace(r[-n_smooth], 0.0, n_smooth)

    # Enforce apex at exactly r = 0
    r[-1] = 0.0

    # ================================================================
    # Build the FULL polygon (both sides)
    # ================================================================
    pts = []

    # --- Right neck (top → baseline) ---
    i_start_right_wall = 0
    pts.append([r_inner_m, 0.0])                   # top-right
    pts.append([r_inner_m, neck_height_m])          # baseline-right (inner)
    i_end_right_wall = len(pts) - 1

    # --- Right tip step (inner → outer at baseline) ---
    i_start_right_tip = len(pts) - 1
    pts.append([r_outer_m, neck_height_m])          # baseline-right (outer)
    i_end_right_tip = len(pts) - 1

    # --- Right free surface (baseline → apex) ---
    i_start_right_fs = len(pts) - 1
    right_contour = np.column_stack([r, z_shifted])
    for pt in right_contour:
        pts.append(pt.tolist())
    i_end_right_fs = len(pts) - 1

    # --- Left free surface (apex → baseline, mirrored)
    # Skip the apex point (already the last point of right contour)
    i_start_left_fs = len(pts) - 1
    r_rev = r[::-1][1:]       # reversed, skip apex
    z_rev = z_shifted[::-1][1:]
    left_contour = np.column_stack([-r_rev, z_rev])
    for pt in left_contour:
        pts.append(pt.tolist())
    i_end_left_fs = len(pts) - 1

    # --- Left tip step (outer → inner at baseline) ---
    i_start_left_tip = len(pts) - 1
    pts.append([-r_outer_m, neck_height_m])
    pts.append([-r_inner_m, neck_height_m])
    i_end_left_tip = len(pts) - 1

    # --- Left inner wall (baseline → top) ---
    i_start_left_wall = len(pts) - 1
    pts.append([-r_inner_m, 0.0])                  # top-left
    i_end_left_wall = len(pts) - 1

    # --- Top wall (left → right, close the polygon) ---
    i_start_top = len(pts) - 1
    pts.append([r_inner_m, 0.0])                   # back to start
    i_end_top = len(pts) - 1

    polygon = np.array(pts, dtype=float)

    # Remove duplicate consecutive points
    keep = [True]
    for i in range(1, len(polygon)):
        if np.linalg.norm(polygon[i] - polygon[i - 1]) > 1e-12:
            keep.append(True)
        else:
            keep.append(False)
    polygon = polygon[np.array(keep)]

    # Ensure closure
    if not np.allclose(polygon[0], polygon[-1], atol=1e-12):
        polygon = np.vstack([polygon, polygon[0]])

    metadata = dict(
        neck_height_m=neck_height_m,
        drop_height_m=drop_height,
        r_inner_m=r_inner_m,
        r_outer_m=r_outer_m,
        n_points=len(polygon),
        # Segment index ranges (approximate — may shift slightly
        # after dedup, but good enough for visualisation / labelling)
        segment_indices=dict(
            right_inner_wall=(i_start_right_wall, i_end_right_wall),
            right_tip_step=(i_start_right_tip, i_end_right_tip),
            free_surface_right=(i_start_right_fs, i_end_right_fs),
            free_surface_left=(i_start_left_fs, i_end_left_fs),
            left_tip_step=(i_start_left_tip, i_end_left_tip),
            left_inner_wall=(i_start_left_wall, i_end_left_wall),
            top_wall=(i_start_top, i_end_top),
        ),
    )

    return polygon, metadata


# ======================================================================
# Convenience bridge for BTP_v2 data format
# ======================================================================
def contour_to_domain(edges_df, baseline_y, pixel_to_m,
                      needle_od_mm, needle_id_mm=None,
                      neck_height_factor=0.8):
    """
    Convert BTP_v2 pendant contour data directly into a closed FEM
    domain polygon.

    Parameters
    ----------
    edges_df : DataFrame
        Must contain columns ``y``, ``left_x``, ``right_x``.
    baseline_y : int or float
        Y coordinate (pixels) of the baseline.
    pixel_to_m : float
        Scale factor (metres per pixel).
    needle_od_mm : float
        Needle outer diameter in mm.
    needle_id_mm : float or None
        Needle inner diameter in mm.  If None, 70 % of OD is assumed.
    neck_height_factor : float
        Fraction of drop height used for the neck extension.

    Returns
    -------
    polygon, metadata : same as ``build_domain_polygon``.
    """
    # Extract drop portion (at and below baseline)
    drop = edges_df[edges_df['y'] >= baseline_y].copy()
    drop = drop.sort_values('y').reset_index(drop=True)

    ys = drop['y'].to_numpy(dtype=float)
    lx = drop['left_x'].to_numpy(dtype=float)
    rx = drop['right_x'].to_numpy(dtype=float)

    # Compute radius and z in metres
    r_m = 0.5 * (rx - lx) * pixel_to_m
    z_m = (ys - baseline_y) * pixel_to_m   # z = 0 at baseline

    # Needle radii in metres
    r_outer = (needle_od_mm / 1000.0) / 2.0
    if needle_id_mm is not None and needle_id_mm > 0:
        r_inner = (needle_id_mm / 1000.0) / 2.0
    else:
        r_inner = r_outer * 0.70

    return build_domain_polygon(
        r_m, z_m,
        r_inner_m=r_inner,
        r_outer_m=r_outer,
        neck_height_factor=neck_height_factor,
    )
