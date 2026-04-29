"""
core.image_processing
=====================

Shared utilities:
    · subpixel_refine_edges          — parabolic peak fit on |∂I/∂x|
    · robust_stable_needle_width     — median + sigma-clip
    · attempt_auto_calibration       — two-stage robust calibration
    · attempt_auto_baseline          — first row widening beyond the needle
"""

import cv2
import numpy as np
import pandas as pd


def subpixel_refine_edges(img_blur, edges_df, search_halfwin=3):
    """Refine per-row (left_x, right_x) to ~0.1 px via parabolic peak fit
    on |∂I/∂x|. Rows dominated by horizontal gradient (side walls) shift
    meaningfully; near-apex rows are left unchanged."""
    gx = cv2.Sobel(img_blur.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    agx = np.abs(gx)
    H, W = agx.shape
    hw = int(search_halfwin)

    def _refine(ys, xs):
        out = xs.astype(np.float64).copy()
        for i, (yv, xv) in enumerate(zip(ys, xs)):
            yi = int(yv); xi = int(round(xv))
            if xi < hw or xi > W - hw - 1 or yi < 0 or yi >= H:
                continue
            win = agx[yi, xi - hw:xi + hw + 1]
            if win.size < 3:
                continue
            k = int(np.argmax(win))
            if 0 < k < win.size - 1:
                a, b, c = win[k - 1], win[k], win[k + 1]
                den = a - 2 * b + c
                offset = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
                out[i] = (xi - hw) + k + offset
            else:
                out[i] = (xi - hw) + k
        return out

    ys = edges_df['y'].to_numpy()
    e = edges_df.copy()
    e['left_x']  = _refine(ys, edges_df['left_x'].to_numpy())
    e['right_x'] = _refine(ys, edges_df['right_x'].to_numpy())
    return e


def robust_stable_needle_width(edges_df, pixel_to_m, needle_tip_diameter_mm,
                               sample_length_mm=2.0, skip_top_rows=5,
                               min_rows=10, sigma_clip=2.5):
    ed = edges_df.reset_index(drop=True).copy()
    pixels_per_mm = 1.0 / (pixel_to_m * 1000.0)
    n_rows = max(min_rows, int(round(sample_length_mm * pixels_per_mm)))
    start_idx = skip_top_rows
    if len(ed) < (start_idx + n_rows):
        n_rows = len(ed) - start_idx
    if n_rows <= 0:
        return None
    sample = ed.iloc[start_idx:start_idx + n_rows]
    widths = (sample['right_x'] - sample['left_x']).to_numpy()
    widths = widths[widths > 0]
    if len(widths) == 0:
        return None
    med = np.median(widths); std = np.std(widths)
    if std > 0:
        mask = np.abs(widths - med) <= sigma_clip * std
        clipped = widths[mask]
        if len(clipped) > 2:
            widths = clipped
    return float(np.median(widths))


def attempt_auto_calibration(edges_df, needle_tip_diameter_mm=0.7176,
                             sample_length_mm=2.0, skip_top_rows=5,
                             min_sample_rows=10, sigma_clip=2.5):
    try:
        if len(edges_df) < 20:
            return None, None
        avg_top = np.median((edges_df['right_x'] - edges_df['left_x']).iloc[:10])
        if avg_top <= 0 or np.isnan(avg_top):
            return None, None
        guess_m = (needle_tip_diameter_mm / 1000.0) / avg_top
        width_px = robust_stable_needle_width(
            edges_df, guess_m, needle_tip_diameter_mm,
            sample_length_mm, skip_top_rows, min_sample_rows, sigma_clip)
        if width_px is None or width_px <= 0:
            return None, None
        pixel_to_m = (needle_tip_diameter_mm / 1000.0) / width_px
        return pixel_to_m, width_px
    except Exception:
        return None, None


def attempt_auto_baseline(edges_df, width_px, deviation_percent=5.0,
                          start_skip_rows=10):
    try:
        ys = edges_df['y'].to_numpy()
        widths = (edges_df['right_x'] - edges_df['left_x']).to_numpy()
        thresh = width_px * (1.0 + deviation_percent / 100.0)
        mask = widths > thresh
        indices = np.where(mask)[0]
        valid = indices[indices > start_skip_rows]
        if len(valid) > 0:
            return int(ys[valid[0]])
        return None
    except Exception:
        return None
