"""
Synthetic pendant-drop images with known Young–Laplace parameters.

Ground truth for the Step-3 drop-shape fit (ADSA-P): a Bashforth–Adams
profile of shape factor β and apex radius R₀ (pixels) is drawn as a dark
drop hanging from a vertical needle on a light background, anti-aliased by
supersampling, optionally tilted (camera roll) and noised.

The profile is integrated independently of core.physics_calcs (own RHS,
own tolerances) so a bug there cannot hide itself in the reference.
"""

import cv2
import numpy as np
from scipy.integrate import solve_ivp


def _yl_rhs(s, y, beta):
    phi, x, z = y
    if x < 1e-12:
        return [1.0, np.cos(phi), np.sin(phi)]
    return [2.0 - beta * z - np.sin(phi) / x, np.cos(phi), np.sin(phi)]


def drop_outline(beta, neck_fraction=0.6, s_max=10.0):
    """Right half of the drop in R₀ units, apex at (0, 0), z up.

    Integrates from the apex past the equator (first maximum of x) and
    stops where the upper branch has narrowed to ``neck_fraction`` of the
    equator radius — that radius becomes the needle radius.  If the neck
    (first minimum of x after the equator) is wider than that, the cut is
    moved just below the neck.  Beyond the neck the YL solution continues
    into an unphysical second bulge, which is never drawn.
    Returns (x, z, needle_r).
    """
    eps = 1e-6
    sol = solve_ivp(_yl_rhs, [eps, s_max], [eps, eps, 0.5 * eps * eps],
                    args=(beta,), rtol=1e-11, atol=1e-13, max_step=0.005)
    phi, x, z = sol.y
    dx = np.diff(x)
    i_eq = int(np.argmax(dx < 0))                  # first x maximum
    if i_eq == 0:
        raise ValueError(f"no equator within s_max (beta={beta})")
    rest = np.diff(x[i_eq:])
    i_neck = i_eq + (int(np.argmax(rest > 0)) if np.any(rest > 0) else len(rest))
    target = max(neck_fraction * x[i_eq], 1.10 * x[i_neck])
    i_cut = i_eq + int(np.argmax(x[i_eq:i_neck + 1] <= target))
    return x[:i_cut + 1], z[:i_cut + 1], float(x[i_cut])


def render_drop(beta, R0_px, tilt_deg=0.0, noise_std=0.0, margin_px=60,
                needle_len_px=None, neck_fraction=0.6, supersample=8,
                seed=0):
    """Render a drop image.  Returns (gray uint8 image, truth dict).

    truth: beta, R0_px, apex (x, y) in pixels, needle_width_px, tilt_deg.
    Pixel centres are at integer coordinates (OpenCV convention).
    """
    x, z, rn = drop_outline(beta, neck_fraction)
    x, z, rn = x * R0_px, z * R0_px, rn * R0_px
    h_drop = float(z[-1])
    if needle_len_px is None:
        needle_len_px = 1.5 * h_drop
    w_max = float(x.max())

    # Outline polygon in (u, v) with v pointing DOWN (image rows), apex at v=0.
    right = np.column_stack([x, -z])
    top = -(h_drop + needle_len_px)
    poly = np.vstack([right,
                      [[rn, top], [-rn, top]],
                      np.column_stack([-x[::-1], -z[::-1]])])

    t = np.deg2rad(tilt_deg)               # roll the whole scene about the apex
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    poly = poly @ rot.T

    W = int(np.ceil(2 * (w_max + margin_px)))
    H = int(np.ceil(h_drop + 0.7 * needle_len_px + margin_px))
    apex = np.array([W / 2.0, H - margin_px], dtype=float)
    poly_img = poly + apex                 # the needle runs off the top edge

    S = int(supersample)
    canvas = np.full((H * S, W * S), 235, np.uint8)
    # Pixel (i, j) covers [j-0.5, j+0.5]: scale about the -0.5 corner.
    pts = np.round((poly_img + 0.5) * S * 16).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 20, lineType=cv2.LINE_8, shift=4)
    img = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(float) + rng.normal(0, noise_std, img.shape),
                      0, 255).astype(np.uint8)

    truth = dict(beta=float(beta), R0_px=float(R0_px),
                 apex=(float(apex[0]), float(apex[1])),
                 needle_width_px=2.0 * rn, tilt_deg=float(tilt_deg),
                 drop_height_px=h_drop)
    return img, truth
