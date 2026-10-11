"""
Synthetic pendant-drop images and videos with a known volume, for tests.

A drop is a Young–Laplace profile (``core.physics_calcs.yl_profile``) with
shape factor ``beta`` and apex radius ``R0_px``, hanging from a needle of
fixed width whose tip sits at row ``tip_y``.  The profile is cut where it
meets the needle on the way up from the apex.  Images are drawn at 4×
resolution and averaged down, so edges are anti-aliased like a camera's.

The true volume is the profile's own integral ∫ π x² dz between the apex
and the needle tip, the same quantity the app measures below the baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.physics_calcs import yl_profile

SS = 4      # supersampling factor


def drop_profile(beta, R0_px, needle_half_px):
    """(x, z) in px from the apex up to the needle cut, and the true volume
    in px³.  z = 0 at the apex, increasing upwards."""
    s, x, z, phi = yl_profile(beta, s_max=8.0, n=6000)
    x, z = x * R0_px, z * R0_px
    # the equator: first point where the profile starts narrowing (the
    # integration continues past the neck, where x grows again)
    i_max = int(np.argmax(np.diff(x) < 0))
    above = np.nonzero(x[i_max:] <= needle_half_px)[0]
    if not len(above):
        raise ValueError("profile never narrows to the needle width; "
                         "use a larger beta or a wider needle")
    j = i_max + int(above[0])
    # linear interpolation to the exact needle radius
    f = (x[j - 1] - needle_half_px) / (x[j - 1] - x[j])
    zc = z[j - 1] + f * (z[j] - z[j - 1])
    x, z = np.append(x[:j], needle_half_px), np.append(z[:j], zc)
    vol = float(np.trapezoid(np.pi * x ** 2, z))
    return x, z, vol


def render_drop(beta, R0_px, needle_half_px=40.0, tip_y=60, width=320,
                height=360, bg=225, fg=25):
    """BGR image of the drop + needle; returns (image, volume_px3, apex_y)."""
    x, z, vol = drop_profile(beta, R0_px, needle_half_px)
    H, W = height * SS, width * SS
    cx = W / 2.0
    apex_y = tip_y + z[-1]
    # polygon in supersampled pixel coordinates (row = apex_y - z)
    right = np.column_stack([cx + x * SS, (apex_y - z) * SS])
    left = right[::-1].copy()
    left[:, 0] = 2 * cx - left[:, 0]
    needle = np.array([[cx + needle_half_px * SS, tip_y * SS],
                       [cx + needle_half_px * SS, 0],
                       [cx - needle_half_px * SS, 0],
                       [cx - needle_half_px * SS, tip_y * SS]])
    poly = np.vstack([right, needle[:2], needle[2:], left])
    img = np.full((H, W), bg, np.uint8)
    cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], fg,
                 lineType=cv2.LINE_AA, shift=4)
    small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if apex_y >= height - 5:
        raise ValueError("drop does not fit in the image")
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), vol, apex_y


def write_swelling_video(path, swell_of_t, duration=10.0, fps=10.0,
                         beta=0.30, R0_px=70.0, **render_kw):
    """Write an MJPG .avi whose drop volume grows by the factor
    ``swell_of_t(t)`` (1 at t = 0).  The drop keeps its shape factor and
    grows by scaling R0 with the cube root of the swelling factor.

    Returns (times, true_volumes_px3) for every frame."""
    n = int(round(duration * fps))
    frames, ts, vols = [], [], []
    for k in range(n):
        t = k / fps
        R0 = R0_px * float(swell_of_t(t)) ** (1.0 / 3.0)
        img, vol, _ = render_drop(beta, R0, **render_kw)
        frames.append(img)
        ts.append(t)
        vols.append(vol)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    if not vw.isOpened():
        raise IOError("cannot open a video writer (MJPG)")
    for f in frames:
        vw.write(f)
    vw.release()
    return np.array(ts), np.array(vols)
