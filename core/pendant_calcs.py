"""
core.pendant_calcs
==================

Pendant-drop image utility: sub-pixel contour extraction.

Overlay drawing has moved to Qt graphics items in gui.pendant_window
so lines stay visible at every zoom and don't alter the underlying pixmap.
"""

import cv2
import numpy as np
import pandas as pd

from core.image_processing import subpixel_refine_edges


def extract_pendant_edges(img, use_subpixel=True):
    """Threshold → external contour → per-row (left_x, right_x) DataFrame.

    With `use_subpixel=True` (default), edges are refined via parabolic
    peak fit on |∂I/∂x|. The dilation trick from the old path is disabled
    in that mode because it would drag the refined edge off the gradient
    peak.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blurred, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if not use_subpixel:
        bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea).squeeze()
    cdf = pd.DataFrame(main_contour, columns=['x', 'y'])
    edges = (cdf.groupby('y')['x']
                .agg(['min', 'max'])
                .rename(columns={'min': 'left_x', 'max': 'right_x'})
                .sort_index()
                .reset_index())

    if use_subpixel:
        edges = subpixel_refine_edges(blurred, edges)

    return edges.reset_index(drop=True)
