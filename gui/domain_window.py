"""
gui.domain_window
==================

Visualization window for the FULL pendant-drop FEM domain polygon.
Displays:
    · Complete closed domain boundary (both sides of the drop + neck)
    · Needle wall (OD) outline with diagonal hatching
    · Colour-coded boundary segments
    · Domain statistics
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QFrame)
from PyQt6.QtCore import Qt
from gui.styles import DARK_STYLESHEET


# Segment colours (r, g, b)
_SEG_COLORS = {
    'right_inner_wall':  (255, 160,  60),   # orange
    'left_inner_wall':   (255, 160,  60),   # orange
    'right_tip_step':    (255, 100, 100),   # red
    'left_tip_step':     (255, 100, 100),   # red
    'free_surface_right':(100, 200, 255),   # cyan
    'free_surface_left': (100, 200, 255),   # cyan
    'top_wall':          (180, 180, 180),   # grey
}


# ======================================================================
# Hatching helper
# ======================================================================
def _hatch_rect(r0, r1, z0, z1, n_lines=12):
    """Generate diagonal hatch lines (upper-left → lower-right) inside
    the rectangle [r0, r1] × [z0, z1].

    Returns xs, ys arrays with NaN separators for batch plotting.
    """
    xs, ys = [], []
    c_start = r0 + z0
    c_end = r1 + z1

    for c in np.linspace(c_start, c_end, n_lines + 2)[1:-1]:
        t_start = max(r0, c - z1)
        t_end   = min(r1, c - z0)
        if t_start < t_end:
            xs.extend([t_start, t_end, np.nan])
            ys.extend([c - t_start, c - t_end, np.nan])

    return np.array(xs), np.array(ys)


class DomainWindow(QWidget):
    """Window showing the full pendant-drop domain polygon."""

    def __init__(self, polygon, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FEM Domain — Pendant Drop")
        self.resize(900, 750)
        self.setStyleSheet(DARK_STYLESHEET)

        self.polygon = polygon
        self.metadata = metadata

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- plot ----
        pg.setConfigOption('background', '#0f1018')
        pg.setConfigOption('foreground', '#c3c8dc')

        self.plot_widget = pg.PlotWidget(title="Pendant Drop Domain (Full)")
        self.plot_widget.setLabel('bottom', 'r  (mm)')
        self.plot_widget.setLabel('left',   'z  (mm)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.invertY(True)      # z increases downward
        root.addWidget(self.plot_widget, stretch=4)

        # ---- info card ----
        info_card = QFrame(objectName="card")
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(14, 10, 14, 10)
        info_layout.setSpacing(24)

        # Statistics
        stats_box = QGroupBox("Domain statistics")
        stats_vl = QVBoxLayout()
        r_max = polygon[:, 0].max() * 1e3
        z_max = polygon[:, 1].max() * 1e3
        neck_h = metadata['neck_height_m'] * 1e3
        drop_h = metadata['drop_height_m'] * 1e3
        ri = metadata['r_inner_m'] * 1e3
        ro = metadata['r_outer_m'] * 1e3
        n_pts = metadata['n_points']
        stats_text = (
            f"Points:      {n_pts}\n"
            f"Max radius:  {r_max:.4f} mm\n"
            f"Total height:{z_max:.4f} mm\n"
            f"Neck height: {neck_h:.4f} mm\n"
            f"Drop height: {drop_h:.4f} mm\n"
            f"Needle ID:   {2*ri:.4f} mm\n"
            f"Needle OD:   {2*ro:.4f} mm"
        )
        lbl_stats = QLabel(stats_text)
        lbl_stats.setStyleSheet(
            "color:#c3c8dc; font-family:'Consolas','Menlo',monospace; "
            "font-size:9.5pt;")
        stats_vl.addWidget(lbl_stats)
        stats_box.setLayout(stats_vl)
        info_layout.addWidget(stats_box, 1)

        # Legend
        legend_box = QGroupBox("Boundary legend")
        legend_vl = QVBoxLayout()
        seen = set()
        for name, rgb in _SEG_COLORS.items():
            # Deduplicate left/right having same colour
            label = name.replace('right_', '').replace('left_', '')
            if label in seen:
                continue
            seen.add(label)
            lbl = QLabel(f"● {label.replace('_', ' ').title()}")
            lbl.setStyleSheet(
                f"color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
                f"font-weight:600; font-size:9.5pt;")
            legend_vl.addWidget(lbl)
        lbl_wall = QLabel("▧ Needle wall (solid)")
        lbl_wall.setStyleSheet(
            "color: rgb(200,200,200); font-weight:600; font-size:9.5pt;")
        legend_vl.addWidget(lbl_wall)
        legend_box.setLayout(legend_vl)
        info_layout.addWidget(legend_box, 1)

        root.addWidget(info_card)

        # ---- draw ----
        self._draw_domain()

    # ------------------------------------------------------------------
    def _draw_needle_wall(self, pw, ri_mm, ro_mm, neck_h_mm, side):
        """Draw needle wall outline + hatching on one side."""
        s = side   # +1 right, -1 left
        pw.plot([s * ro_mm, s * ro_mm], [0, neck_h_mm],
                pen=pg.mkPen(color=(200, 200, 200, 220), width=2))
        pw.plot([s * ri_mm, s * ro_mm], [0, 0],
                pen=pg.mkPen(color=(200, 200, 200, 220), width=2))
        pw.plot([s * ri_mm, s * ro_mm], [neck_h_mm, neck_h_mm],
                pen=pg.mkPen(color=(200, 200, 200, 220), width=2))
        hx, hy = _hatch_rect(min(ri_mm, ro_mm), max(ri_mm, ro_mm),
                              0, neck_h_mm, n_lines=14)
        pw.plot(s * hx, hy,
                pen=pg.mkPen(color=(180, 180, 180, 90), width=1))

    # ------------------------------------------------------------------
    def _draw_domain(self):
        pw = self.plot_widget
        poly_mm = self.polygon * 1e3

        ri_mm = self.metadata['r_inner_m'] * 1e3
        ro_mm = self.metadata['r_outer_m'] * 1e3
        neck_h_mm = self.metadata['neck_height_m'] * 1e3

        # Needle wall hatching (both sides)
        self._draw_needle_wall(pw, ri_mm, ro_mm, neck_h_mm, side=+1)
        self._draw_needle_wall(pw, ri_mm, ro_mm, neck_h_mm, side=-1)

        # Full domain outline (bold cyan)
        pw.plot(poly_mm[:, 0], poly_mm[:, 1],
                pen=pg.mkPen(color=(100, 200, 255, 200), width=2))

        # Colour-coded boundary segments
        segments = self.metadata.get('segment_indices', {})
        for seg_name, (i0, i1) in segments.items():
            if i1 >= len(poly_mm):
                i1 = len(poly_mm) - 1
            seg_pts = poly_mm[i0:i1 + 1]
            if len(seg_pts) < 2:
                continue
            rgb = _SEG_COLORS.get(seg_name, (200, 200, 200))
            pw.plot(seg_pts[:, 0], seg_pts[:, 1],
                    pen=pg.mkPen(color=(*rgb, 255), width=3))

        # Symmetry axis (visual reference only — not a boundary)
        z_range = [poly_mm[:, 1].min(), poly_mm[:, 1].max()]
        pw.plot([0, 0], z_range,
                pen=pg.mkPen(color=(90, 255, 140, 50), width=1,
                             style=Qt.PenStyle.DashDotLine))
