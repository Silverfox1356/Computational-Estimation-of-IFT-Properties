"""
gui.fem_window
===============

Visualization window for FEM assembly results.  Shows:
    · Interface vs wall boundary edges on the mesh
    · Sanity-check results (symmetry, sparsity, diagonal)
    · Matrix statistics (size, non-zeros, sparsity %)
    · Node classification counts
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QFrame)
from PyQt6.QtCore import Qt
from gui.styles import DARK_STYLESHEET
from gui.domain_window import _hatch_rect


class FEMWindow(QWidget):
    """Window showing FEM assembly results and boundary classification."""

    @staticmethod
    def _add_check_widget(layout, name, ok, detail):
        """Add a single check row (icon + name + detail) to *layout*."""
        icon = "✓" if ok else "✗"
        color = "#4fd07b" if ok else "#ff6a4d"
        lbl = QLabel(f"{icon}  {name}")
        lbl.setStyleSheet(f"color:{color}; font-weight:600; "
                          f"font-size:10pt;")
        layout.addWidget(lbl)
        lbl_d = QLabel(f"    {detail}")
        lbl_d.setStyleSheet("color:#8e95ae; font-size:8.5pt;")
        layout.addWidget(lbl_d)


    def __init__(self, fem_results, polygon, metadata, parent=None):
        """
        Parameters
        ----------
        fem_results : dict from ``assemble_fem_system``
        polygon : (N, 2) ndarray
        metadata : dict from ``build_domain_polygon``
        """
        super().__init__(parent)
        self.setWindowTitle("FEM Assembly — Pendant Drop")
        self.resize(1000, 820)
        self.setStyleSheet(DARK_STYLESHEET)

        self.fem = fem_results
        self.polygon = polygon
        self.metadata = metadata

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- plot ----
        pg.setConfigOption('background', '#0f1018')
        pg.setConfigOption('foreground', '#c3c8dc')

        self.plot_widget = pg.PlotWidget(
            title="Boundary Classification: Interface vs Wall")
        self.plot_widget.setLabel('bottom', 'r  (mm)')
        self.plot_widget.setLabel('left',   'z  (mm)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.12)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.invertY(True)
        root.addWidget(self.plot_widget, stretch=4)

        # ---- cards ----
        cards = QHBoxLayout()
        cards.setSpacing(10)

        # Sanity checks
        s = fem_results['sanity']
        sanity_box = QGroupBox("Sanity checks")
        sv = QVBoxLayout()

        # --- H symmetry (strict) ---
        self._add_check_widget(sv, "H symmetric", s['H_symmetric'],
                   f"error = {s['H_symmetry_error']:.2e}")

        # --- K symmetry, relative to ‖K‖ (3-tier: ok / warning / fail) ---
        k_status = s.get('K_symmetry_status', 'fail')
        if k_status == 'ok':
            k_color, k_icon = "#4fd07b", "✓"
        elif k_status == 'warning':
            k_color, k_icon = "#ffb84d", "⚠"   # amber
        else:
            k_color, k_icon = "#ff6a4d", "✗"
        lbl_k = QLabel(f"{k_icon}  K symmetric")
        lbl_k.setStyleSheet(f"color:{k_color}; font-weight:600; "
                            f"font-size:10pt;")
        sv.addWidget(lbl_k)
        lbl_kd = QLabel(f"    rel. error = {s['K_symmetry_error']:.2e}  "
                        f"[{k_status}]")
        lbl_kd.setStyleSheet("color:#8e95ae; font-size:8.5pt;")
        sv.addWidget(lbl_kd)

        # --- remaining checks ---
        checks = [
            ("H diag ≥ 0", s['H_diag_all_positive'],
             f"min = {s['H_diag_min']:.2e}"),
            ("K diag > 0", s.get('K_diag_positive', False),
             f"min = {s.get('K_diag_min', 0):.2e}"),
            ("K conserves mass", s.get('K_conservative', False),
             f"max |column sum| = {s.get('K_colsum_rel', float('nan')):.1e}"),
            ("Sparsity < 1%", s['sparsity_ok'],
             f"H={s['H_sparsity_pct']:.3f}%  K={s['K_sparsity_pct']:.3f}%"),
            ("Boundary terms", s['boundary_terms_ok'],
             f"Kb nnz={s['Kb_nnz']}  F≠0={s['F_nonzero']}"),
            ("F only on interface", s.get('F_only_on_interface', False),
             f"max={s.get('F_max',0):.2e}  "
             f"min>0={s.get('F_min_nonzero',0):.2e}"),
        ]

        for name, ok, detail in checks:
            self._add_check_widget(sv, name, ok, detail)

        overall_pass = s['all_pass']
        lbl_all = QLabel("ALL CHECKS PASS" if overall_pass
                         else "⚠ SOME CHECKS FAILED")
        lbl_all.setStyleSheet(
            f"color:{'#4fd07b' if overall_pass else '#ff6a4d'}; "
            f"font-weight:700; font-size:11pt; margin-top:6px;")
        sv.addWidget(lbl_all)
        sanity_box.setLayout(sv)
        cards.addWidget(sanity_box, 1)

        # Matrix stats
        stats_box = QGroupBox("Matrix statistics")
        stv = QVBoxLayout()
        N = fem_results['H'].shape[0]
        stats_text = (
            f"System size:     {N} × {N}\n"
            f"─────────────────────────\n"
            f"H  non-zeros:    {fem_results['H'].nnz:,}\n"
            f"K  non-zeros:    {fem_results['K'].nnz:,}\n"
            f"Kb non-zeros:    {fem_results['Kb'].nnz:,}\n"
            f"─────────────────────────\n"
            f"Interface edges: {s['n_interface_nodes']} nodes\n"
            f"Wall edges:      {s['n_wall_nodes']} nodes\n"
            f"Interior nodes:  {s['n_interior_nodes']}\n"
        )
        lbl_stats = QLabel(stats_text)
        lbl_stats.setStyleSheet(
            "color:#c3c8dc; font-family:'Consolas','Menlo',monospace; "
            "font-size:9.5pt;")
        stv.addWidget(lbl_stats)
        stats_box.setLayout(stv)
        cards.addWidget(stats_box, 1)

        root.addLayout(cards)

        self._draw()

    # ------------------------------------------------------------------
    def _edges_to_lines(self, pts_mm, edges):
        """Convert edge list to polyline with NaN separators."""
        xs, ys = [], []
        for i, j in edges:
            xs.extend([pts_mm[i, 0], pts_mm[j, 0], np.nan])
            ys.extend([pts_mm[i, 1], pts_mm[j, 1], np.nan])
        return np.array(xs), np.array(ys)

    # ------------------------------------------------------------------
    def _draw(self):
        pw = self.plot_widget
        pts = self.fem['points']
        pts_mm = pts * 1e3
        poly_mm = self.polygon * 1e3

        ri_mm = self.metadata['r_inner_m'] * 1e3
        ro_mm = self.metadata['r_outer_m'] * 1e3
        nh_mm = self.metadata['neck_height_m'] * 1e3

        # Needle wall hatching
        for side in (+1, -1):
            pw.plot([side * ro_mm, side * ro_mm], [0, nh_mm],
                    pen=pg.mkPen(color=(200, 200, 200, 120), width=1))
            pw.plot([side * ri_mm, side * ro_mm], [0, 0],
                    pen=pg.mkPen(color=(200, 200, 200, 120), width=1))
            pw.plot([side * ri_mm, side * ro_mm], [nh_mm, nh_mm],
                    pen=pg.mkPen(color=(200, 200, 200, 120), width=1))
            hx, hz = _hatch_rect(ri_mm, ro_mm, 0, nh_mm, n_lines=10)
            pw.plot(side * hx, hz,
                    pen=pg.mkPen(color=(150, 150, 150, 40), width=0.6))

        # Domain outline (faded)
        pw.plot(poly_mm[:, 0], poly_mm[:, 1],
                pen=pg.mkPen(color=(100, 200, 255, 60), width=1))

        # Interface edges (cyan, thick)
        if self.fem['interface_edges']:
            ix, iy = self._edges_to_lines(pts_mm, self.fem['interface_edges'])
            pw.plot(ix, iy,
                    pen=pg.mkPen(color=(80, 220, 255, 255), width=3),
                    connect='finite', name='Interface')

        # Wall edges (orange)
        if self.fem['wall_edges']:
            wx, wy = self._edges_to_lines(pts_mm, self.fem['wall_edges'])
            pw.plot(wx, wy,
                    pen=pg.mkPen(color=(255, 160, 60, 255), width=3),
                    connect='finite', name='Wall')

        # Legend labels
        pw.addLegend(offset=(10, 10))
        pw.plot([], [], pen=pg.mkPen(color=(80, 220, 255), width=3),
                name='Interface (Robin BC)')
        pw.plot([], [], pen=pg.mkPen(color=(255, 160, 60), width=3),
                name='Wall (no-flux)')
