"""
gui.solver_window
==================

Visualization window for the forward diffusion simulation results.
Shows:
    · Concentration contour snapshots at selected timesteps
    · Cs(t) interface concentration plot
    · γ(t) surface tension plot
    · Simulation diagnostics
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                             QGroupBox, QTabWidget, QPlainTextEdit)

# Use matplotlib for tricontourf (pyqtgraph doesn't support it)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.styles import DARK_STYLESHEET


class _MplCanvas(FigureCanvasQTAgg):
    """Thin wrapper around a Matplotlib figure embedded in Qt."""

    def __init__(self, width=6, height=4.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='#0f1018')
        super().__init__(self.fig)


class SolverWindow(QWidget):
    """Window showing diffusion simulation results."""

    def __init__(self, sim_results, metadata, parent=None):
        """
        Parameters
        ----------
        sim_results : dict   from ``solve_diffusion``
        metadata    : dict   from ``build_domain_polygon``
        """
        super().__init__(parent)
        self.setWindowTitle("Diffusion Simulation — Pendant Drop")
        self.resize(1100, 850)
        self.setStyleSheet(DARK_STYLESHEET)

        self.sim = sim_results
        self.metadata = metadata

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #2a2d3e; }
            QTabBar::tab { background:#1a1d2e; color:#c3c8dc;
                           padding:8px 18px; margin-right:2px;
                           border-top-left-radius:6px;
                           border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#252840; color:#fff;
                                    font-weight:600; }
        """)

        # Tab 1: Concentration contours
        tabs.addTab(self._build_contour_tab(), "Concentration Contours")
        # Tab 2: Time series
        tabs.addTab(self._build_timeseries_tab(), "Time Series")
        # Tab 3: Diagnostics
        tabs.addTab(self._build_diagnostics_tab(), "Diagnostics")

        root.addWidget(tabs)

    # ------------------------------------------------------------------
    # Tab 1: Contour snapshots
    # ------------------------------------------------------------------
    def _build_contour_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        snapshots = self.sim['snapshots']
        pts = self.sim['points'] * 1e3           # → mm
        tris = self.sim['triangles']

        n_snaps = len(snapshots)
        if n_snaps == 0:
            layout.addWidget(QLabel("No snapshots recorded."))
            return widget

        # Determine grid layout
        cols = min(n_snaps, 2)
        rows = (n_snaps + cols - 1) // cols

        canvas = _MplCanvas(width=5 * cols, height=4.5 * rows, dpi=100)
        fig = canvas.fig

        sorted_steps = sorted(snapshots.keys())

        for idx, step in enumerate(sorted_steps):
            C = snapshots[step]
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor('#0f1018')

            # Triangular contour fill
            tcf = ax.tricontourf(
                pts[:, 0], pts[:, 1], tris, C,
                levels=20,
                cmap='inferno')
            ax.tricontour(
                pts[:, 0], pts[:, 1], tris, C,
                levels=10, colors='white', linewidths=0.3, alpha=0.3)

            cbar = fig.colorbar(tcf, ax=ax, shrink=0.85, pad=0.02)
            cbar.ax.tick_params(colors='#c3c8dc', labelsize=7)
            cbar.set_label('C', color='#c3c8dc', fontsize=8)

            tau = step * self.sim.get('dtau', 0.01)
            ax.set_title(f"Step {step}  (τ = {tau:.3f})",
                         color='#c3c8dc', fontsize=10)
            ax.set_xlabel('r (mm)', color='#8e95ae', fontsize=8)
            ax.set_ylabel('z (mm)', color='#8e95ae', fontsize=8)
            ax.tick_params(colors='#8e95ae', labelsize=7)
            ax.set_aspect('equal')
            ax.invert_yaxis()

            # Needle OD outline
            ri = self.metadata['r_inner_m'] * 1e3
            ro = self.metadata['r_outer_m'] * 1e3
            nh = self.metadata['neck_height_m'] * 1e3
            for s in (+1, -1):
                ax.plot([s*ro, s*ro], [0, nh], 'w-', lw=0.8, alpha=0.5)
                ax.plot([s*ri, s*ro], [nh, nh], 'w-', lw=0.8, alpha=0.5)

        fig.tight_layout(pad=1.5)
        layout.addWidget(canvas)
        return widget

    # ------------------------------------------------------------------
    # Tab 2: Cs(t) and γ(t) time series
    # ------------------------------------------------------------------
    def _build_timeseries_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        canvas = _MplCanvas(width=10, height=7, dpi=100)
        fig = canvas.fig

        time = self.sim['time_history']
        Cs = self.sim['Cs_history']
        gamma = self.sim['gamma_history']

        # Cs(t)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_facecolor('#0f1018')
        ax1.plot(time, Cs, color='#64d8ff', linewidth=2)
        ax1.set_xlabel('τ (dimensionless time)', color='#8e95ae')
        ax1.set_ylabel('Cs (interface conc.)', color='#8e95ae')
        ax1.set_title('Interface Concentration Evolution',
                       color='#c3c8dc', fontsize=11)
        ax1.tick_params(colors='#8e95ae')
        ax1.grid(True, alpha=0.15)
        ax1.set_xlim(left=0)
        ax1.set_ylim(-0.02, 1.02)

        # γ(t)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_facecolor('#0f1018')
        ax2.plot(time, gamma, color='#ff8a5c', linewidth=2)
        ax2.set_xlabel('τ (dimensionless time)', color='#8e95ae')
        ax2.set_ylabel('γ (mN/m)', color='#8e95ae')
        ax2.set_title('Surface Tension Evolution',
                       color='#c3c8dc', fontsize=11)
        ax2.tick_params(colors='#8e95ae')
        ax2.grid(True, alpha=0.15)
        ax2.set_xlim(left=0)

        fig.tight_layout(pad=2.0)
        layout.addWidget(canvas)
        return widget

    # ------------------------------------------------------------------
    # Tab 3: Diagnostics
    # ------------------------------------------------------------------
    def _build_diagnostics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Parameters card
        params_box = QGroupBox("Simulation parameters")
        pv = QVBoxLayout()
        kD = self.sim.get('kD', 0)
        C_final = self.sim['C_final']
        params_text = (
            f"Biot number kD:    {kD:.4f}\n"
            f"Time steps:        {len(self.sim['time_history'])}\n"
            f"─────────────────────────\n"
            f"Final Cs:          {self.sim['Cs_history'][-1]:.6f}\n"
            f"Final C min:       {C_final.min():.6e}\n"
            f"Final C max:       {C_final.max():.6e}\n"
            f"Has NaN:           {bool(np.any(np.isnan(C_final)))}\n"
            f"Has negative:      {bool(np.any(C_final < 0))}\n"
            f"─────────────────────────\n"
            f"γ range:           [{min(self.sim['gamma_history']):.2f}, "
            f"{max(self.sim['gamma_history']):.2f}] mN/m"
        )
        lbl = QLabel(params_text)
        lbl.setStyleSheet(
            "color:#c3c8dc; font-family:'Consolas','Menlo',monospace; "
            "font-size:10pt;")
        pv.addWidget(lbl)
        params_box.setLayout(pv)
        layout.addWidget(params_box)

        # Step log. Long runs produce one line per step (100k+ for a
        # multi-minute sim), so show only the head and tail — dumping every
        # line into a widget freezes the UI. QPlainTextEdit also handles large
        # text far better than QLabel.
        log_box = QGroupBox("Step log")
        lv = QVBoxLayout()
        diagnostics = self.sim.get('diagnostics', [])
        max_lines = 500
        if len(diagnostics) > max_lines:
            head, tail = diagnostics[:20], diagnostics[-(max_lines - 20):]
            omitted = len(diagnostics) - len(head) - len(tail)
            shown = head + [f"… {omitted} steps omitted …"] + tail
        else:
            shown = diagnostics
        log_view = QPlainTextEdit("\n".join(shown))
        log_view.setReadOnly(True)
        log_view.setStyleSheet(
            "color:#8e95ae; font-family:'Consolas','Menlo',monospace; "
            "font-size:8.5pt; padding:6px;")
        lv.addWidget(log_view)
        log_box.setLayout(lv)
        layout.addWidget(log_box, stretch=2)

        return widget
