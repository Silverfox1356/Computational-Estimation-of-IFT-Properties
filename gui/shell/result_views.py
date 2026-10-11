"""
gui.shell.result_views
======================

Graphics tabs for stored analysis results (core.analyses): each view draws
from the ``data`` a run saved in ``analyses/<name>.json``, so results show
again when a session is reopened.

    SwellingView   V / V0 against time, threshold line and band
    BagalkotView   measured and model volume; fit misfit F against D
    YangView       measured and fitted IFT; E against D for each kD (Yang's
                   Fig 7a, curves can be hidden); the lowest E for each kD
    ModelView      the model Yang and the simulation solve on: domain outline by
                   boundary, mesh, and the FEM system's sanity checks
    SimulationView the forward simulation: concentration snapshots, Cs(t),
                   gamma(t) against the measured IFT, and the step log
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPlainTextEdit, QPushButton, QScrollArea, QSlider, QVBoxLayout,
                             QWidget)

from gui.shell.panels import fmt
from gui.shell.theme import PLOT_FAIL, PLOT_LINE, PLOT_PASS

NEAR = '#ff8c1a'
BEST = '#2f9e44'


def make_plot(title, xlabel, ylabel, logx=False, logy=False):
    p = pg.PlotWidget(background='w')
    p.setTitle(title, color='#222', size='10pt')
    for ax in ('left', 'bottom'):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen('#555'))
        a.setTextPen(pg.mkPen('#222'))
    p.setLabel('bottom', xlabel)
    p.setLabel('left', ylabel)
    p.showGrid(x=True, y=True, alpha=0.25)
    p.setLogMode(x=logx, y=logy)
    p.addLegend(offset=(-10, 10), labelTextColor='#222')
    p.setMinimumSize(220, 160)
    return p


def _arr(v):
    return np.asarray([np.nan if x is None else x for x in (v or [])], float)


class _View(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.note = QLabel('')
        self.note.setObjectName('Note')
        self.note.setWordWrap(True)


class SwellingView(_View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = make_plot('Drop volume', 'Time (s)', 'V / V0')
        self.grid.addWidget(self.plot, 0, 0)
        self.grid.addWidget(self.note, 1, 0)

    def set_data(self, data):
        p = self.plot
        p.clear()
        r, c = data['result'], data['curve']
        t, V = _arr(c['t']), _arr(c['V'])
        if not len(V):
            return
        V0 = r.get('V0') or V[0]
        p.plot(t, V / V0, pen=pg.mkPen(PLOT_LINE, width=1), symbol='o', symbolSize=6,
               symbolBrush=PLOT_LINE, symbolPen=None, name='measured')
        thr = 1.0 + data['settings']['threshold']
        p.addItem(pg.InfiniteLine(pos=thr, angle=0, label=f"threshold {100 * (thr - 1):g} %",
                                  labelOpts=dict(color=PLOT_FAIL, position=0.15, anchors=[(0, 1), (0, 1)]),
                                  pen=pg.mkPen(PLOT_FAIL, width=1.5, style=pg.QtCore.Qt.PenStyle.DashLine)))
        self.note.setText(r.get('message', ''))


class BagalkotView(_View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit = make_plot('Drop volume: measured and model', 'Time (s)', 'Volume (uL)')
        self.sweep = make_plot('Fit misfit against D', 'D (m2/s)', 'F', logx=True)
        self.grid.addWidget(self.fit, 0, 0)
        self.grid.addWidget(self.sweep, 0, 1)
        self.grid.addWidget(self.note, 1, 0, 1, 2)

    def set_data(self, data):
        r, c, s = data['result'], data['curve'], data['sweep']
        self.fit.clear()
        self.fit.plot(_arr(c['t']), _arr(c['V_exp']), pen=None, symbol='o', symbolSize=6,
                      symbolBrush=PLOT_LINE, symbolPen=None, name='measured')
        self.fit.plot(_arr(c['t']), _arr(c['V_model']), pen=pg.mkPen(NEAR, width=2),
                      name=f"model, D = {fmt(r['D'], 3)} m2/s")
        D, F = _arr(s['D']), _arr(s['F'])
        near = F <= r['F_min'] * (1 + r['near_tol']) + 1e-15
        self.sweep.clear()
        self.sweep.plot(D, F, pen=pg.mkPen('#7a8899', width=1), symbol='o', symbolSize=5,
                        symbolBrush='#7a8899', symbolPen=None, name='tried')
        self.sweep.plot(D[near], F[near], pen=None, symbol='o', symbolSize=8,
                        symbolBrush=NEAR, symbolPen=None, name='equally good')
        self.sweep.plot([r['D']], [r['F_min']], pen=None, symbol='star', symbolSize=14,
                        symbolBrush=BEST, symbolPen=None, name='best')
        lo, hi = r['D_near']
        self.note.setText(f"D = {fmt(r['D'], 4)} m2/s ({fmt(r['D_per_min'], 4)} m2/min); "
                          f"fits within {100 * r['near_tol']:g} % of the best span "
                          f"{fmt(lo, 3)} to {fmt(hi, 3)} m2/s."
                          + (" The best D is at the edge of the search range: widen it."
                             if r['at_edge'] else ''))


# One colour and marker per kD curve (Yang's Fig 7a uses markers; colours help).
CURVE_COLOURS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                 '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
CURVE_SYMBOLS = ('o', 's', 't', 'd', 't1', 'p', 'h', 'star', '+', 'x')


class YangView(_View):
    """Yang's fit: measured and fitted IFT; the error E against D for each kD
    (his Fig 7a), each curve can be shown or hidden; the lowest E for each kD."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit = make_plot('Interfacial tension: measured and fitted', 'Time (s)', 'IFT (mN/m)')
        self.curves = make_plot('Fit error E against D, for each kD', 'D (m2/s)', 'E (%)',
                                logx=True)
        self.err = make_plot('Lowest E for each kD', 'kD', 'E (%)', logx=True)
        # The colour list beside the plot is the legend; plain powers of ten on D.
        self.curves.plotItem.legend.setVisible(False)
        self.curves.getAxis('bottom').enableAutoSIPrefix(False)
        self.list = QListWidget()
        self.list.setToolTip('Tick the kD curves to show')
        self.list.setMaximumWidth(150)
        self.list.setMinimumHeight(90)
        self.list.itemChanged.connect(self._toggled)
        show_all, hide_all = QPushButton('Show all'), QPushButton('Hide all')
        show_all.clicked.connect(lambda: self._set_all(True))
        hide_all.clicked.connect(lambda: self._set_all(False))
        side = QVBoxLayout()
        side.addWidget(QLabel('kD curves'))
        side.addWidget(self.list, 1)
        side.addWidget(show_all)
        side.addWidget(hide_all)
        self.grid.addWidget(self.fit, 0, 0)
        self.grid.addWidget(self.err, 0, 1, 1, 2)
        self.grid.addWidget(self.curves, 1, 0, 1, 2)
        self.grid.addLayout(side, 1, 2)
        self.grid.addWidget(self.note, 2, 0, 1, 3)
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(1, 2)
        self.grid.setRowStretch(0, 2)
        self.grid.setRowStretch(1, 3)
        self.items = {}                        # kD -> plot items of its curve

    def set_data(self, data):
        r, c, sw = data['result'], data['curve'], data['sweep']
        t = _arr(c["t"])
        self.fit.clear()
        self.fit.plot(t, _arr(c['gamma_exp']), pen=None, symbol='o', symbolSize=6,
                      symbolBrush=PLOT_PASS, symbolPen=None, name='measured')
        self.fit.plot(t, _arr(c['gamma_fit']), pen=pg.mkPen(NEAR, width=2),
                      name=f"fit, E = {fmt(r['E_percent'], 3)} %")
        kD, E = _arr(sw['kD']), _arr(sw['E'])
        near = E <= r['E_percent'] * (1 + r['near_tol'])
        p = self.err
        p.clear()
        p.plot(kD, E, pen=pg.mkPen('#7a8899', width=1), symbol='o', symbolSize=5,
               symbolBrush='#7a8899', symbolPen=None, name='tried')
        p.plot(kD[near], E[near], pen=None, symbol='o', symbolSize=8, symbolBrush=NEAR,
               symbolPen=None, name='equally good')
        p.plot([r['kD']], [r['E_percent']], pen=None, symbol='star', symbolSize=14,
               symbolBrush=BEST, symbolPen=None, name='best')
        ax = p.getAxis('bottom')
        if 1 < len(kD) <= 12:                   # a kD grid: label its values only
            ax.setTicks([[(float(np.log10(v)), f"{v:g}") for v in kD], []])
        else:
            ax.setTicks(None)
        self._draw_curves(data.get('curves') or [], r)
        verdict = ("D is well determined by this curve." if r['D_well_determined'] else
                   "D is NOT well determined: equally good fits span a wide range of D.")
        self.note.setText(f"D = {fmt(r['D'], 4)} m2/s, k = {fmt(r['k'], 4)} m/s, kD = "
                          f"{fmt(r['kD'], 4)}, E = {fmt(r['E_percent'], 3)} %. {verdict} "
                          f"{r.get('message', '')}")

    def _draw_curves(self, curves, r):
        p = self.curves
        p.clear()
        self.items = {}
        self.list.blockSignals(True)
        self.list.clear()
        lows = []
        for i, cv in enumerate(curves):
            colour = CURVE_COLOURS[i % len(CURVE_COLOURS)]
            sym = CURVE_SYMBOLS[i % len(CURVE_SYMBOLS)]
            D, E = _arr(cv['D']), _arr(cv['E'])
            best = abs(cv['kD'] - r['kD']) <= 1e-9 * max(1.0, r['kD'])
            line = p.plot(D, E, pen=pg.mkPen(colour, width=2.5 if best else 1.5),
                          symbol=sym, symbolSize=5, symbolBrush=colour, symbolPen=None)
            low = p.plot([cv['D_min']], [cv['E_min']], pen=None, symbol='star',
                         symbolSize=13 if best else 9, symbolBrush=BEST if best else colour,
                         symbolPen=pg.mkPen('#222', width=0.8))
            self.items[cv['kD']] = (line, low)
            lows.append(cv['E_min'])
            it = QListWidgetItem(f"kD = {fmt(cv['kD'], 4)}" + (' (best)' if best else ''))
            it.setData(Qt.ItemDataRole.UserRole, cv['kD'])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setForeground(pg.mkColor(colour))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if lows:
            # Near the minima, as in Yang's figure; the user can zoom out.
            top = min(max(lows), 4.0 * min(lows) + 1.0)
            p.setYRange(0, max(top, 1.5 * min(lows)), padding=0.05)

    def _toggled(self, item):
        parts = self.items.get(item.data(Qt.ItemDataRole.UserRole))
        if parts:
            on = item.checkState() == Qt.CheckState.Checked
            for part in parts:
                part.setVisible(on)

    def _set_all(self, on):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if on
                                            else Qt.CheckState.Unchecked)


# Boundary segments of the domain (core.domain_builder), as the Domain window.
SEGMENTS = (('free_surface', 'Drop interface (solute enters here)', '#1c7ed6'),
            ('tip_step', 'Needle tip', '#e8590c'),
            ('inner_wall', 'Needle inner wall', '#f59f00'),
            ('top_wall', 'Top of the domain', '#868e96'))


class ModelView(_View):
    """The model Yang and the simulation solve on: the domain outline coloured
    by boundary, the mesh, and the FEM system's checks (the Domain, Mesh and
    FEM windows of the Pendant window, in one tab)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = make_plot('Model: domain and mesh (drop cross-section)', 'r (mm)', 'z (mm)')
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)                 # z grows downward from the needle
        self.info = QLabel('')
        self.info.setObjectName('ModelInfoText')
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info.setMinimumWidth(200)
        scroll = QScrollArea()
        scroll.setObjectName('ModelInfo')
        scroll.setWidget(self.info)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.grid.addWidget(self.plot, 0, 0)
        self.grid.addWidget(scroll, 0, 1)
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(1, 2)
        self.grid.addWidget(self.note, 1, 0, 1, 2)

    def set_data(self, data):
        m = data['model']
        P = np.asarray(m['points'], float) * 1e3
        T = np.asarray(m['triangles'], int)
        p = self.plot
        p.clear()
        if len(T):
            e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
            e = np.unique(np.sort(e, axis=1), axis=0)
            xy = P[e.reshape(-1)]
            p.plot(xy[:, 0], xy[:, 1], connect='pairs', pen=pg.mkPen('#9fb3c8', width=0.5))
        poly = np.asarray(m.get('polygon') or [], float) * 1e3
        dom = m.get('domain') or {}
        seg = dom.get('segment_indices') or {}
        for key, label, colour in SEGMENTS:
            for name, (a, b) in seg.items():
                if key in name and len(poly):
                    part = poly[a:b + 1]
                    p.plot(part[:, 0], part[:, 1], pen=pg.mkPen(colour, width=2.5))
        if dom:
            ri, ro, nh = (dom[k] * 1e3 for k in ('r_inner_m', 'r_outer_m', 'neck_height_m'))
            for side in (1, -1):              # needle wall (solid, not in the domain)
                x = side * np.array([ri, ro, ro, ri, ri])
                p.plot(x, [0, 0, nh, nh, 0], pen=pg.mkPen('#495057', width=1.2))
        self.info.setText(self._info_html(m))
        self.note.setText(f"Drop outline from frame {m['frame_index']} "
                          f"(t = {fmt(m['frame_time_s'], 4)} s). The model is axisymmetric; "
                          f"both halves are shown.")

    @staticmethod
    def _info_html(m):
        dom, fem = m.get('domain'), m.get('fem')
        rows = ['<b>Boundaries</b>']
        rows += [f"<span style='color:{c}'><b>&#8212;&#8212;</b></span> {label}"
                 for _, label, c in SEGMENTS]
        rows += ["<span style='color:#495057'><b>&#9633;</b></span> Needle wall (solid, "
                 "not part of the model)",
                 "<span style='color:#9fb3c8'><b>&#9651;</b></span> Mesh triangles", '']
        if dom:
            rows.append('<b>Domain</b>')
            rows += [f"Outline points: {dom['n_points']}",
                     f"Max radius: {fmt(dom['r_max_m'] * 1e3, 4)} mm",
                     f"Total height: {fmt(dom['height_m'] * 1e3, 4)} mm",
                     f"Neck height: {fmt(dom['neck_height_m'] * 1e3, 4)} mm",
                     f"Drop height: {fmt(dom['drop_height_m'] * 1e3, 4)} mm",
                     f"Needle ID / OD: {fmt(2 * dom['r_inner_m'] * 1e3, 4)} / "
                     f"{fmt(2 * dom['r_outer_m'] * 1e3, 4)} mm"]
        rows.append('<br><b>Mesh</b>')
        rows += [f"{m['n_nodes']} nodes, {m['n_elements']} triangles",
                 f"Smallest angle: {fmt(m['min_angle'], 3)} degrees"]
        if m.get('area_ratio') == m.get('area_ratio') and m.get('area_ratio') is not None:
            rows.append(f"Largest / smallest triangle: {fmt(m['area_ratio'], 3)}")
        if fem:
            ok = fem['all_pass']
            rows.append(f"<br><b>FEM system</b>: {fem['size']} x {fem['size']}")
            rows += [f"Non-zeros: H {fem['H_nnz']}, K {fem['K_nnz']}, Kb {fem['Kb_nnz']}",
                     f"Interface nodes {fem['n_interface_nodes']}, wall nodes "
                     f"{fem['n_wall_nodes']}, interior {fem['n_interior_nodes']}",
                     '<br><b>Sanity checks</b>: '
                     + (f"<span style='color:{PLOT_PASS}'>all pass</span>" if ok else
                        f"<span style='color:{PLOT_FAIL}'>some fail</span>")]
            for name, passed, detail in fem['checks']:
                mark = (f"<span style='color:{PLOT_PASS}'>pass</span>" if passed
                        else f"<span style='color:{PLOT_FAIL}'><b>FAIL</b></span>")
                rows.append(f"{mark} {name} <span style='color:#6b7785'>({detail})</span>")
        return '<br>'.join(rows)


class SimulationView(_View):
    """The forward simulation (the Pendant window's Simulation window):
    concentration in the drop at the stored times (slider), the interface
    concentration Cs(t), gamma(t) against the measured IFT, and diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.canvas = None                      # matplotlib, made on first use
        self.field_box = QWidget()
        self.field_lay = QVBoxLayout(self.field_box)
        self.field_lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('Time of the concentration snapshot')
        self.slider.valueChanged.connect(self._draw_field)
        self.slider_label = QLabel('')
        row = QHBoxLayout()
        row.addWidget(QLabel('Snapshot'))
        row.addWidget(self.slider, 1)
        row.addWidget(self.slider_label)
        self.field_lay.addLayout(row)
        self.cs = make_plot('Interface concentration', 'Time (s)', 'Cs (c / c_sat)')
        self.gamma = make_plot('Interfacial tension', 'Time (s)', 'IFT (mN/m)')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName('StepLog')
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setMaximumHeight(110)
        self.log.setToolTip('Step log of the solver (first and last steps)')
        self.grid.addWidget(self.field_box, 0, 0, 2, 1)
        self.grid.addWidget(self.cs, 0, 1)
        self.grid.addWidget(self.gamma, 1, 1)
        self.grid.addWidget(self.note, 2, 0, 1, 2)
        self.grid.addWidget(self.log, 3, 0, 1, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

    def _make_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(4, 4.5), dpi=90, facecolor='white')
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumSize(220, 220)
        self.field_lay.insertWidget(0, self.canvas, 1)

    def set_data(self, data):
        self.data = data
        if self.canvas is None:
            self._make_canvas()
        snaps = data['snapshots']
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(snaps) - 1))
        self.slider.setValue(max(0, len(snaps) - 1))
        self.slider.blockSignals(False)
        self._draw_field()
        r, st, c = data['result'], data['settings'], data['curve']
        t = _arr(c['t'])
        self.cs.clear()
        self.cs.plot(np.r_[0.0, t], np.r_[0.0, _arr(c['Cs'])], pen=pg.mkPen(PLOT_LINE, width=2),
                     name='Cs, simulated')
        self.gamma.clear()
        if c.get('gamma') is not None:
            self.gamma.plot(t, _arr(c['gamma']), pen=pg.mkPen(NEAR, width=2),
                            name='simulated (gamma(c) calibration)')
        meas = data.get('measured')
        if meas:
            self.gamma.plot(_arr(meas['t']), _arr(meas['gamma']), pen=None, symbol='o',
                            symbolSize=6, symbolBrush=PLOT_PASS, symbolPen=None,
                            name='measured (Yang fit input)')
        src = 'the Yang fit' if st.get('source') == 'yang' else 'values entered'
        steps = ('automatic steps' if st.get('stepping') == 'graded'
                 else f"steps of {fmt(st.get('dt_s'))} s")
        g = (f" Final IFT {fmt(r['gamma_final'], 4)} mN/m, range {fmt(r['gamma_range'][0], 4)} "
             f"to {fmt(r['gamma_range'][1], 4)} mN/m." if r.get('gamma_final') is not None
             else ' No gamma(c) calibration in the session, so only Cs(t) is simulated.')
        flags = []
        if r.get('has_nan'):
            flags.append('the final field has NaN values')
        if r.get('has_negative'):
            flags.append('the final field has negative values')
        self.note.setText(
            f"D = {fmt(st['D'], 4)} m2/s, k = {fmt(st['k'], 4)} m/s (from {src}), kD = "
            f"{fmt(r['kD'], 4)}; {fmt(r['simulated_s'], 5)} s in {r['n_steps']} {steps}. "
            f"Final Cs {fmt(r['Cs_final'], 4)}; C from {fmt(r['C_min'], 3)} to "
            f"{fmt(r['C_max'], 3)} over the run.{g}"
            + (f" Check: {'; '.join(flags)}." if flags else ''))
        self.log.setPlainText('\n'.join(data.get('diagnostics') or []))

    def _draw_field(self, *_):
        if self.data is None or self.canvas is None:
            return
        snaps = self.data['snapshots']
        fig = self.canvas.figure
        fig.clear()
        if not snaps:
            self.slider_label.setText('no snapshots')
            self.canvas.draw_idle()
            return
        snap = snaps[min(self.slider.value(), len(snaps) - 1)]
        m = self.data['model']
        P = np.asarray(m['points'], float) * 1e3
        T = np.asarray(m['triangles'], int)
        C = _arr(snap['C'])
        ax = fig.add_subplot(111)
        tcf = ax.tricontourf(P[:, 0], P[:, 1], T, C, levels=np.linspace(0, 1, 21),
                             cmap='viridis', extend='both')
        ax.tricontour(P[:, 0], P[:, 1], T, C, levels=10, colors='white', linewidths=0.3,
                      alpha=0.4)
        cb = fig.colorbar(tcf, ax=ax, shrink=0.85, pad=0.02, ticks=np.linspace(0, 1, 6))
        cb.set_label('c / c_sat', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        dom = m.get('domain') or {}
        if dom:
            ri, ro, nh = (dom[k] * 1e3 for k in ('r_inner_m', 'r_outer_m', 'neck_height_m'))
            for side in (1, -1):
                ax.fill(side * np.array([ri, ro, ro, ri]), [0, 0, nh, nh], color='#adb5bd')
        ax.set_title(f"Concentration at t = {fmt(snap['t'], 4)} s", fontsize=10)
        ax.set_xlabel('r (mm)', fontsize=8)
        ax.set_ylabel('z (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        fig.tight_layout()
        self.canvas.draw_idle()
        self.slider_label.setText(f"{self.slider.value() + 1} of {len(snaps)}")
