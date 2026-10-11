"""
gui.shell.result_views
======================

Graphics tabs for stored analysis results (core.analyses): each view draws
from the ``data`` a run saved in ``analyses/<name>.json``, so results show
again when a session is reopened.

    SwellingView   V / V0 against time, threshold line and band
    BagalkotView   measured and model volume; fit misfit F against D
    YangView       measured and fitted IFT; fit error E and best D against kD
    MeshView       the FEM mesh Yang's model was built on (drop cross-section)
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

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


class YangView(_View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit = make_plot('Interfacial tension: measured and fitted', 'Time (s)', 'IFT (mN/m)')
        self.err = make_plot('Fit error against kD', 'kD', 'E (%)', logx=True)
        self.dkd = make_plot('Best D at each kD', 'kD', 'D (m2/s)', logx=True, logy=True)
        self.grid.addWidget(self.fit, 0, 0, 1, 2)
        self.grid.addWidget(self.err, 1, 0)
        self.grid.addWidget(self.dkd, 1, 1)
        self.grid.addWidget(self.note, 2, 0, 1, 2)

    def set_data(self, data):
        r, c, sw = data['result'], data['curve'], data['sweep']
        t = _arr(c["t"])
        self.fit.clear()
        self.fit.plot(t, _arr(c['gamma_exp']), pen=None, symbol='o', symbolSize=6,
                      symbolBrush=PLOT_PASS, symbolPen=None, name='measured (QC passed)')
        self.fit.plot(t, _arr(c['gamma_fit']), pen=pg.mkPen(NEAR, width=2),
                      name=f"fit, E = {fmt(r['E_percent'], 3)} %")
        kD, E, D = _arr(sw['kD']), _arr(sw['E']), _arr(sw['D'])
        near = E <= r['E_percent'] * (1 + r['near_tol'])
        for p, y, best in ((self.err, E, r['E_percent']), (self.dkd, D, r['D'])):
            p.clear()
            p.plot(kD, y, pen=pg.mkPen('#7a8899', width=1), symbol='o', symbolSize=5,
                   symbolBrush='#7a8899', symbolPen=None, name='tried')
            p.plot(kD[near], y[near], pen=None, symbol='o', symbolSize=8, symbolBrush=NEAR,
                   symbolPen=None, name='equally good')
            p.plot([r['kD']], [best], pen=None, symbol='star', symbolSize=14,
                   symbolBrush=BEST, symbolPen=None, name='best')
        verdict = ("D is well determined by this curve." if r['D_well_determined'] else
                   "D is NOT well determined: equally good fits span a wide range of D.")
        self.note.setText(f"D = {fmt(r['D'], 4)} m2/s, k = {fmt(r['k'], 4)} m/s, kD = "
                          f"{fmt(r['kD'], 4)}. {verdict} {r.get('message', '')}")


class MeshView(_View):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = make_plot('Model mesh (drop cross-section)', 'x (mm)', 'z (mm)')
        self.plot.setAspectLocked(True)
        self.grid.addWidget(self.plot, 0, 0)
        self.grid.addWidget(self.note, 1, 0)

    def set_data(self, data):
        m = data['model']
        P = np.asarray(m['points'], float) * 1e3
        T = np.asarray(m['triangles'], int)
        self.plot.clear()
        if len(T):
            e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
            e = np.unique(np.sort(e, axis=1), axis=0)
            xy = P[e.reshape(-1)]
            self.plot.plot(xy[:, 0], xy[:, 1], connect='pairs', pen=pg.mkPen('#4a6a95', width=0.6))
        poly = np.asarray(m.get('polygon') or [], float)
        if len(poly):
            poly = np.vstack([poly, poly[:1]]) * 1e3
            self.plot.plot(poly[:, 0], poly[:, 1], pen=pg.mkPen('#1f2328', width=1.5))
        self.note.setText(f"{m['n_nodes']} nodes, {m['n_elements']} elements, smallest angle "
                          f"{fmt(m['min_angle'], 3)} degrees; drop outline from frame "
                          f"{m['frame_index']} (t = {fmt(m['frame_time_s'], 4)} s).")
