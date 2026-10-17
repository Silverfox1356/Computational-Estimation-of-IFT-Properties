"""
gui.shell.result_views
======================

Graphics tabs for stored analysis results (core.analyses): each view draws
from the ``data`` a run saved in ``analyses/<name>.json``, so results show
again when a session is reopened.

    SwellingView   V / V0 against time, threshold line and band
    BagalkotView   measured and model volume; fit misfit F against D
    YangView       measured and fitted IFT; E against D for each kD (Yang's
                   Fig 7a; a picker chooses which kD curves show); the lowest
                   E for each kD
    ModelView      the model Yang and the simulation solve on: domain outline by
                   boundary, mesh, and the FEM system's sanity checks
    SimulationView the forward simulation: concentration snapshots, Cs(t),
                   gamma(t) against the measured IFT, and the step log

Each view scrolls when the Graphics area is smaller than its plots, so plots
and text never overlap.  Warnings about a result are on its Settings page,
not here; the line under the plots gives the numbers only.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QLinearGradient, QPainter
from PyQt6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                             QScrollArea, QSlider, QToolButton, QToolTip, QVBoxLayout, QWidget)

from gui.shell.panels import fmt
from gui.shell.theme import PLOT_FAIL, PLOT_LINE, PLOT_PASS

NEAR = '#ff8c1a'
BEST = '#2f9e44'


_SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci_label(v):
    """1e-09 -> '10⁻⁹', 3.5e-09 -> '3.5×10⁻⁹', 2 -> '2'."""
    if not (np.isfinite(v) and v > 0):
        return ''
    if 1e-2 <= v < 1e4:
        return f"{v:.4g}"
    e = int(np.floor(np.log10(v) + 1e-9))
    m = round(v / 10 ** e, 3)
    if m >= 10:
        m, e = m / 10, e + 1
    p = '10' + str(e).translate(_SUP)
    return p if abs(m - 1) < 1e-9 else f"{m:g}×{p}"


class LogAxis(pg.AxisItem):
    """A log axis whose labels never overlap.  pyqtgraph's own labels every
    minor tick (2, 3 ... 9 x 10^n) once the view is inside one decade, so the
    labels run into each other when zoomed in.  Here: whole decades (every
    k-th, so they fit) when the view spans a decade or more, else round
    numbers evenly spaced in D, as many as fit the axis length."""

    LABEL_PX = 75                               # room one label needs
    MINOR = -999.5                              # level tag: ticks without labels

    def tickValues(self, minVal, maxVal, size):
        if not self.logMode:
            return super().tickValues(minVal, maxVal, size)
        lo, hi = sorted((minVal, maxVal))
        n_max = max(2, int(size // self.LABEL_PX))
        if hi - lo >= 1:
            decades = np.arange(np.ceil(lo), np.floor(hi) + 1)
            k = int(np.ceil(len(decades) / n_max)) or 1
            major = decades[::k]
            # Few decades in view: label 2 and 5 x 10^n as well.
            mids = (2, 5) if len(major) * 3 <= n_max else ()

            def at(ms):
                return [np.log10(m) + d for d in np.arange(np.floor(lo), np.ceil(hi))
                        for m in ms if lo <= np.log10(m) + d <= hi]
            levels = [(float(k), major.tolist())]
            if mids:
                levels.append((float(np.log10(2)), at(mids)))
            levels.append((self.MINOR, at([m for m in range(2, 10) if m not in mids])))
            return levels
        a, b = 10 ** lo, 10 ** hi
        raw = (b - a) / n_max
        mag = 10 ** np.floor(np.log10(raw))
        step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
        ticks = np.arange(np.ceil(a / step) * step, b + step * 1e-9, step)
        ticks = ticks[ticks > 0]
        return [(float(np.log10(step)), np.log10(ticks).tolist())]

    def tickStrings(self, values, scale, spacing):
        if not self.logMode:
            return super().tickStrings(values, scale, spacing)
        if spacing == self.MINOR:
            return [''] * len(values)             # 2..9 x 10^n between labelled decades
        return [sci_label(10 ** v) for v in values]


def make_plot(title, xlabel, ylabel, logx=False, logy=False):
    axes = {}
    if logx:
        axes['bottom'] = LogAxis('bottom')
    if logy:
        axes['left'] = LogAxis('left')
    p = pg.PlotWidget(background='w', axisItems=axes)
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


class Tile(QFrame):
    """One plot (or panel) of a result tab: an optional header row (e.g. a
    picker) above the content.  Double-clicking the content asks the view
    to enlarge the tile to the whole tab, or to put it back."""

    def __init__(self, view, name, content):
        super().__init__()
        self.setObjectName('Tile')
        self.view, self.name, self.content = view, name, content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.header = QHBoxLayout()
        self.header.setContentsMargins(4, 0, 0, 0)
        lay.addLayout(self.header)
        lay.addWidget(content, 1)
        self.watch(content)

    def add_header(self, *widgets):
        for w in widgets:
            self.header.addWidget(w)
        self.header.addStretch(1)

    def watch(self, widget):
        """Catch double-clicks on ``widget`` (and the viewport of a plot)."""
        widget.installEventFilter(self)
        if hasattr(widget, 'viewport'):
            widget.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonDblClick:
            self.view.toggle_max(self.name)
            return True
        return False


class _View(QWidget):
    """A result tab: its plots are tiles on a grid of equal cells, as MATLAB's
    tiledlayout.  ``LAYOUTS`` lists the arrangements the Layout box offers:
    (label, rows, columns, {tile: (row, column, rows spanned, columns
    spanned)}).  Double-clicking a plot fills the tab with it; again puts it
    back.  When the tab is smaller than the plots' minimum sizes it scrolls
    instead of laying them over each other."""

    LAYOUTS = ()
    layout_chosen = pyqtSignal(str)             # label of the arrangement picked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tiles = {}
        self.maxed = None                       # name of the enlarged tile
        self.layout_i = 0
        body = QWidget()
        body.setObjectName('ResultBody')
        self.grid = QGridLayout(body)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(6)
        self.layout_box = QComboBox()
        self.layout_box.setToolTip('How the plots share the tab')
        self.layout_box.activated.connect(self._layout_picked)
        self.hint = QLabel('Double-click a plot to enlarge it')
        self.hint.setObjectName('Note')
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 0)
        bar.setSpacing(6)
        bar.addStretch(1)
        bar.addWidget(self.hint)
        bar.addSpacing(16)
        self.layout_label = QLabel('Layout')
        bar.addWidget(self.layout_label)
        bar.addWidget(self.layout_box)
        scroll = QScrollArea()
        scroll.setObjectName('ResultScroll')
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # The note line sits below the scroll area: a wrapping label inside it
        # would make the area size the plots to their preferred heights.
        self.note = QLabel('')
        self.note.setObjectName('Note')
        self.note.setWordWrap(True)
        self.note.setContentsMargins(6, 2, 6, 4)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(bar)
        outer.addWidget(scroll, 1)
        outer.addWidget(self.note)
        self._outer = outer

    # -- tiles and layouts ------------------------------------------------------
    def add_tile(self, name, widget):
        self.tiles[name] = Tile(self, name, widget)
        return self.tiles[name]

    def add_strip(self, widget):
        """A full-width strip under the tiles (e.g. a text log), not a tile."""
        self._outer.insertWidget(self._outer.indexOf(self.note), widget)

    def finish_layout(self):
        """Fill the Layout box from LAYOUTS and place the tiles."""
        self.layout_box.clear()
        for label, *_ in self.LAYOUTS:
            self.layout_box.addItem(label)
        several = len(self.LAYOUTS) > 1
        self.layout_box.setVisible(several)
        self.layout_label.setVisible(several)
        self.apply_layout(0)

    def layout_names(self):
        return [label for label, *_ in self.LAYOUTS]

    def set_layout(self, label):
        """Use the arrangement called ``label`` (ignored if unknown)."""
        names = self.layout_names()
        if label in names:
            self.apply_layout(names.index(label))

    def _layout_picked(self, i):
        self.apply_layout(i)
        self.layout_chosen.emit(self.LAYOUTS[i][0])

    def _clear_grid(self):
        for t in self.tiles.values():
            self.grid.removeWidget(t)
        for i in range(self.grid.rowCount()):
            self.grid.setRowStretch(i, 0)
        for i in range(self.grid.columnCount()):
            self.grid.setColumnStretch(i, 0)

    def apply_layout(self, i):
        self.layout_i = i
        self.maxed = None
        self.layout_box.setCurrentIndex(i)
        _label, rows, cols, cells = self.LAYOUTS[i]
        self._clear_grid()
        for name, t in self.tiles.items():
            if name in cells:
                self.grid.addWidget(t, *cells[name])
            t.setVisible(name in cells)
        for r in range(rows):                   # equal cells
            self.grid.setRowStretch(r, 1)
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)

    def toggle_max(self, name):
        """Fill the tab with one tile, or put the layout back."""
        if self.maxed == name or name not in self.tiles:
            self.apply_layout(self.layout_i)
            return
        self._clear_grid()
        for n, t in self.tiles.items():
            t.setVisible(n == name)
        self.grid.addWidget(self.tiles[name], 0, 0)
        self.grid.setRowStretch(0, 1)
        self.grid.setColumnStretch(0, 1)
        self.maxed = name


class SwellingView(_View):
    LAYOUTS = (('Single', 1, 1, dict(plot=(0, 0, 1, 1))),)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = make_plot('Drop volume', 'Time (s)', 'V / V0')
        self.add_tile('plot', self.plot)
        self.finish_layout()

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
        self.note.setText(f"Swelling {fmt(100 * r['S'], 3)} % (threshold "
                          f"{100 * data['settings']['threshold']:g} %)."
                          if r.get('S') is not None and r['S'] == r['S'] else '')


class BagalkotView(_View):
    LAYOUTS = (('Side by side', 1, 2, dict(fit=(0, 0, 1, 1), sweep=(0, 1, 1, 1))),
               ('Stacked', 2, 1, dict(fit=(0, 0, 1, 1), sweep=(1, 0, 1, 1))))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit = make_plot('Drop volume: measured and model', 'Time (s)', 'Volume (uL)')
        self.sweep = make_plot('Fit misfit against D', 'D (m2/s)', 'F', logx=True)
        self.add_tile('fit', self.fit)
        self.add_tile('sweep', self.sweep)
        self.finish_layout()

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
                          f"{fmt(lo, 3)} to {fmt(hi, 3)} m2/s.")


# kD curves are coloured along one sequential colour map, low kD dark blue to
# high kD orange, so a thin colour scale replaces a legend entry per curve
# and works the same for 5 or 60 kD values.  The map's pale yellow end is
# left out: it doesn't show on white.
CURVE_MAP = pg.colormap.get('plasma')
CURVE_MAP_RANGE = (0.0, 0.82)
# With more kD curves than this, only the best and its neighbours show at
# first (the picker brings back any other); more make the plot unreadable.
MAX_SHOWN = 9


def curve_colour(i, n):
    """Colour of the i-th smallest of n kD values."""
    lo, hi = CURVE_MAP_RANGE
    x = lo + (hi - lo) * (i / (n - 1) if n > 1 else 0.5)
    return CURVE_MAP.map(np.array([x]), mode='qcolor')[0]


class KdScale(QWidget):
    """The colour scale beside the E-against-D plot: a bar running through
    the curve colours, smallest kD at the bottom, labelled with kD values
    (every one if they fit, else an even selection plus the best).  Hidden
    curves' labels are grey."""

    BAR_W = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values, self.best, self.hidden = [], None, set()
        self.setFixedWidth(78)
        self.setMinimumHeight(120)
        self.setToolTip('kD of each curve: colour scale (hover a curve for its values)')

    def set_values(self, values, best=None, hidden=()):
        self.values, self.best, self.hidden = list(values), best, set(hidden)
        self.update()

    def _labelled(self, rows_fit):
        n = len(self.values)
        if n <= rows_fit:
            return list(range(n))
        pick = set(np.round(np.linspace(0, n - 1, max(2, rows_fit - 1))).astype(int).tolist())
        if self.best in self.values:
            b = self.values.index(self.best)
            pick = {i for i in pick if abs(i - b) * (rows_fit - 1) >= n - 1 or i == b}
            pick.add(b)
        return sorted(pick)

    def paintEvent(self, _ev):
        n = len(self.values)
        if not n:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = p.fontMetrics()
        top, bottom = fm.height() + 8, self.height() - fm.height() // 2 - 4
        p.setPen(QColor('#222'))
        p.drawText(2, fm.ascent() + 2, 'kD')
        x0 = 4
        grad = QLinearGradient(0, bottom, 0, top)
        for i in range(n):
            grad.setColorAt(i / (n - 1) if n > 1 else 0.5, curve_colour(i, n))
        p.fillRect(QRectF(x0, top, self.BAR_W, bottom - top), grad)

        def y_of(i):
            return bottom - (bottom - top) * (i / (n - 1) if n > 1 else 0.5)
        rows_fit = max(2, int((bottom - top) // (fm.height() + 2)) + 1)
        bold = QFont(p.font())
        bold.setBold(True)
        for i in self._labelled(rows_fit):
            v, y = self.values[i], y_of(i)
            best = v == self.best
            p.setPen(QColor('#9aa3ad') if v in self.hidden else QColor('#222'))
            p.drawLine(QPointF(x0 + self.BAR_W, y), QPointF(x0 + self.BAR_W + 3, y))
            p.setFont(bold if best else self.font())
            p.drawText(QPointF(x0 + self.BAR_W + 6, y + fm.ascent() / 2 - 1),
                       f"{v:.3g}" + (' best' if best else ''))
        p.end()


class CurvePicker(QFrame):
    """A popup listing the kD curves with a check box each, and All / None /
    Best only.  The list scrolls, so any number of kD values fits."""

    def __init__(self, view):
        super().__init__(view, Qt.WindowType.Popup)
        self.setObjectName('CurvePicker')
        self.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(4)
        for text, slot in (('All', lambda: view._set_all(True)),
                           ('None', lambda: view._set_all(False)),
                           ('Best only', view._best_only)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        self.list = QListWidget()
        self.list.setToolTip('Tick the kD curves to show')
        v.addWidget(self.list)

    def popup(self, under):
        rows = min(max(self.list.count(), 1), 14)
        self.list.setFixedHeight(rows * self.list.sizeHintForRow(0) + 6
                                 if self.list.count() else 40)
        self.adjustSize()
        self.move(under.mapToGlobal(under.rect().bottomLeft()))
        self.show()


class YangView(_View):
    """Yang's fit: measured and fitted IFT; the error E against D for each kD
    (his Fig 7a), the curves shown chosen with a picker; the lowest E for
    each kD."""

    LAYOUTS = (('2 + 1 wide', 2, 2, dict(fit=(0, 0, 1, 1), err=(0, 1, 1, 1),
                                         curves=(1, 0, 1, 2))),
               ('1 wide + 2', 2, 2, dict(curves=(0, 0, 1, 2), fit=(1, 0, 1, 1),
                                         err=(1, 1, 1, 1))),
               ('3 in a row', 1, 3, dict(fit=(0, 0, 1, 1), err=(0, 1, 1, 1),
                                         curves=(0, 2, 1, 1))),
               ('Stacked', 3, 1, dict(fit=(0, 0, 1, 1), err=(1, 0, 1, 1),
                                      curves=(2, 0, 1, 1))))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit = make_plot('Interfacial tension: measured and fitted', 'Time (s)', 'IFT (mN/m)')
        self.curves = make_plot('Fit error E against D, for each kD', 'D (m2/s)', 'E (%)',
                                logx=True)
        self.err = make_plot('Lowest E for each kD', 'kD', 'E (%)', logx=True)
        self.curves.getAxis('bottom').enableAutoSIPrefix(False)    # plain powers of ten
        # No legend: the colour scale beside the plot says which kD each
        # colour is, and hovering a curve names it.
        self.curves.plotItem.legend.setVisible(False)
        self.scale = KdScale()
        self.curves.scene().sigMouseMoved.connect(self._hover)
        self.picker = CurvePicker(self)
        self.list = self.picker.list
        self.list.itemChanged.connect(self._toggled)
        self.pick_button = QToolButton()
        self.pick_button.setObjectName('PickerButton')
        self.pick_button.setArrowType(Qt.ArrowType.DownArrow)
        self.pick_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.pick_button.setText('kD curves')
        self.pick_button.setToolTip('Choose which kD curves the plot shows')
        self.pick_button.clicked.connect(lambda: self.picker.popup(self.pick_button))
        self.add_tile('fit', self.fit)
        self.add_tile('err', self.err)
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(self.curves, 1)
        row.addWidget(self.scale)
        tile = self.add_tile('curves', box)
        tile.watch(self.curves)
        tile.add_header(QLabel('Show'), self.pick_button)
        self.finish_layout()
        self.items = {}                        # kD -> plot items of its curve
        self.data_of = {}                      # kD -> (log10 D, E, D_min, E_min)
        self.best_kD = None

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
        if 1 < len(kD) <= 12:                   # a kD grid: tick its values, label those
            lg = np.log10(np.sort(kD))           # that are not too close to the last label
            gap = 0.07 * max(lg[-1] - lg[0], 1e-9)
            major, minor, last = [], [], -np.inf
            for v, x in zip(np.sort(kD), lg):
                if x - last >= gap:
                    major.append((float(x), f"{v:g}"))
                    last = x
                else:
                    minor.append((float(x), ''))
            ax.setTicks([major, minor])
        elif len(kD) > 12:                      # many: 1-2-5 steps, so labels don't overlap
            lo, hi = np.nanmin(kD), np.nanmax(kD)
            nice = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 2, 5)]
            ax.setTicks([[(float(np.log10(v)), f"{v:g}") for v in nice
                          if lo / 1.05 <= v <= hi * 1.05], []])
        else:
            ax.setTicks(None)
        self._draw_curves(data.get('curves') or [], r)
        self.note.setText(f"D = {fmt(r['D'], 4)} m2/s, k = {fmt(r['k'], 4)} m/s, kD = "
                          f"{fmt(r['kD'], 4)}, E = {fmt(r['E_percent'], 3)} %.")

    def _draw_curves(self, curves, r):
        p = self.curves
        p.clear()
        self.items = {}
        self.data_of = {}
        self.list.blockSignals(True)
        self.list.clear()
        lows = []
        n = len(curves)
        order = sorted(range(n), key=lambda i: curves[i]['kD'])
        best_i = next((i for i in order if abs(curves[i]['kD'] - r['kD'])
                       <= 1e-9 * max(1.0, r['kD'])), None)
        self.best_kD = curves[best_i]['kD'] if best_i is not None else None
        # Too many curves: show the best and its neighbours in kD at first.
        if n > MAX_SHOWN and best_i is not None:
            k = order.index(best_i)
            lo = min(max(0, k - MAX_SHOWN // 2), n - MAX_SHOWN)
            shown = set(order[lo:lo + MAX_SHOWN])
        else:
            shown = set(range(n))
        for rank, i in enumerate(order):
            cv = curves[i]
            colour = curve_colour(rank, n)
            D, E = _arr(cv['D']), _arr(cv['E'])
            best = i == best_i
            line = p.plot(D, E, pen=pg.mkPen(colour, width=2.5 if best else 1.5),
                          symbol='o', symbolSize=4, symbolBrush=colour, symbolPen=None)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.data_of[cv['kD']] = (np.log10(D), E, cv['D_min'], cv['E_min'])
            low = p.plot([cv['D_min']], [cv['E_min']], pen=None, symbol='star',
                         symbolSize=13 if best else 9, symbolBrush=BEST if best else colour,
                         symbolPen=pg.mkPen('#222', width=0.8))
            label = f"kD = {fmt(cv['kD'], 4)}" + (' (best)' if best else '')
            self.items[cv['kD']] = (line, low, label)
            on = i in shown
            line.setVisible(on)
            low.setVisible(on)
            lows.append(cv['E_min'])
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, cv['kD'])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            it.setForeground(pg.mkColor(colour))
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._legend()
        if lows:
            # Near the minima, as in Yang's figure; the user can zoom out.
            top = min(max(lows), 4.0 * min(lows) + 1.0)
            p.setYRange(0, max(top, 1.5 * min(lows)), padding=0.05)

    def _legend(self):
        """The colour scale (hidden curves greyed) and the picker's count."""
        kds = sorted(self.items)
        hidden = [k for k in kds if not self.items[k][0].isVisible()]
        self.scale.set_values(kds, self.best_kD, hidden)
        self.pick_button.setText(f"kD curves: {len(kds) - len(hidden)} of {len(kds)}")

    def _hover(self, pos):
        """Name the curve under the mouse (within 10 px) in a tooltip."""
        vb = self.curves.plotItem.vb
        if not self.data_of or not vb.sceneBoundingRect().contains(pos):
            QToolTip.hideText()
            return
        pt = vb.mapSceneToView(pos)
        xs, ys = vb.viewPixelSize()
        best, near = None, 10.0
        for kd, (lx, E, Dm, Em) in self.data_of.items():
            if not self.items[kd][0].isVisible() or not len(lx):
                continue
            with np.errstate(invalid='ignore'):
                d = np.hypot((lx - pt.x()) / xs, (E - pt.y()) / ys)
            if np.isfinite(d).any() and np.nanmin(d) < near:
                best, near = kd, float(np.nanmin(d))
        if best is None:
            QToolTip.hideText()
            return
        Dm, Em = self.data_of[best][2:]
        QToolTip.showText(QCursor.pos(), f"kD = {best:g}{' (best)' if best == self.best_kD else ''}"
                                         f"\nlowest E {fmt(Em, 3)} % at D = {sci_label(Dm)} m2/s")

    def _toggled(self, item):
        parts = self.items.get(item.data(Qt.ItemDataRole.UserRole))
        if parts:
            on = item.checkState() == Qt.CheckState.Checked
            for part in parts[:2]:
                part.setVisible(on)
            self._legend()

    def _set_all(self, on, keep=None):
        """Tick every curve (``on``) or none; ``keep`` stays ticked either way."""
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            tick = on or it.data(Qt.ItemDataRole.UserRole) == keep
            it.setCheckState(Qt.CheckState.Checked if tick else Qt.CheckState.Unchecked)
            for part in self.items[it.data(Qt.ItemDataRole.UserRole)][:2]:
                part.setVisible(tick)
        self.list.blockSignals(False)
        self._legend()

    def _best_only(self):
        self._set_all(False, keep=self.best_kD)


# Boundary segments of the domain (core.domain_builder), as the Domain window.
SEGMENTS = (('free_surface', 'Drop interface (solute enters here)', '#1c7ed6'),
            ('tip_step', 'Needle tip', '#e8590c'),
            ('inner_wall', 'Needle inner wall', '#f59f00'),
            ('top_wall', 'Top of the domain', '#868e96'))


class ModelView(_View):
    """The model Yang and the simulation solve on: the domain outline coloured
    by boundary, the mesh, and the FEM system's checks (the Domain, Mesh and
    FEM windows of the Pendant window, in one tab)."""

    LAYOUTS = (('Side by side', 1, 2, dict(plot=(0, 0, 1, 1), info=(0, 1, 1, 1))),
               ('Stacked', 2, 1, dict(plot=(0, 0, 1, 1), info=(1, 0, 1, 1))))

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
        self.add_tile('plot', self.plot)
        self.add_tile('info', scroll)
        self.finish_layout()

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
    concentration Cs(t), gamma(t) against the measured IFT, and diagnostics.

    A typed time shows the snapshot at the solver step nearest to it; when
    none is stored there, "Compute" asks for a re-run that stores it."""

    compute_requested = pyqtSignal(float)       # time (s) to add as a snapshot
    LAYOUTS = (('1 tall + 2', 2, 2, dict(field=(0, 0, 2, 1), cs=(0, 1, 1, 1),
                                         gamma=(1, 1, 1, 1))),
               ('2 + 1 wide', 2, 2, dict(cs=(0, 0, 1, 1), gamma=(0, 1, 1, 1),
                                         field=(1, 0, 1, 2))),
               ('3 in a row', 1, 3, dict(field=(0, 0, 1, 1), cs=(0, 1, 1, 1),
                                         gamma=(0, 2, 1, 1))))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self._asked_t = None                    # typed time with no stored snapshot
        self._want_t = None                     # time asked for, shown after the re-run
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
        row = QHBoxLayout()
        row.addWidget(QLabel('Time (s)'))
        self.time_edit = QLineEdit()
        self.time_edit.setPlaceholderText('e.g. 1200')
        self.time_edit.setToolTip('Show the concentration at this time')
        self.time_edit.setMaximumWidth(110)
        self.time_edit.returnPressed.connect(self.go_to_time)
        row.addWidget(self.time_edit)
        self.show_btn = QPushButton('Show')
        self.show_btn.clicked.connect(self.go_to_time)
        row.addWidget(self.show_btn)
        self.compute_btn = QPushButton('Compute')
        self.compute_btn.setToolTip('Run the simulation again, also storing this time')
        self.compute_btn.clicked.connect(self._compute)
        self.compute_btn.hide()
        row.addWidget(self.compute_btn)
        self.time_info = QLabel('')
        self.time_info.setObjectName('Note')
        row.addWidget(self.time_info, 1)
        self.field_lay.addLayout(row)
        self.cs = make_plot('Interface concentration', 'Time (s)', 'Cs (c / c_sat)')
        self.gamma = make_plot('Interfacial tension', 'Time (s)', 'IFT (mN/m)')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName('StepLog')
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setMaximumHeight(110)
        self.log.setToolTip('Step log of the solver (first and last steps)')
        self.add_tile('field', self.field_box)
        self.add_tile('cs', self.cs)
        self.add_tile('gamma', self.gamma)
        self.add_strip(self.log)
        self.finish_layout()

    def _make_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(4, 4.5), dpi=90, facecolor='white')
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumSize(220, 220)
        self.field_lay.insertWidget(0, self.canvas, 1)
        self.tiles['field'].watch(self.canvas)

    def set_data(self, data):
        self.data = data
        if self.canvas is None:
            self._make_canvas()
        snaps = data['snapshots']
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(snaps) - 1))
        self.slider.setValue(max(0, len(snaps) - 1))
        self.slider.blockSignals(False)
        self.compute_btn.hide()
        self.time_info.setText('')
        self._draw_field()
        if self._want_t is not None:
            t, self._want_t = self._want_t, None
            self.time_edit.setText(f"{t:g}")
            self.go_to_time()
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
        self.note.setText(
            f"D = {fmt(st['D'], 4)} m2/s, k = {fmt(st['k'], 4)} m/s (from {src}), kD = "
            f"{fmt(r['kD'], 4)}; {fmt(r['simulated_s'], 5)} s in {r['n_steps']} {steps}. "
            f"Final Cs {fmt(r['Cs_final'], 4)}; C from {fmt(r['C_min'], 3)} to "
            f"{fmt(r['C_max'], 3)} over the run.{g}")
        self.log.setPlainText('\n'.join(data.get('diagnostics') or []))

    def go_to_time(self):
        """Show the snapshot at the solver step nearest the typed time."""
        self.compute_btn.hide()
        if self.data is None:
            return
        try:
            t = float(self.time_edit.text().strip().replace(',', '.'))
        except ValueError:
            self.time_info.setText('Type a time in seconds.')
            return
        steps = _arr(self.data['curve']['t'])
        snaps = self.data['snapshots']
        if not len(steps) or not snaps:
            self.time_info.setText('No snapshots stored.')
            return
        if not (np.isfinite(t) and 0 < t <= steps[-1] * (1 + 1e-9)):
            self.time_info.setText(f"Outside the run: 0 to {fmt(steps[-1], 5)} s.")
            return
        ts = steps[np.argmin(np.abs(steps - t))]          # the step a re-run would store
        snap_t = np.array([s['t'] for s in snaps], float)
        j = int(np.argmin(np.abs(snap_t - ts)))
        self.slider.setValue(j)                            # nearest stored one meanwhile
        if abs(snap_t[j] - ts) <= 1e-9 * max(1.0, ts):
            self.time_info.setText('' if abs(ts - t) <= 1e-6 * max(1.0, t) else
                                   f"Nearest solver step: {fmt(ts, 5)} s.")
            return
        self._asked_t = t
        self.time_info.setText(f"Not stored; showing {fmt(snap_t[j], 5)} s.")
        self.compute_btn.show()

    def compute_ended(self):
        """The re-run ended; if it stored nothing new (failed, stopped), say so."""
        if self._want_t is not None:
            self._want_t = None
            self.time_info.setText('Not computed.')

    def _compute(self):
        self.compute_btn.hide()
        self.time_info.setText('Computing...')
        self._want_t = self._asked_t
        self.compute_requested.emit(self._asked_t)

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
        # The colour bar hangs on the drop's axes (not on a slice of the
        # figure), so drop and bar stay together, centred in the panel.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes('right', size='6%', pad=0.12)
        cb = fig.colorbar(tcf, cax=cax, ticks=np.linspace(0, 1, 6))
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
