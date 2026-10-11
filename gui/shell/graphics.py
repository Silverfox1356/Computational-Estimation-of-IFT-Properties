"""
gui.shell.graphics
==================

The Graphics area: one tab widget whose tabs appear when their data exists.

    Start      shown when no session is open: open a folder, recent sessions
    Frame      any logged frame (zoom, pan, step through), with its numbers
    Log Plot   any logged column against time, QC pass / fail coloured;
               clicking a point shows that frame

Later steps add Mesh, Concentration and Fit Diagnostics tabs the same way.
"""

import math

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton, QSlider,
                             QTabWidget, QToolBar, QVBoxLayout, QWidget)

from core import frame_log as fl
from core.session import SessionError
from gui.shell.icons import icon
from gui.shell.panels import fmt
from gui.shell.theme import PLOT_FAIL, PLOT_LINE, PLOT_PASS


def bgr_to_pixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


# ---------------------------------------------------------------------------
class StartPage(QWidget):
    open_requested = pyqtSignal()
    recent_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Canvas')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 30, 40, 30)
        card = QFrame()
        card.setObjectName('CalloutInfo')
        card.setMaximumWidth(620)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 18, 22, 18)
        title = QLabel('Drop Analysis Studio')
        title.setObjectName('PageTitle')
        lay.addWidget(title)
        note = QLabel('A session folder holds the logged frames, their measurements '
                      '(log.csv) and how they were made (session.json). Open one to '
                      'look through its frames and plot the logged quantities.')
        note.setWordWrap(True)
        note.setObjectName('Note')
        lay.addWidget(note)
        row = QHBoxLayout()
        self.btn_open = QPushButton(icon('open'), 'Open session folder...')
        self.btn_open.setObjectName('Primary')
        self.btn_open.clicked.connect(self.open_requested)
        self.btn_new = QPushButton(icon('new'), 'New session...')
        self.btn_new.setEnabled(False)
        self.btn_new.setToolTip('Logging a new video, image or camera feed is '
                                'built in the next step of the overhaul.')
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_new)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel('Recent sessions'))
        self.recent = QListWidget()
        self.recent.setMaximumHeight(170)
        self.recent.itemActivated.connect(
            lambda it: self.recent_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.recent.itemClicked.connect(
            lambda it: self.recent_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.recent)
        outer.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addStretch(1)

    def set_recent(self, paths):
        self.recent.clear()
        for p in paths:
            it = QListWidgetItem(icon('project'), p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent.addItem(it)
        if not paths:
            it = QListWidgetItem('No recent sessions.')
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(it)


# ---------------------------------------------------------------------------
class ZoomView(QGraphicsView):
    """Wheel zooms around the cursor, drag pans, Fit shows the whole image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._fit = True

    def set_image(self, pixmap):
        first = self.item.pixmap().isNull()
        self.item.setPixmap(pixmap)
        self.scene().setSceneRect(self.item.boundingRect())
        if first or self._fit:
            self.fit()

    def fit(self):
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit = True

    def zoom(self, f):
        self.scale(f, f)
        self._fit = False

    def wheelEvent(self, e):
        self.zoom(1.2 if e.angleDelta().y() > 0 else 1 / 1.2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()


class FrameViewer(QWidget):
    frame_changed = pyqtSignal(int)          # frame_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder = None
        self._order = []                     # frame indices sorted by time
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.addAction(icon('zoomfit'), 'Fit to window', lambda: self.view.fit())
        tb.addAction(icon('zin'), 'Zoom in', lambda: self.view.zoom(1.25))
        tb.addAction(icon('zout'), 'Zoom out', lambda: self.view.zoom(0.8))
        tb.addSeparator()
        self.chk_prepared = QCheckBox('Show thresholded')
        self.chk_prepared.setToolTip('Show the frame after the session threshold, '
                                     'as the fit sees it')
        self.chk_prepared.toggled.connect(lambda _: self.show_frame(self.current()))
        tb.addWidget(self.chk_prepared)
        lay.addWidget(tb)
        canvas = QWidget()
        canvas.setObjectName('Canvas')
        cl = QVBoxLayout(canvas)
        cl.setContentsMargins(6, 6, 6, 6)
        self.view = ZoomView()
        cl.addWidget(self.view, 1)
        self.info = QLabel('')
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setWordWrap(True)          # never forces the window wider
        cl.addWidget(self.info)
        nav = QHBoxLayout()
        self.btns = []
        for name, tip, step in (('first', 'First frame', 'first'), ('prev', 'Previous frame', -1),
                                ('nextf', 'Next frame', 1), ('last', 'Last frame', 'last')):
            b = QPushButton(icon(name), '')
            b.setToolTip(tip)
            b.setFixedWidth(34)
            b.clicked.connect(lambda _=False, s=step: self.step(s))
            self.btns.append(b)
        nav.addWidget(self.btns[0])
        nav.addWidget(self.btns[1])
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._slider_moved)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.btns[2])
        nav.addWidget(self.btns[3])
        cl.addLayout(nav)
        lay.addWidget(canvas, 1)

    def set_folder(self, folder):
        self.folder = folder
        rows = sorted(folder.log.rows, key=lambda r: r['time_s']) if folder else []
        self._rows = {r['frame_index']: r for r in rows}
        stored = set(folder.frame_indices()) if folder else set()
        self._order = [r['frame_index'] for r in rows if r['frame_index'] in stored]
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self._order) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        for w in self.btns + [self.slider]:
            w.setEnabled(len(self._order) > 1)
        if self._order:
            self.show_frame(self._order[0])
        else:
            self.view.item.setPixmap(QPixmap())
            self.info.setText('No stored frames.')

    def current(self):
        if not self._order:
            return None
        return self._order[self.slider.value()]

    def step(self, s):
        if not self._order:
            return
        i = {'first': 0, 'last': len(self._order) - 1}.get(s)
        if i is None:
            i = min(max(self.slider.value() + s, 0), len(self._order) - 1)
        self.slider.setValue(i)

    def _slider_moved(self, i):
        if self._order:
            self.show_frame(self._order[i], emit=True)

    def show_frame(self, frame_index, emit=False):
        if frame_index is None or self.folder is None:
            return
        if frame_index in self._order:
            i = self._order.index(frame_index)
            if self.slider.value() != i:
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
        try:
            img = self.folder.read_frame(frame_index, prepared=self.chk_prepared.isChecked())
        except SessionError as e:
            self.info.setText(str(e))
            return
        self.view.set_image(bgr_to_pixmap(img))
        r = self._rows.get(frame_index, {})
        parts = [f"Frame {frame_index}", f"t = {fmt(r.get('time_s'), 6)} s"]
        for key in ('ift_mNm', 'volume_uL', 'R0_mm', 'beta'):
            if key in r:
                c = fl.COLUMNS[key]
                parts.append(f"{c.label} {fmt(r[key])} {c.unit}".strip())
        parts.append('QC pass' if r.get('qc_pass') else
                     ('QC fail' if r.get('fit_ok') else f"fit failed: {r.get('error', '')}"))
        n = len(self._order)
        pos = self._order.index(frame_index) + 1 if frame_index in self._order else 0
        self.info.setText(f"{pos} of {n}   |   " + '   |   '.join(parts))
        if emit:
            self.frame_changed.emit(frame_index)


# ---------------------------------------------------------------------------
class LogPlot(QWidget):
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        tb = QToolBar()
        tb.addWidget(QLabel(' Plot '))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.combo.currentIndexChanged.connect(self.redraw)
        tb.addWidget(self.combo)
        tb.addWidget(QLabel('  against time   '))
        self.chk_failed = QCheckBox('Include frames that failed QC')
        self.chk_failed.setChecked(True)
        self.chk_failed.toggled.connect(self.redraw)
        tb.addWidget(self.chk_failed)
        lay.addWidget(tb)
        self.plot = pg.PlotWidget(background='w')
        for ax in ('left', 'bottom'):
            a = self.plot.getAxis(ax)
            a.setPen(pg.mkPen('#555'))
            a.setTextPen(pg.mkPen('#222'))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel('bottom', 'Time (s)')
        self.line = self.plot.plot([], [], pen=pg.mkPen(PLOT_LINE, width=1))
        self.pts = pg.ScatterPlotItem(size=7, pen=pg.mkPen(None), hoverable=True)
        self.pts.sigClicked.connect(self._clicked)
        self.plot.addItem(self.pts)
        self.legend = QLabel('')
        self.legend.setObjectName('Note')
        self.legend.setWordWrap(True)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.legend)

    def set_log(self, log):
        self.log = log
        keep = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        for key in (log.columns if log else ()):
            if key in fl.OPTIONAL_KEYS:
                c = fl.COLUMNS[key]
                self.combo.addItem(f"{c.label} ({c.unit})" if c.unit else c.label, key)
        i = self.combo.findData(keep)
        if i < 0:
            i = max(self.combo.findData('ift_mNm'), 0)
        self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self.redraw()

    def redraw(self):
        key = self.combo.currentData()
        if self.log is None or key is None:
            self.line.setData([], [])
            self.pts.setData([])
            return
        rows = sorted(self.log.rows, key=lambda r: r['time_s'])
        if not self.chk_failed.isChecked():
            rows = [r for r in rows if r['qc_pass']]
        rows = [r for r in rows if math.isfinite(r[key])]
        t = np.array([r['time_s'] for r in rows])
        y = np.array([r[key] for r in rows])
        self.line.setData(t, y)
        self.pts.setData([dict(pos=(a, b), data=r['frame_index'],
                               brush=pg.mkBrush(PLOT_PASS if r['qc_pass'] else PLOT_FAIL))
                          for a, b, r in zip(t, y, rows)])
        c = fl.COLUMNS[key]
        self.plot.setLabel('left', f"{c.label} ({c.unit})" if c.unit else c.label)
        self.plot.enableAutoRange()
        n_fail = sum(1 for r in rows if not r['qc_pass'])
        self.legend.setText(f"  {len(rows)} point(s). Green: passed QC. Red: failed QC "
                            f"({n_fail}). Click a point to show its frame.")

    def _clicked(self, _item, points, *_):
        if len(points):
            self.frame_clicked.emit(int(points[0].data()))


# ---------------------------------------------------------------------------
class GraphicsArea(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('GraphicsTabs')
        self.setDocumentMode(False)
        self.start = StartPage()
        self.frames = FrameViewer()
        self.logplot = LogPlot()
        self.show_session(None)

    def show_session(self, folder):
        self.clear()
        if folder is None:
            self.addTab(self.start, icon('project'), 'Start')
            return
        self.frames.set_folder(folder)
        self.logplot.set_log(folder.log)
        self.addTab(self.frames, icon('image'), 'Frame')
        if len(folder.log) > 1 and self.logplot.combo.count():
            self.addTab(self.logplot, icon('plot'), 'Log Plot')
