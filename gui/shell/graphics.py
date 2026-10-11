"""
gui.shell.graphics
==================

The Graphics area: one tab widget whose tabs appear when their data exists.

    Start      shown when nothing is open: new session from a video, image
               or camera; open a session folder; recent sessions
    Frame      a session's logged frames (zoom, pan, step through) with
               their numbers, or the reference frame of a session being
               set up, with the set-up tools (crop box, scale reference,
               baseline row) and the fitted outline
    Log Plot   any logged column against time, QC pass / fail coloured;
               clicking a point shows that frame

Later steps add Mesh, Concentration and Fit Diagnostics tabs the same way.
"""

import math

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGraphicsEllipseItem,
                             QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsScene, QGraphicsView,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPushButton, QSlider, QTabWidget, QToolBar, QVBoxLayout,
                             QWidget)

from core import frame_log as fl
from core.session import SessionError
from gui.shell.icons import icon
from gui.shell.panels import fmt
from gui.shell.theme import PLOT_FAIL, PLOT_LINE, PLOT_PASS
from gui.styles import make_cosmetic_pen as _cosmetic_pen

COL_BASELINE = '#e03131'
COL_SCALE = '#00b8d9'
COL_EDGE = '#22c55e'
COL_FIT = '#ff8c1a'
COL_PICK = '#ffd400'


def make_cosmetic_pen(color, dashed=False):
    """Two screen pixels wide at every zoom; ``color`` may be '#rrggbb'."""
    return _cosmetic_pen(QColor(color), dashed=dashed)


def bgr_to_pixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


# ---------------------------------------------------------------------------
class StartPage(QWidget):
    open_requested = pyqtSignal()
    new_requested = pyqtSignal(str)          # 'video' | 'image' | 'camera'
    recent_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Canvas')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 30, 40, 30)
        card = QFrame()
        card.setObjectName('CalloutInfo')
        card.setMaximumWidth(640)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 18, 22, 18)
        title = QLabel('Drop Analysis Studio')
        title.setObjectName('PageTitle')
        lay.addWidget(title)
        note = QLabel('Start a session from a drop video, a single image or the camera: '
                      'prepare the first frame, then log the frames. Each session is a '
                      'folder holding the frames, their measurements (log.csv) and how '
                      'they were made (session.json); open one to look through it again.')
        note.setWordWrap(True)
        note.setObjectName('Note')
        lay.addWidget(note)
        row = QHBoxLayout()
        for kind, ic, text in (('video', 'video', 'New from video...'),
                               ('image', 'image', 'New from image...'),
                               ('camera', 'camera', 'New from camera...')):
            b = QPushButton(icon(ic), text)
            b.clicked.connect(lambda _=False, k=kind: self.new_requested.emit(k))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.btn_open = QPushButton(icon('open'), 'Open session folder...')
        self.btn_open.setObjectName('Primary')
        self.btn_open.clicked.connect(self.open_requested)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel('Recent sessions'))
        self.recent = QListWidget()
        self.recent.setMaximumHeight(170)
        self.recent.itemClicked.connect(
            lambda it: it.data(Qt.ItemDataRole.UserRole) and
            self.recent_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
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
    """Wheel zooms around the cursor, drag pans, Fit shows the whole image.

    Pick modes for the set-up tools (Esc cancels):
        'rect'     drag a rectangle          -> picked_rect(x, y, w, h)
        'points2'  click two points          -> picked_points(x1, y1, x2, y2)
        'hline'    click a row               -> picked_row(y)
    Coordinates are image pixels.  Overlays are drawn with pens that stay
    two screen pixels wide at every zoom."""
    picked_rect = pyqtSignal(int, int, int, int)
    picked_points = pyqtSignal(float, float, float, float)
    picked_row = pyqtSignal(float)
    pick_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._fit = True
        self._mode = None
        self._pts = []
        self._temp = []
        self._overlays = []

    # -- image --------------------------------------------------------------
    def set_image(self, pixmap):
        old = self.item.pixmap()
        size_changed = old.isNull() or old.size() != pixmap.size()
        self.item.setPixmap(pixmap)
        self.scene().setSceneRect(self.item.boundingRect())
        if size_changed or self._fit:
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

    # -- overlays -----------------------------------------------------------
    def clear_overlays(self):
        for it in self._overlays:
            self.scene().removeItem(it)
        self._overlays = []

    def _keep(self, item, color, dashed=False):
        item.setPen(make_cosmetic_pen(color, dashed=dashed))
        self.scene().addItem(item)
        self._overlays.append(item)

    def add_polyline(self, xs, ys, color, dashed=False):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 2:
            return
        path = QPainterPath(QPointF(xs[0] + .5, ys[0] + .5))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x + .5, y + .5)
        self._keep(QGraphicsPathItem(path), color, dashed)

    def add_line(self, x1, y1, x2, y2, color, dashed=False):
        self._keep(QGraphicsLineItem(x1 + .5, y1 + .5, x2 + .5, y2 + .5), color, dashed)

    def add_hline(self, y, color, dashed=False):
        self.add_line(0, y, max(self.item.pixmap().width() - 1, 1), y, color, dashed)

    def add_circle(self, cx, cy, r, color):
        self._keep(QGraphicsEllipseItem(cx - r + .5, cy - r + .5, 2 * r, 2 * r), color)

    # -- picking ------------------------------------------------------------
    def begin_pick(self, mode):
        self._end_pick()
        self._mode = mode
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()

    def picking(self):
        return self._mode is not None

    def cancel_pick(self):
        if self._mode is not None:
            self._end_pick()
            self.pick_cancelled.emit()

    def _end_pick(self):
        for it in self._temp:
            self.scene().removeItem(it)
        self._temp = []
        self._mode = None
        self._pts = []
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().unsetCursor()

    def _temp_item(self, item, color=COL_PICK):
        item.setPen(make_cosmetic_pen(color, dashed=True))
        self.scene().addItem(item)
        self._temp.append(item)
        return item

    def _scene_pt(self, pos):
        p = self.mapToScene(pos)
        r = self.item.boundingRect()
        return (min(max(p.x(), 0.0), max(r.width() - 1, 0.0)),
                min(max(p.y(), 0.0), max(r.height() - 1, 0.0)))

    def press_at(self, x, y):
        """A left click at image pixel (x, y) in the current pick mode."""
        if self._mode == 'rect':
            self._pts = [(x, y)]
            self._rubber = self._temp_item(QGraphicsRectItem(x, y, 0, 0))
        elif self._mode == 'points2':
            self._pts.append((x, y))
            self._temp_item(QGraphicsEllipseItem(x - 2, y - 2, 4, 4), '#ff3b30')
            if len(self._pts) == 1:
                self._guide = self._temp_item(QGraphicsLineItem(x, y, x, y))
            else:
                (x1, y1), (x2, y2) = self._pts
                self._end_pick()
                self.picked_points.emit(x1, y1, x2, y2)
        elif self._mode == 'hline':
            self._end_pick()
            self.picked_row.emit(float(round(y)))

    def move_to(self, x, y):
        if self._mode == 'rect' and self._pts:
            x0, y0 = self._pts[0]
            self._rubber.setRect(min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif self._mode == 'points2' and len(self._pts) == 1:
            self._guide.setLine(self._pts[0][0], self._pts[0][1], x, y)
        elif self._mode == 'hline':
            if not self._temp:
                self._guide = self._temp_item(QGraphicsLineItem())
            self._guide.setLine(0, y + .5, self.item.boundingRect().width(), y + .5)

    def release_at(self, x, y):
        if self._mode == 'rect' and self._pts:
            x0, y0 = self._pts[0]
            self._end_pick()
            self.picked_rect.emit(int(round(min(x0, x))), int(round(min(y0, y))),
                                  int(round(abs(x - x0))), int(round(abs(y - y0))))

    def mousePressEvent(self, e):
        if self._mode is None or e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.press_at(*self._scene_pt(e.position().toPoint()))

    def mouseMoveEvent(self, e):
        if self._mode is None:
            return super().mouseMoveEvent(e)
        self.move_to(*self._scene_pt(e.position().toPoint()))

    def mouseReleaseEvent(self, e):
        if self._mode == 'rect' and e.button() == Qt.MouseButton.LeftButton:
            self.release_at(*self._scene_pt(e.position().toPoint()))
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self._mode is not None:
            self.cancel_pick()
            return
        super().keyPressEvent(e)


# ---------------------------------------------------------------------------
class FrameViewer(QWidget):
    """Either a session's logged frames (step through them) or the
    reference frame of a session being set up, with its overlays, or the
    live camera picture."""
    frame_changed = pyqtSignal(int)          # frame_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder = None
        self.draft = None
        self.live_frame = None
        self.scale_marker = None             # ('line', p1, p2) | ('circle', c, r)
        self._order = []                     # frame indices sorted by time
        self._rows = {}
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
        self.chk_prepared.setToolTip('Show the frame after the threshold, as the fit sees it')
        self.chk_prepared.toggled.connect(lambda _: self.redraw())
        tb.addWidget(self.chk_prepared)
        self.chk_overlay = QCheckBox('Overlays')
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip('Baseline, scale reference and the fitted outline')
        self.chk_overlay.toggled.connect(lambda _: self.redraw())
        tb.addWidget(self.chk_overlay)
        lay.addWidget(tb)
        self.hint = QLabel('')
        self.hint.setObjectName('PickHint')
        self.hint.setWordWrap(True)
        self.hint.setVisible(False)
        lay.addWidget(self.hint)
        canvas = QWidget()
        canvas.setObjectName('Canvas')
        cl = QVBoxLayout(canvas)
        cl.setContentsMargins(6, 6, 6, 6)
        self.view = ZoomView()
        self.view.picked_rect.connect(lambda *_: self.end_pick())
        self.view.picked_points.connect(lambda *_: self.end_pick())
        self.view.picked_row.connect(lambda *_: self.end_pick())
        self.view.pick_cancelled.connect(self.end_pick)
        cl.addWidget(self.view, 1)
        self.info = QLabel('')
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setWordWrap(True)          # never forces the window wider
        cl.addWidget(self.info)
        self.nav = QWidget()
        nav = QHBoxLayout(self.nav)
        nav.setContentsMargins(0, 0, 0, 0)
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
        cl.addWidget(self.nav)
        lay.addWidget(canvas, 1)

    # -- picking hint ---------------------------------------------------------
    def begin_pick(self, mode, hint):
        self.hint.setText(hint + '  Press Esc to cancel.')
        self.hint.setVisible(True)
        self.view.begin_pick(mode)

    def end_pick(self):
        self.hint.setVisible(False)

    # -- session frames ---------------------------------------------------------
    def set_folder(self, folder, keep_position=False):
        cur = self.current() if keep_position else None
        self.draft = None
        self.live_frame = None
        self.scale_marker = None
        self.folder = folder
        rows = sorted(folder.log.rows, key=lambda r: r['time_s']) if folder else []
        self._rows = {r['frame_index']: r for r in rows}
        stored = set(folder.frame_indices()) if folder else set()
        self._order = [r['frame_index'] for r in rows if r['frame_index'] in stored]
        self.nav.setVisible(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self._order) - 1, 0))
        self.slider.setValue(self._order.index(cur) if cur in self._order else 0)
        self.slider.blockSignals(False)
        for w in self.btns + [self.slider]:
            w.setEnabled(len(self._order) > 1)
        if self._order:
            self.show_frame(self.current())
        else:
            self.view.item.setPixmap(QPixmap())
            self.view.clear_overlays()
            self.info.setText('No stored frames yet.')

    def current(self):
        if not self._order:
            return None
        return self._order[min(self.slider.value(), len(self._order) - 1)]

    def step(self, s):
        if not self._order:
            return
        i = {'first': 0, 'last': len(self._order) - 1}.get(s)
        if i is None:
            i = min(max(self.slider.value() + s, 0), len(self._order) - 1)
        if i == self.slider.value():
            self.show_frame(self._order[i])
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
        self.view.clear_overlays()
        if self.chk_overlay.isChecked():
            self.view.add_hline(self.folder.session.baseline['y_px'], COL_BASELINE, dashed=True)
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

    # -- a session being set up -----------------------------------------------
    def set_draft(self, draft):
        self.folder = None
        self.draft = draft
        self._order = []
        self.nav.setVisible(False)
        self.redraw()

    def set_live(self, frame):
        """Camera preview: show this raw frame (crop applied); None stops."""
        self.live_frame = frame
        self.redraw()

    def redraw(self):
        if self.folder is not None:
            self.show_frame(self.current())
            return
        d = self.draft
        if d is None:
            return
        raw = self.live_frame
        img = d.prepared_frame(raw) if self.chk_prepared.isChecked() else d.stored_frame(raw)
        self.view.set_image(bgr_to_pixmap(img))
        self.view.clear_overlays()
        if self.chk_overlay.isChecked():
            if d.baseline is not None:
                self.view.add_hline(d.baseline['y_px'], COL_BASELINE, dashed=True)
            m = self.scale_marker
            if m is not None and m[0] == 'line':
                self.view.add_line(*m[1], *m[2], COL_SCALE)
            elif m is not None:
                self.view.add_circle(*m[1], m[2], COL_SCALE)
            res = (d.preview or {}).get('results') if raw is None else None
            if res:
                ys = np.asarray(res['ys_use'])
                self.view.add_polyline(res['lxs'], ys, COL_EDGE)
                self.view.add_polyline(res['rxs'], ys, COL_EDGE)
                if res.get('ys_fit') is not None:
                    self.view.add_polyline(res['lxs_fit'], res['ys_fit'], COL_FIT, dashed=True)
                    self.view.add_polyline(res['rxs_fit'], res['ys_fit'], COL_FIT, dashed=True)
        H, W = img.shape[:2]
        parts = [f"{'Live camera' if raw is not None else 'Reference frame'}, {W} x {H} px"]
        if d.kind == 'video':
            parts.append(f"t = {fmt(d.sampling.get('t_start_s', 0.0), 6)} s (drop formed)")
        if d.preview and raw is None:
            r = d.preview
            if r.get('fit_ok'):
                parts.append(f"IFT {fmt(r['ift_mNm'])} mN/m")
                parts.append('QC pass' if r.get('qc_pass') else 'QC fail')
            else:
                parts.append(f"fit failed: {r.get('error', '')}")
        self.info.setText('   |   '.join(parts))


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
        self._maybe_add_plot(folder)

    def show_draft(self, draft):
        self.clear()
        self.frames.set_draft(draft)
        self.addTab(self.frames, icon('image'), 'Frame')

    def live_update(self, folder):
        """While logging: show the newest frame and grow the plot."""
        if self.frames.folder is not folder:
            self.clear()
            self.addTab(self.frames, icon('image'), 'Frame')
        self.frames.set_folder(folder)
        self.frames.step('last')
        self.logplot.set_log(folder.log)
        self._maybe_add_plot(folder)

    def _maybe_add_plot(self, folder):
        if len(folder.log) > 1 and self.logplot.combo.count() and self.indexOf(self.logplot) < 0:
            self.addTab(self.logplot, icon('plot'), 'Log Plot')
