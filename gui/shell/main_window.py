"""
gui.shell.main_window
=====================

Drop Analysis Studio's main window:

    ribbon (File menu + tabs of button groups)
    left:   Analysis Builder tree  |  Settings for the selected node
    centre: Graphics tabs (Start / Frame / Log Plot ...)
    bottom: Messages | Progress | Table (the frame log)
    status bar: state, source, scale, fluids, frames

The window holds either an open session folder (core.session.SessionFolder)
or a session being set up (core.session_draft.SessionDraft): a video, an
image or the camera, prepared on its reference frame.  Start logging creates
the folder and logs the frames in the background, with the table and plots
growing as frames arrive; when it ends the folder is opened like any other.

The panels are dock widgets: they can be resized, moved, closed and
floated; the layout is remembered between runs and Window > Reset Desktop
restores the default.
"""

import datetime
import re
import shutil
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QSettings, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QDockWidget, QFileDialog,
                             QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                             QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from core.frame_series import FrameSeriesCancelled
from core.session import SessionError, SessionFolder
from core.session_draft import DraftError, SessionDraft
from gui.shell import nodes
from gui.shell.analysis_pages import AnalysisPages
from gui.shell.draft_pages import DraftPages
from gui.shell.graphics import GraphicsArea
from gui.shell.icons import icon, status_icon
from gui.shell.pages import SettingsPanel
from gui.shell.panels import Job, LogTable, MessagesPanel, ProgressPanel, fmt
from gui.shell.ribbon import Ribbon
from gui.shell.theme import STYLESHEET, light_palette
from gui.window_utils import fit_to_screen

APP_NAME = 'Drop Analysis Studio'
MAX_RECENT = 8
REPORT_LATER = 'The printable report is the next step of the overhaul.'
CAMERA_TYPES = ['Basler', 'Webcam']
VIDEO_FILTER = 'Video files (*.mp4 *.avi *.mkv *.mov *.wmv);;All files (*)'
IMAGE_FILTER = 'Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)'
LIVE_NODES = ('source', 'crop', 'threshold')     # camera: show the live picture here


def _default_camera(camera_type):
    from core.camera_handler import CameraHandler
    return CameraHandler(camera_type)


class NewSessionDialog(QDialog):
    """Where the session folder goes: a location and a folder name."""
    BAD = re.compile(r'[\\/:*?"<>|]')

    def __init__(self, parent, location, name):
        super().__init__(parent)
        self.setWindowTitle('Start logging')
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('The frames, the log and the settings are saved in a new '
                             'folder.'))
        form = QFormLayout()
        self.location = QLineEdit(location)
        browse = QPushButton(icon('open'), 'Browse...')
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.location, 1)
        row.addWidget(browse)
        form.addRow('Location', row)
        self.name = QLineEdit(name)
        form.addRow('Folder name', self.name)
        lay.addLayout(form)
        self.error = QLabel('')
        self.error.setStyleSheet('color:#c0392b;')
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText('Start logging')
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(560, self.sizeHint().height())

    def _browse(self):
        p = QFileDialog.getExistingDirectory(self, 'Where to save the session', self.location.text())
        if p:
            self.location.setText(p)

    def path(self):
        return Path(self.location.text().strip()) / self.name.text().strip()

    def _accept(self):
        name = self.name.text().strip()
        loc = Path(self.location.text().strip())
        if not name or self.BAD.search(name) or name in ('.', '..'):
            self.error.setText('The folder name may not be empty or contain \\ / : * ? " < > |')
            return
        if not loc.is_dir():
            self.error.setText('The location is not an existing folder.')
            return
        if (loc / name).exists():
            self.error.setText('A folder with this name already exists there.')
            return
        self.accept()


class StudioWindow(QMainWindow):
    def __init__(self, settings_store=None, camera_factory=None):
        super().__init__()
        self.setObjectName('StudioRoot')
        self.setWindowTitle(APP_NAME)
        self.setPalette(light_palette())      # not the OS dark mode
        self.setStyleSheet(STYLESHEET)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks
                            | QMainWindow.DockOption.AllowTabbedDocks)
        self.store = settings_store or QSettings('IFTProperties', 'DropAnalysisStudio')
        self.camera_factory = camera_factory or _default_camera
        self.folder = None
        self.draft = None
        self.camera = None
        self.logging = False
        self.scale_method = 'needle'
        self.statuses = nodes.node_statuses(None)
        self.job = None
        self._pick_cb = None
        self._latest = None
        self._latest_lock = threading.Lock()
        self.draft_pages = DraftPages(self)
        self.analysis_pages = AnalysisPages(self)
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._camera_tick)

        self._make_actions()
        self._make_ribbon()
        self.graphics = GraphicsArea()
        self.setCentralWidget(self.graphics)
        self._make_docks()
        self._make_statusbar()
        self._wire()

        fit_to_screen(self, 1280, 800)
        self._default_state = self.saveState()
        geo, state = self.store.value('geometry'), self.store.value('state')
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        self._refresh_recent()
        self.refresh_status()
        self.select_node('session')
        self.messages.add('Ready. Start a new session or open a session folder.')

    # ------------------------------------------------------------------ setup
    def _act(self, name, text, slot=None, tip='', enabled=True, shortcut=None):
        a = QAction(icon(name), text, self)
        if slot is not None:
            a.triggered.connect(slot)
        a.setToolTip(tip or text.replace('\n', ' '))
        a.setStatusTip(tip or text.replace('\n', ' '))
        a.setEnabled(enabled)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        return a

    def _make_actions(self):
        A = self._act
        self.a_open = A('open', 'Open', self.open_session_dialog,
                        'Open a session folder (Ctrl+O)', shortcut='Ctrl+O')
        self.a_close = A('close', 'Close', self.close_current, 'Close the session')
        self.a_exit = A('exit', 'Exit', self.close)
        self.a_video = A('video', 'Video', lambda: self.new_session('video'),
                         'New session from a drop video')
        self.a_image = A('image', 'Image', lambda: self.new_session('image'),
                         'New session from a single image')
        self.a_camera = A('camera', 'Camera', lambda: self.new_session('camera'),
                          'New session from the live camera')
        self.a_crop = A('crop', 'Crop', lambda: self.select_node('crop'), 'Crop the frames')
        self.a_thresh = A('threshold', 'Threshold', lambda: self.select_node('threshold'),
                          'Set the threshold')
        self.a_scale = A('ruler', 'Calibrate\nScale', lambda: self.select_node('scale'),
                         'Calibrate the scale')
        self.a_base = A('baseline', 'Baseline', lambda: self.select_node('baseline'),
                        'Set the baseline at the needle tip')
        self.a_measure = A('drop', 'Measure\nFrame', self.measure_reference,
                           'Fit the reference frame (what the old Static mode did)')
        self.a_start = A('compute', 'Start\nLogging', self.start_logging,
                         'Create the session folder and log the frames')
        self.a_stop = A('stop', 'Stop', self.cancel_job, 'Stop the running task', enabled=False)
        self.a_rebuild = A('rebuild', 'Rebuild\nLog', lambda: self.select_node('log'),
                           'Fit the stored frames again (choose the columns first)')
        self.a_table = A('table', 'Frame\nTable', self.show_table, 'Show the frame log table')
        self.a_next = A('next', 'Next', self.go_next_required,
                        'Go to the first node that needs input')
        self.a_calib = A('calcurve', 'Load\nCalibration', self.load_calibration,
                         'Add a gamma(c) calibration to the session')
        self.a_swell = A('swell', 'Swelling\nCheck', lambda: self.select_node('swelling'),
                         'Swelling check on the logged volume')
        self.a_yang = A('inv', 'Yang\n(D, k)', lambda: self.select_node('yang'),
                        'Diffusivity and mass-transfer coefficient from IFT(t)')
        self.a_bagalkot = A('inv', 'Bagalkot\n(D)', lambda: self.select_node('bagalkot'),
                            'Diffusivity from the drop volume V(t)')
        self.a_results = A('plot', 'Results', lambda: self.select_node('results'),
                           'Stored analysis results')
        self.a_export = A('export', 'Export\nLog', self.export_log, 'Save a copy of log.csv')
        self.a_report = A('report', 'Report', tip=REPORT_LATER, enabled=False)
        self.a_reset = A('reset', 'Reset\nDesktop', self.reset_desktop,
                         'Put every panel back in its default place')

    def _make_ribbon(self):
        r = self.ribbon = Ribbon()
        m = r.file_menu
        new = m.addMenu(icon('new'), 'New session')
        for a in (self.a_video, self.a_image, self.a_camera):
            new.addAction(a)
        m.addAction(self.a_open)
        self.recent_menu = m.addMenu('Open recent')
        m.addAction(self.a_close)
        m.addSeparator()
        m.addAction(self.a_export)
        m.addSeparator()
        m.addAction(self.a_exit)
        home = r.add_page('Home')
        home.add_group('New Session', [self.a_video, self.a_image, self.a_camera])
        home.add_group('Session', [self.a_open, self.a_close])
        home.add_group('Prepare', [self.a_crop, self.a_thresh, self.a_scale, self.a_base])
        home.add_group('Logger', [self.a_measure, self.a_start, self.a_stop])
        home.add_group('Frame Log', [self.a_table, self.a_rebuild])
        home.add_group('Analyses', [self.a_swell, self.a_yang, self.a_bagalkot])
        home.add_group('Guide', [self.a_next])
        src = r.add_page('Source')
        src.add_group('New Session', [self.a_video, self.a_image, self.a_camera])
        src.add_group('Prepare', [self.a_crop, self.a_thresh])
        logp = r.add_page('Logger')
        logp.add_group('Calibrate', [self.a_scale, self.a_base])
        logp.add_group('Logging', [self.a_measure, self.a_start, self.a_stop])
        logp.add_group('Frame Log', [self.a_table, self.a_rebuild, self.a_export])
        an = r.add_page('Analyses')
        an.add_group('Inputs', [self.a_calib])
        an.add_group('Run', [self.a_swell, self.a_yang, self.a_bagalkot])
        res = r.add_page('Results')
        res.add_group('Results', [self.a_results])
        res.add_group('Export', [self.a_export, self.a_report])
        self.setMenuWidget(r)

    def _make_docks(self):
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setIndentation(14)
        self._items = {}
        for n in nodes.NODES:
            parent = self._items.get(n.parent, self.tree)
            it = QTreeWidgetItem(parent, [n.label, ''])
            it.setIcon(0, icon(n.icon))
            it.setData(0, Qt.ItemDataRole.UserRole, n.key)
            self._items[n.key] = it
        f = it.font(0)
        f.setBold(True)
        for n in nodes.NODES:
            if n.parent in (None, 'session'):
                self._items[n.key].setFont(0, f)
        self.tree.expandAll()
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.Fixed)
        hdr.resizeSection(1, 24)
        v.addWidget(self.tree, 1)
        legend = QWidget()
        g = QGridLayout(legend)
        g.setContentsMargins(8, 4, 8, 6)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(2)
        for i, s in enumerate(nodes.STATUSES):
            cell = QHBoxLayout()
            ic = QLabel()
            ic.setPixmap(status_icon(s).pixmap(13, 13))
            lbl = QLabel(nodes.STATUS_LABELS[s])
            lbl.setObjectName('LegendLabel')
            cell.addWidget(ic)
            cell.addWidget(lbl)
            cell.addStretch(1)
            g.addLayout(cell, i // 3, i % 3)
        v.addWidget(legend)
        self.dock_tree = self._dock('Analysis Builder', host, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.settings = SettingsPanel(self)
        self.dock_settings = self._dock('Settings', self.settings, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.splitDockWidget(self.dock_tree, self.dock_settings, Qt.Orientation.Horizontal)

        self.bottom = QTabWidget()
        self.bottom.setObjectName('BottomTabs')
        self.messages = MessagesPanel()
        self.progress = ProgressPanel()
        self.table = LogTable()
        self.bottom.addTab(self.messages, 'Messages')
        self.bottom.addTab(self.progress, 'Progress')
        self.bottom.addTab(self.table, 'Table - Frame Log')
        self.dock_bottom = self._dock('Output', self.bottom, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.resizeDocks([self.dock_tree, self.dock_settings], [250, 330], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_bottom], [190], Qt.Orientation.Vertical)

        win = self.ribbon.add_page('Window')
        win.add_group('Panels', [self._dock_action(d, name) for d, name in
                                 ((self.dock_tree, 'windows'), (self.dock_settings, 'params'),
                                  (self.dock_bottom, 'table'))])
        win.add_group('Layout', [self.a_reset])

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title.replace(' ', ''))
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(area, d)
        return d

    def _dock_action(self, dock, icon_name):
        a = dock.toggleViewAction()
        a.setIcon(icon(icon_name))
        return a

    def _make_statusbar(self):
        sb = self.statusBar()
        self.sb_icon = QLabel()
        self.sb_state = QLabel()
        self.sb_source = QLabel()
        self.sb_scale = QLabel()
        self.sb_fluids = QLabel()
        self.sb_frames = QLabel()
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(6, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(self.sb_icon)
        h.addWidget(self.sb_state)
        sb.addWidget(box)
        for w in (self.sb_source, self.sb_scale, self.sb_fluids, self.sb_frames):
            sb.addPermanentWidget(w)

    def _wire(self):
        self.tree.currentItemChanged.connect(
            lambda cur, _prev: cur and self._node_selected(cur.data(0, Qt.ItemDataRole.UserRole)))
        st = self.graphics.start
        st.open_requested.connect(self.open_session_dialog)
        st.new_requested.connect(self.new_session)
        st.recent_requested.connect(self.open_session)
        fr = self.graphics.frames
        fr.frame_changed.connect(self.table.select_frame)
        fr.view.picked_rect.connect(lambda *a: self._picked(*a))
        fr.view.picked_points.connect(lambda *a: self._picked(*a))
        fr.view.picked_row.connect(lambda *a: self._picked(*a))
        fr.view.pick_cancelled.connect(lambda: setattr(self, '_pick_cb', None))
        self.table.row_selected.connect(self._show_frame)
        self.graphics.logplot.frame_clicked.connect(self._show_frame)
        self.progress.cancel_requested.connect(self.cancel_job)

    # --------------------------------------------------------------- sessions
    def _leave_current(self):
        """Ask before throwing away a session being set up; False = stay."""
        if self.job_running():
            QMessageBox.information(self, 'Task running',
                                    'Wait for the running task to finish, or stop it first.')
            return False
        if self.draft is not None:
            r = QMessageBox.question(self, 'Discard the new session?',
                                     'The session being set up has not been logged yet. '
                                     'Discard its settings?')
            if r != QMessageBox.StandardButton.Yes:
                return False
        self._stop_camera()
        return True

    def open_session_dialog(self):
        start = self.store.value('last_dir', str(Path.home()))
        path = QFileDialog.getExistingDirectory(self, 'Open session folder', start)
        if path:
            self.store.setValue('last_dir', str(Path(path).parent))
            self.open_session(path)

    def open_session(self, path, ask=True):
        if ask and not self._leave_current():
            return False
        try:
            folder = SessionFolder.open(path)
        except (SessionError, OSError) as e:
            self.messages.add(f"Could not open {path}: {e}", 'error')
            QMessageBox.warning(self, 'Cannot open session', str(e))
            return False
        self.draft = None
        self.folder = folder
        self._remember(str(folder.path))
        self.setWindowTitle(f"{APP_NAME} - {folder.path.name}")
        self.messages.add(f"Opened {folder.path}: {len(folder.log)} frame(s) logged.", 'ok')
        for p in folder.problems:
            self.messages.add(p.message, 'warn')
        self.reload_views()
        self.select_node('session')
        return True

    def close_current(self):
        if self.draft is not None:
            self.discard_draft()
            return
        if self.folder is None or self.job_running():
            return
        self.messages.add(f"Closed {self.folder.path}.")
        self.folder = None
        self.setWindowTitle(APP_NAME)
        self.reload_views()
        self.select_node('session')

    def _remember(self, path):
        recent = [p for p in (self.store.value('recent') or []) if p != path]
        self.store.setValue('recent', ([path] + recent)[:MAX_RECENT])
        self._refresh_recent()

    def _refresh_recent(self):
        recent = [p for p in (self.store.value('recent') or []) if Path(p).is_dir()]
        self.graphics.start.set_recent(recent)
        self.recent_menu.clear()
        for p in recent:
            self.recent_menu.addAction(icon('project'), p, lambda p=p: self.open_session(p))
        self.recent_menu.setEnabled(bool(recent))

    # ----------------------------------------------------------- new sessions
    def new_session(self, kind, path=None, camera_type=None):
        """Start setting up a session from a video, an image or the camera.
        ``path`` / ``camera_type`` skip the file or camera question."""
        if not self._leave_current():
            return False
        last = self.store.value('last_media_dir', str(Path.home()))
        try:
            if kind == 'camera':
                if camera_type is None:
                    camera_type, ok = QInputDialog.getItem(
                        self, 'New session from camera', 'Camera', CAMERA_TYPES, 0, False)
                    if not ok:
                        return False
                draft = self._start_camera(camera_type)
            else:
                if path is None:
                    path, _ = QFileDialog.getOpenFileName(
                        self, f"New session from {'a video' if kind == 'video' else 'an image'}",
                        last, VIDEO_FILTER if kind == 'video' else IMAGE_FILTER)
                    if not path:
                        return False
                    self.store.setValue('last_media_dir', str(Path(path).parent))
                draft = (SessionDraft.from_video(path) if kind == 'video'
                         else SessionDraft.from_image(path))
        except (DraftError, OSError, RuntimeError, Exception) as e:   # camera errors are plain
            self._stop_camera()
            self.messages.add(f"Could not start a new session: {e}", 'error')
            QMessageBox.warning(self, 'Cannot start', str(e))
            return False
        self.folder = None
        self.draft = draft
        self.graphics.frames.scale_marker = None
        self.setWindowTitle(f"{APP_NAME} - new session ({draft.source.get('name', '')})")
        self.messages.add(f"New session from {kind}: {draft.source.get('name', '')}. "
                          f"Prepare the reference frame, then start logging.", 'ok')
        self.reload_views()
        self.select_node('session')
        return True

    def discard_draft(self):
        if self.draft is None or not self._leave_current():
            return
        self.messages.add('New session discarded.')
        self.draft = None
        self.setWindowTitle(APP_NAME)
        self.reload_views()
        self.select_node('session')

    def _start_camera(self, camera_type):
        cam = self.camera_factory(camera_type)
        cam.start()
        frame = None
        for _ in range(50):
            ok, f = cam.get_frame()
            if ok and f is not None:
                frame = f
                break
            time.sleep(0.05)
        if frame is None:
            cam.stop()
            raise RuntimeError('the camera started but sent no picture')
        self.camera = cam
        with self._latest_lock:
            self._latest = frame
        self.cam_timer.start(40)
        return SessionDraft.from_camera(frame, camera_type)

    def _stop_camera(self):
        self.cam_timer.stop()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:                   # releasing must never block closing
                pass
        self.camera = None
        with self._latest_lock:
            self._latest = None
        self.graphics.frames.live_frame = None

    def _camera_tick(self):
        if self.camera is None:
            return
        ok, f = self.camera.get_frame()
        if not ok or f is None:
            return
        with self._latest_lock:
            self._latest = f
        key = self._current_key()
        if self.draft is not None and key in LIVE_NODES and not self.graphics.frames.view.picking():
            self.graphics.frames.set_live(f)
        elif self.draft is not None and self.graphics.frames.live_frame is not None:
            self.graphics.frames.set_live(None)

    def latest_frame(self):
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    def capture_reference(self):
        f = self.latest_frame()
        if self.draft is None or f is None:
            return
        self.draft.set_reference(f)
        self.messages.add('Reference frame captured from the camera.', 'ok')
        self.graphics.frames.set_live(None)
        self.draft_changed(rebuild=True)

    def draft_changed(self, rebuild=False):
        """After an edit of the draft: tree, status bar, and optionally the page."""
        self.refresh_status(rebuild_page=rebuild)
        self.graphics.frames.redraw()

    # ----------------------------------------------------------- picking tools
    def pick(self, mode, hint, callback):
        """Let the user draw on the reference frame; ``callback`` gets the result."""
        if self.draft is None:
            return
        self.graphics.frames.set_live(None)
        self.graphics.setCurrentWidget(self.graphics.frames)
        self._pick_cb = callback
        self.graphics.frames.begin_pick(mode, hint)

    def _picked(self, *values):
        cb, self._pick_cb = self._pick_cb, None
        if cb is not None:
            cb(*values)

    # ----------------------------------------------------------------- views
    def reload_views(self):
        """After the session, draft or log changed: everything that shows data."""
        if self.draft is not None:
            self.graphics.show_draft(self.draft)
            self.table.set_log(None)
        else:
            self.graphics.show_session(self.folder)
            self.table.set_log(self.folder.log if self.folder else None)
        self.refresh_status()

    def refresh_status(self, rebuild_page=True):
        self.statuses = (nodes.draft_statuses(self.draft) if self.draft is not None
                         else nodes.node_statuses(self.folder))
        if self.logging:
            self.statuses['logger'] = ('ready', f"Logging: {len(self.folder.log)} frame(s) so far.")
        for key, it in self._items.items():
            status, reason = self.statuses[key]
            it.setIcon(1, status_icon(status))
            tip = f"{nodes.STATUS_LABELS[status]}" + (f": {reason}" if reason else '')
            it.setToolTip(0, tip)
            it.setToolTip(1, tip)
        self._items['session'].setText(
            0, 'New session' if self.draft is not None
            else (self.folder.path.name if self.folder else 'Session'))
        has = self.folder is not None
        busy = self.job_running()
        d = self.draft
        for a in (self.a_table, self.a_rebuild, self.a_export, self.a_swell, self.a_yang,
                  self.a_bagalkot, self.a_results):
            a.setEnabled(has and not busy)
        self.a_calib.setEnabled((has or d is not None) and not busy)
        self.a_close.setEnabled((has or d is not None) and not busy)
        for a in (self.a_open, self.a_video, self.a_image, self.a_camera):
            a.setEnabled(not busy)
        for a in (self.a_crop, self.a_thresh, self.a_scale, self.a_base):
            a.setEnabled(d is not None and not busy)
        self.a_measure.setEnabled(d is not None and not busy
                                  and d.scale is not None and d.baseline is not None)
        self.a_start.setEnabled(d is not None and not busy and not d.missing())
        self.a_stop.setEnabled(busy)
        self.a_next.setEnabled(has or d is not None)
        if rebuild_page:
            self.settings.refresh()
        self._update_statusbar()

    def _update_statusbar(self):
        f, d = self.folder, self.draft
        if self.job_running():
            state, text = 'ready', self.job.title
        elif d is not None:
            state, text = (('input', 'New session: setting up') if d.missing()
                           else ('ready', 'New session: ready to log'))
        elif f is None:
            state, text = 'off', 'No session'
        elif any(s == 'warn' for s, _ in self.statuses.values()):
            state, text = 'warn', 'Session open, with warnings'
        else:
            state, text = 'done', 'Session open'
        self.sb_icon.setPixmap(status_icon(state).pixmap(13, 13))
        self.sb_state.setText(f"<b>{text}</b>")
        if f is None and d is None:
            for w in (self.sb_source, self.sb_scale, self.sb_fluids, self.sb_frames):
                w.setText('')
            return
        if d is not None:
            src, scale, fluids = dict(d.source, kind=d.kind), d.scale, d.fluids
            frames = 'not logged yet'
        else:
            s = f.session
            src, scale, fluids = s.source, s.scale, s.fluids
            frames = f"{len(f.log)} frames"
        self.sb_source.setText(f"{src.get('kind', '').capitalize()}: {src.get('name') or '-'}")
        self.sb_scale.setText(f"{fmt(scale['pixel_to_m'] * 1e6)} um/px" if scale
                              else 'scale not set')
        cond = [fluids.get('name') or 'fluids']
        if 'MPa' in cond[0]:                     # preset names already give P and T
            pass
        elif fluids.get('pressure_MPa') is not None:
            cond.append(f"{fmt(fluids['pressure_MPa'])} MPa")
        if fluids.get('temperature_C') is not None and 'MPa' not in cond[0]:
            cond.append(f"{fmt(fluids['temperature_C'])} C")
        self.sb_fluids.setText(' | '.join(cond))
        self.sb_frames.setText(frames)

    def _current_key(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def select_node(self, key):
        it = self._items[key]
        if self.tree.currentItem() is it:
            self._node_selected(key)
        else:
            self.tree.setCurrentItem(it)

    def _node_selected(self, key):
        self.settings.show_node(key)
        if self.folder is None and self.draft is None:
            return
        if key == 'log':
            self.bottom.setCurrentWidget(self.table)
        elif key in ('source', 'crop', 'threshold', 'scale', 'baseline', 'logger'):
            self.graphics.setCurrentWidget(self.graphics.frames)

    def _show_frame(self, frame_index):
        if self.folder is None:
            return
        self.graphics.frames.show_frame(frame_index)
        self.graphics.setCurrentWidget(self.graphics.frames)
        self.table.select_frame(frame_index)

    def show_table(self):
        self.dock_bottom.show()
        self.bottom.setCurrentWidget(self.table)

    def go_next_required(self):
        key = nodes.next_required(self.statuses)
        if key is None:
            self.messages.add('Nothing needs input.', 'ok')
            self.bottom.setCurrentWidget(self.messages)
        else:
            self.select_node(key)

    def reset_desktop(self):
        self.restoreState(self._default_state)
        for d in (self.dock_tree, self.dock_settings, self.dock_bottom):
            d.show()
            d.setFloating(False)
        self.resizeDocks([self.dock_tree, self.dock_settings], [250, 330], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_bottom], [190], Qt.Orientation.Vertical)

    # --------------------------------------------------------------- actions
    def load_calibration(self, path=None):
        if self.folder is None and self.draft is None:
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, 'Load gamma(c) calibration',
                self.store.value('last_dir', str(Path.home())), 'CSV files (*.csv)')
            if not path:
                return
        try:
            if self.draft is not None:
                from core.calibration import CalibrationError, parse_calibration_template
                try:
                    parse_calibration_template(path)
                except CalibrationError as e:
                    raise SessionError(f"calibration not accepted: {e}") from None
                self.draft.calibration_path = str(path)
            else:
                self.folder.set_calibration(path)
        except (SessionError, OSError) as e:
            self.messages.add(str(e), 'error')
            QMessageBox.warning(self, 'Calibration not loaded', str(e))
            return
        self.messages.add(f"Calibration {Path(path).name} "
                          f"{'will be copied' if self.draft else 'copied'} into the session.", 'ok')
        self.refresh_status()

    def export_log(self):
        if self.folder is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save a copy of the frame log',
                                              str(Path.home() / f"{self.folder.path.name}_log.csv"),
                                              'CSV files (*.csv)')
        if not path:
            return
        try:
            shutil.copyfile(self.folder.log_path, path)
        except OSError as e:
            QMessageBox.warning(self, 'Could not save', str(e))
            return
        self.messages.add(f"Frame log saved to {path}.", 'ok')

    def rebuild_log(self, columns, prep_ops=None):
        """Fit the stored frames again (other columns and/or threshold)."""
        folder = self.folder

        def task(progress, is_cancelled):
            return folder.rebuild_log(
                columns, prep_ops=prep_ops,
                progress_cb=lambda i, n, rec: progress(i, n, f"Frame {i} of {n}"),
                is_cancelled=is_cancelled)

        def done(log):
            self.messages.add(f"Log rebuilt: {len(log)} frame(s), "
                              f"{sum(1 for r in log.rows if r['fit_ok'])} measured.", 'ok')
            self.reload_views()

        self.run_job('Rebuilding the frame log', task, done)

    def run_analysis(self, name, **settings):
        """Run a stored analysis (core.analyses) on the open session."""
        from core.analyses import LABELS, RUNNERS
        folder = self.folder
        if folder is None:
            return

        def task(progress, is_cancelled):
            return RUNNERS[name](folder, progress=progress, is_cancelled=is_cancelled, **settings)

        def done(data):
            r = data['result']
            if name == 'swelling':
                summary = r.get('message', '')
            elif name == 'bagalkot':
                summary = f"D = {fmt(r['D'], 4)} m2/s (SF {fmt(r['SF'], 3)})"
            else:
                summary = (f"D = {fmt(r['D'], 4)} m2/s, k = {fmt(r['k'], 4)} m/s, "
                           f"E = {fmt(r['E_percent'], 3)} %")
            self.messages.add(f"{LABELS[name]}: {summary}", 'ok')
            self.graphics.show_results(folder)
            self.graphics.show_result(name)

        def finally_():
            if self.folder is folder:
                self.select_node(name)

        self.run_job(f"Running the {LABELS[name].lower()}", task, done, on_finally=finally_)

    def measure_reference(self):
        d = self.draft
        if d is None or d.scale is None or d.baseline is None:
            return

        def done(rec):
            if rec['fit_ok']:
                self.messages.add(f"Reference frame: IFT {fmt(rec['ift_mNm'])} mN/m, beta "
                                  f"{fmt(rec['beta'])}, R0 {fmt(rec['R0_mm'])} mm, volume "
                                  f"{fmt(rec['volume_uL'])} uL, QC "
                                  f"{'passed' if rec['qc_pass'] else 'failed'}.",
                                  'ok' if rec['qc_pass'] else 'warn')
                for name, ok, detail in (rec['results'] or {}).get('qc_checks', []):
                    if not ok:
                        self.messages.add(f"QC {name}: {detail}", 'warn')
            else:
                self.messages.add(f"Reference frame fit failed: {rec['error']}", 'error')
            self.draft_changed(rebuild=True)

        self.run_job('Measuring the reference frame',
                     lambda progress, cancelled: d.measure_reference(), done)

    # ----------------------------------------------------------------- logging
    def start_logging(self, path=None):
        """Create the session folder and log the frames in the background.
        ``path`` skips the folder question."""
        d = self.draft
        if d is None or self.job_running():
            return False
        if d.missing():
            key, msg = d.missing()[0]
            self.messages.add(msg, 'warn')
            self.select_node(key)
            return False
        if path is None:
            loc = self.store.value('sessions_dir', str(Path.home() / 'Documents'))
            stem = Path(d.source.get('name', 'session')).stem.replace(' ', '_') or 'session'
            name = f"{stem}_{datetime.datetime.now():%Y-%m-%d_%H%M}"
            dlg = NewSessionDialog(self, loc, name)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return False
            path = dlg.path()
            self.store.setValue('sessions_dir', str(path.parent))
        try:
            session = d.to_session()
            folder = SessionFolder.create(path, session)
            if d.calibration_path:
                folder.set_calibration(d.calibration_path)
        except (DraftError, SessionError, OSError) as e:
            self.messages.add(f"Could not create the session: {e}", 'error')
            QMessageBox.warning(self, 'Cannot start logging', str(e))
            return False

        kind = d.kind
        if kind == 'video':
            times = d.sample_plan()['times']

            def task(progress, is_cancelled):
                return folder.log_video(
                    times, progress_cb=lambda i, n, rec: progress(i, n, f"Frame {i} of {n}"),
                    is_cancelled=is_cancelled)
        elif kind == 'image':
            ref = d.reference

            def task(progress, is_cancelled):
                folder.log_image(ref)
                progress(1, 1, 'Image measured')
                return 1
        else:
            plan = d.sample_plan()
            task = self._camera_task(folder, plan)

        self.logging = True
        self.draft = None
        self.folder = folder
        self.setWindowTitle(f"{APP_NAME} - {folder.path.name} (logging)")
        self.graphics.live_update(folder)
        self.table.set_log(folder.log)
        self.messages.add(f"Logging into {folder.path}.", 'ok')

        def done(n):
            self.messages.add(f"Logging finished: {n} frame(s).", 'ok')

        def finally_():
            self.logging = False
            try:
                folder.finish()
            except (SessionError, OSError) as e:
                self.messages.add(f"Could not seal the log: {e}", 'error')
            self._stop_camera()
            self.open_session(folder.path, ask=False)

        self.run_job('Logging frames', task, done, on_progress=lambda: self._log_progress(folder),
                     on_finally=finally_)
        return True

    def _camera_task(self, folder, plan):
        interval, duration = plan['interval_s'], plan['duration_s']
        n_expected = plan['n'] or 0

        def task(progress, is_cancelled):
            t0 = time.monotonic()
            k, next_t = 0, 0.0
            while not is_cancelled():
                now = time.monotonic() - t0
                if duration and now > duration:
                    break
                if now < next_t:
                    time.sleep(min(0.05, next_t - now))
                    continue
                frame = self.latest_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                clock = datetime.datetime.now().isoformat(timespec='milliseconds')
                folder.log_frame(frame, now, k, wall_clock=clock)
                k += 1
                progress(k, n_expected, f"{k} frame(s), t = {now:.1f} s")
                next_t += interval
                if next_t < time.monotonic() - t0:      # the fit was slower than the interval
                    next_t = time.monotonic() - t0
            return k
        return task

    def _log_progress(self, folder):
        if self.folder is not folder:
            return
        self.graphics.live_update(folder)
        self.table.set_log(folder.log)
        self.table.scrollToBottom()
        self.refresh_status(rebuild_page=False)

    # ------------------------------------------------------------------ jobs
    def job_running(self):
        return self.job is not None

    def run_job(self, title, fn, on_done, on_progress=None, on_finally=None):
        if self.job_running():
            QMessageBox.information(self, 'Task running', 'Another task is running.')
            return
        self.job = Job(title, fn, self)
        self.job.on_finally = on_finally

        def progressed(i, n, t):
            self.progress.update_progress(i, n, t)
            if on_progress is not None:
                on_progress()
        self.job.progressed.connect(progressed)
        self.job.succeeded.connect(on_done)
        self.job.failed.connect(self._job_failed)
        self.job.cancelled.connect(lambda: self.messages.add(f"{title}: stopped.", 'warn'))
        self.job.finished.connect(self._job_finished)
        self.progress.start(title)
        self.bottom.setCurrentWidget(self.progress)
        self.messages.add(f"{title}...")
        self.job.start()
        self.refresh_status()

    def _job_failed(self, msg):
        self.messages.add(f"{self.job.title} failed: {msg}", 'error')
        QMessageBox.warning(self, 'Task failed', msg)

    def _job_finished(self):
        job = self.job
        self.job = None
        job.deleteLater()
        self.progress.finish(f"{job.title}: finished.")
        if job.on_finally is not None:
            job.on_finally()
        self.refresh_status()

    def cancel_job(self):
        if self.job is not None:
            self.job.cancel()
            self.messages.add('Stopping after the current frame...', 'warn')

    def closeEvent(self, e):
        if self.job_running():
            r = QMessageBox.question(self, 'Task running',
                                     f"'{self.job.title}' is still running. Stop it and close? "
                                     f"Frames logged so far are kept.")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.job.cancel()
            self.job.wait(60000)
            if self.job is not None and self.job.on_finally is not None and self.logging:
                try:
                    self.folder.finish()
                except (SessionError, OSError):
                    pass
        self._stop_camera()
        self.store.setValue('geometry', self.saveGeometry())
        self.store.setValue('state', self.saveState())
        super().closeEvent(e)
