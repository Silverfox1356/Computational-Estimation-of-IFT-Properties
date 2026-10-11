"""
gui.shell.main_window
=====================

Drop Analysis Studio's main window:

    ribbon (File menu + tabs of button groups)
    left:   Analysis Builder tree  |  Settings for the selected node
    centre: Graphics tabs (Start / Frame / Log Plot ...)
    bottom: Messages | Progress | Table (the frame log)
    status bar: state, source, scale, fluids, frames

The panels are dock widgets: they can be resized, moved, closed and
floated; the layout is remembered between runs and Window > Reset Desktop
restores the default.  Everything revolves around one open session folder
(core.session.SessionFolder).
"""

import shutil
from pathlib import Path

from PyQt6.QtCore import QSettings, QSize, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QDockWidget, QFileDialog, QGridLayout, QHBoxLayout,
                             QLabel, QMainWindow, QMessageBox, QTabWidget,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from core.session import SessionError, SessionFolder
from gui.shell import nodes
from gui.shell.graphics import GraphicsArea
from gui.shell.icons import icon, status_icon
from gui.shell.pages import SettingsPanel
from gui.shell.panels import Job, LogTable, MessagesPanel, ProgressPanel, fmt
from gui.shell.ribbon import Ribbon
from gui.shell.theme import STYLESHEET, light_palette
from gui.window_utils import fit_to_screen

APP_NAME = 'Drop Analysis Studio'
MAX_RECENT = 8
STEP3 = 'Built in the next step of the overhaul (logging a video, image or camera feed).'
STEP4 = 'Built in a later step (running analyses from a session).'


class StudioWindow(QMainWindow):
    def __init__(self, settings_store=None):
        super().__init__()
        self.setObjectName('StudioRoot')
        self.setWindowTitle(APP_NAME)
        self.setPalette(light_palette())      # not the OS dark mode
        self.setStyleSheet(STYLESHEET)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks
                            | QMainWindow.DockOption.AllowTabbedDocks)
        self.store = settings_store or QSettings('IFTProperties', 'DropAnalysisStudio')
        self.folder = None
        self.statuses = nodes.node_statuses(None)
        self.job = None

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
        self.messages.add('Ready. Open a session folder to begin.')

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
        self.a_close = A('close', 'Close', self.close_session, 'Close the session')
        self.a_exit = A('exit', 'Exit', self.close)
        self.a_video = A('video', 'Video', tip='Log a drop video. ' + STEP3, enabled=False)
        self.a_image = A('image', 'Image', tip='Log a single image. ' + STEP3, enabled=False)
        self.a_camera = A('camera', 'Camera', tip='Log a live camera feed. ' + STEP3, enabled=False)
        self.a_crop = A('crop', 'Crop', tip=STEP3, enabled=False)
        self.a_thresh = A('threshold', 'Threshold', tip=STEP3, enabled=False)
        self.a_scale = A('ruler', 'Calibrate\nScale', tip=STEP3, enabled=False)
        self.a_base = A('baseline', 'Baseline', tip=STEP3, enabled=False)
        self.a_start = A('compute', 'Start\nLogging', tip=STEP3, enabled=False)
        self.a_stop = A('stop', 'Stop', self.cancel_job, 'Stop the running task', enabled=False)
        self.a_rebuild = A('rebuild', 'Rebuild\nLog', lambda: self.select_node('log'),
                           'Fit the stored frames again (choose the columns first)')
        self.a_table = A('table', 'Frame\nTable', self.show_table, 'Show the frame log table')
        self.a_next = A('next', 'Next', self.go_next_required,
                        'Go to the first node that needs input')
        self.a_calib = A('calcurve', 'Load\nCalibration', self.load_calibration,
                         'Copy a gamma(c) calibration into the session')
        self.a_swell = A('swell', 'Swelling\nCheck', tip=STEP4, enabled=False)
        self.a_yang = A('inv', 'Yang\n(D, k)', tip=STEP4, enabled=False)
        self.a_bagalkot = A('inv', 'Bagalkot\n(D)', tip=STEP4, enabled=False)
        self.a_export = A('export', 'Export\nLog', self.export_log, 'Save a copy of log.csv')
        self.a_report = A('report', 'Report', tip=STEP4, enabled=False)
        self.a_reset = A('reset', 'Reset\nDesktop', self.reset_desktop,
                         'Put every panel back in its default place')

    def _make_ribbon(self):
        r = self.ribbon = Ribbon()
        m = r.file_menu
        m.addAction(self.a_open)
        self.recent_menu = m.addMenu('Open recent')
        m.addAction(self.a_close)
        m.addSeparator()
        m.addAction(self.a_export)
        m.addSeparator()
        m.addAction(self.a_exit)
        home = r.add_page('Home')
        home.add_group('Session', [self.a_open, self.a_close])
        home.add_group('Source', [self.a_video, self.a_image, self.a_camera])
        home.add_group('Logger', [self.a_start, self.a_stop])
        home.add_group('Frame Log', [self.a_table, self.a_rebuild])
        home.add_group('Analyses', [self.a_swell, self.a_yang, self.a_bagalkot])
        home.add_group('Guide', [self.a_next])
        home.add_group('Layout', [self.a_reset])
        src = r.add_page('Source')
        src.add_group('Source', [self.a_video, self.a_image, self.a_camera])
        src.add_group('Prepare', [self.a_crop, self.a_thresh])
        logp = r.add_page('Logger')
        logp.add_group('Calibrate', [self.a_scale, self.a_base])
        logp.add_group('Logging', [self.a_start, self.a_stop])
        logp.add_group('Frame Log', [self.a_table, self.a_rebuild, self.a_export])
        an = r.add_page('Analyses')
        an.add_group('Inputs', [self.a_calib])
        an.add_group('Run', [self.a_swell, self.a_yang, self.a_bagalkot])
        res = r.add_page('Results')
        res.add_group('Export', [self.a_export, self.a_report])
        self.setMenuWidget(r)

    def _make_docks(self):
        # Analysis Builder
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
        self.graphics.start.open_requested.connect(self.open_session_dialog)
        self.graphics.start.recent_requested.connect(self.open_session)
        self.graphics.frames.frame_changed.connect(self.table.select_frame)
        self.table.row_selected.connect(self._show_frame)
        self.graphics.logplot.frame_clicked.connect(self._show_frame)
        self.progress.cancel_requested.connect(self.cancel_job)

    # --------------------------------------------------------------- sessions
    def open_session_dialog(self):
        start = self.store.value('last_dir', str(Path.home()))
        path = QFileDialog.getExistingDirectory(self, 'Open session folder', start)
        if path:
            self.store.setValue('last_dir', str(Path(path).parent))
            self.open_session(path)

    def open_session(self, path):
        if self.job_running():
            QMessageBox.information(self, 'Task running',
                                    'Wait for the running task to finish, or stop it first.')
            return False
        try:
            folder = SessionFolder.open(path)
        except (SessionError, OSError) as e:
            self.messages.add(f"Could not open {path}: {e}", 'error')
            QMessageBox.warning(self, 'Cannot open session', str(e))
            return False
        self.folder = folder
        self._remember(str(folder.path))
        self.setWindowTitle(f"{APP_NAME} - {folder.path.name}")
        self.messages.add(f"Opened {folder.path}: {len(folder.log)} frame(s) logged.", 'ok')
        for p in folder.problems:
            self.messages.add(p.message, 'warn')
        self.reload_views()
        self.select_node('session')
        return True

    def close_session(self):
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

    # ----------------------------------------------------------------- views
    def reload_views(self):
        """After the session or its log changed: everything that shows data."""
        self.graphics.show_session(self.folder)
        self.table.set_log(self.folder.log if self.folder else None)
        self.refresh_status()

    def refresh_status(self):
        self.statuses = nodes.node_statuses(self.folder)
        for key, it in self._items.items():
            status, reason = self.statuses[key]
            it.setIcon(1, status_icon(status))
            tip = f"{nodes.STATUS_LABELS[status]}" + (f": {reason}" if reason else '')
            it.setToolTip(0, tip)
            it.setToolTip(1, tip)
        self._items['session'].setText(0, self.folder.path.name if self.folder else 'Session')
        has = self.folder is not None
        busy = self.job_running()
        for a in (self.a_close, self.a_table, self.a_rebuild, self.a_export, self.a_calib):
            a.setEnabled(has and not busy)
        self.a_open.setEnabled(not busy)
        self.a_stop.setEnabled(busy)
        self.a_next.setEnabled(has)
        self.settings.refresh()
        self._update_statusbar()

    def _update_statusbar(self):
        f = self.folder
        if self.job_running():
            state, text = 'ready', self.job.title
        elif f is None:
            state, text = 'off', 'No session'
        elif any(s == 'warn' for s, _ in self.statuses.values()):
            state, text = 'warn', 'Session open, with warnings'
        else:
            state, text = 'done', 'Session open'
        self.sb_icon.setPixmap(status_icon(state).pixmap(13, 13))
        self.sb_state.setText(f"<b>{text}</b>")
        if f is None:
            for w in (self.sb_source, self.sb_scale, self.sb_fluids, self.sb_frames):
                w.setText('')
            return
        s = f.session
        self.sb_source.setText(f"{s.source.get('kind', '').capitalize()}: "
                               f"{s.source.get('name') or '-'}")
        self.sb_scale.setText(f"{fmt(s.scale['pixel_to_m'] * 1e6)} um/px")
        cond = [s.fluids.get('name') or 'fluids']
        if s.fluids.get('pressure_MPa') is not None:
            cond.append(f"{fmt(s.fluids['pressure_MPa'])} MPa")
        if s.fluids.get('temperature_C') is not None:
            cond.append(f"{fmt(s.fluids['temperature_C'])} C")
        self.sb_fluids.setText(' | '.join(cond))
        self.sb_frames.setText(f"{len(f.log)} frames")

    def select_node(self, key):
        it = self._items[key]
        if self.tree.currentItem() is it:
            self._node_selected(key)
        else:
            self.tree.setCurrentItem(it)

    def _node_selected(self, key):
        self.settings.show_node(key)
        if self.folder is None:
            return
        if key == 'log':
            self.bottom.setCurrentWidget(self.table)
        elif key in ('source', 'crop', 'threshold', 'scale', 'baseline', 'logger'):
            self.graphics.setCurrentWidget(self.graphics.frames)

    def _show_frame(self, frame_index):
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
    def load_calibration(self):
        if self.folder is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load gamma(c) calibration',
                                              self.store.value('last_dir', str(Path.home())),
                                              'CSV files (*.csv)')
        if not path:
            return
        try:
            self.folder.set_calibration(path)
        except (SessionError, OSError) as e:
            self.messages.add(str(e), 'error')
            QMessageBox.warning(self, 'Calibration not loaded', str(e))
            return
        self.messages.add(f"Calibration {Path(path).name} copied into the session.", 'ok')
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

    def rebuild_log(self, columns):
        folder = self.folder

        def task(progress, is_cancelled):
            return folder.rebuild_log(
                columns, progress_cb=lambda i, n, rec: progress(i, n, f"Frame {i} of {n}"),
                is_cancelled=is_cancelled)

        def done(log):
            self.messages.add(f"Log rebuilt: {len(log)} frame(s), "
                              f"{sum(1 for r in log.rows if r['fit_ok'])} measured.", 'ok')
            self.reload_views()

        self.run_job('Rebuilding the frame log', task, done)

    # ------------------------------------------------------------------ jobs
    def job_running(self):
        return self.job is not None

    def run_job(self, title, fn, on_done):
        if self.job_running():
            QMessageBox.information(self, 'Task running', 'Another task is running.')
            return
        self.job = Job(title, fn, self)
        self.job.progressed.connect(lambda i, n, t: self.progress.update_progress(i, n, t))
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
        title = self.job.title
        self.job.deleteLater()
        self.job = None
        self.progress.finish(f"{title}: finished.")
        self.refresh_status()

    def cancel_job(self):
        if self.job is not None:
            self.job.cancel()
            self.messages.add('Stopping after the current frame...', 'warn')

    def closeEvent(self, e):
        if self.job_running():
            r = QMessageBox.question(self, 'Task running',
                                     f"'{self.job.title}' is still running. Stop it and close?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.job.cancel()
            self.job.wait(30000)
        self.store.setValue('geometry', self.saveGeometry())
        self.store.setValue('state', self.saveState())
        super().closeEvent(e)
