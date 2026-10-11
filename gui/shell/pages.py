"""
gui.shell.pages
===============

The Settings panel: one page per Analysis Builder node.  Every page starts
with the node's title, where it sits in the tree, and its status with the
reason, then its sections.

In this step the pages show what the session recorded.  Two actions are
live: loading a gamma(c) calibration into the session, and rebuilding the
frame log (with a different choice of columns).  Editing the logging
settings and running analyses come in later steps; their buttons are shown
disabled with a note saying so.
"""

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QFormLayout, QFrame, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton,
                             QScrollArea, QSizePolicy, QToolButton, QVBoxLayout,
                             QWidget)

from core import frame_log as fl
from gui.shell.icons import icon, status_icon
from gui.shell.nodes import NODE, STATUS_LABELS, children
from gui.shell.panels import fmt

LATER_LOGGING = ('Fixed for this session: to change it, start a new session from the same '
                 'source (File > New session) and log again.')
ANALYSIS_KEYS = ('swelling', 'bagalkot', 'yang', 'yang_model', 'results', 'threshold')
LATER_ANALYSIS = ('Running analyses from a session is built in a later step; until then '
                  'use the Pendant Drop window.')


PLANS = {'dense_early': 'Dense early, then sparser', 'interval': 'One frame every N seconds'}


def sampling_rows(kind, sampling):
    """(label, value, unit) rows describing how frames were sampled."""
    s = sampling or {}
    if kind == 'image':
        return [('Plan', 'Single image, logged as one frame at t = 0', '')]
    if kind == 'camera':
        dur = s.get('duration_s')
        return [('Interval', fmt(s.get('interval_s')), 's'),
                ('Stop after', f"{fmt(dur / 60.0)} min" if dur else 'stopped by hand', '')]
    rows = [('Plan', PLANS.get(s.get('mode'), s.get('mode') or '-'), '')]
    if s.get('mode') == 'interval':
        rows.append(('Interval', fmt(s.get('interval_s')), 's'))
    else:
        rows.append(('Frames requested', s.get('n_frames', '-'), ''))
        rows.append(('First step', fmt(s['first_step_s']) if s.get('first_step_s') else 'automatic',
                     's' if s.get('first_step_s') else ''))
    rows.append(('Drop formed at', fmt(s.get('t_start_s', 0.0)), 's'))
    rows.append(('Logged until', fmt(s['t_end_s']) if s.get('t_end_s') is not None
                 else 'end of the video', 's' if s.get('t_end_s') is not None else ''))
    if s.get('planned_frames'):
        rows.append(('Frames planned', s['planned_frames'], ''))
    return rows


def _crumb(key):
    parts = []
    k = NODE[key].parent
    while k is not None:
        parts.append(NODE[k].label)
        k = NODE[k].parent
    return ' > '.join(reversed(parts)) or 'Analysis Builder'


class Section(QWidget):
    """A collapsible section with a form inside."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName('SectionHeader')
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setArrowType(Qt.ArrowType.DownArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._toggle)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(4, 6, 2, 2)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(5)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.header)
        lay.addWidget(self.body)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.header.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def field(self, label, value, unit='', result=False):
        v = QLabel(str(value))
        v.setObjectName('FieldValueResult' if result else 'FieldValue')
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.setWordWrap(True)
        v.setToolTip(str(value))             # long paths are clipped, not lost
        v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.setMinimumWidth(60)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(v, 1)
        u = QLabel(unit)
        u.setObjectName('FieldUnit')
        u.setMinimumWidth(42)
        row.addWidget(u)
        self.form.addRow(label, row)

    def widget(self, w):
        self.form.addRow(w)


def callout(text, kind='warn'):
    box = QFrame()
    box.setObjectName({'warn': 'Callout', 'good': 'CalloutGood', 'info': 'CalloutInfo'}[kind])
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 6, 8, 6)
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lay.addWidget(lbl)
    return box


def note(text):
    lbl = QLabel(text)
    lbl.setObjectName('Note')
    lbl.setWordWrap(True)
    return lbl


class SettingsPanel(QScrollArea):
    """Shows the page of the selected node; rebuilt on every refresh."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.key = 'session'
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(280)

    def show_node(self, key):
        self.key = key
        self.refresh()

    def refresh(self):
        body = QWidget()
        body.setObjectName('PageBody')
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(4)
        key = self.key
        draft = getattr(self.window, 'draft', None)
        if key == 'session' and draft is not None:
            name = 'New session'
        elif key == 'session' and self.window.folder is not None:
            name = self.window.folder.path.name
        else:
            name = NODE[key].label
        title = QLabel(name)
        title.setObjectName('PageTitle')
        title.setWordWrap(True)
        lay.addWidget(title)
        crumb = QLabel(_crumb(key))
        crumb.setObjectName('PageCrumb')
        lay.addWidget(crumb)
        status, reason = self.window.statuses.get(key, ('off', ''))
        srow = QHBoxLayout()
        si = QLabel()
        si.setPixmap(status_icon(status).pixmap(16, 16))
        srow.addWidget(si, 0, Qt.AlignmentFlag.AlignTop)
        st = QLabel(f"<b>{STATUS_LABELS[status]}</b>" + (f". {reason}" if reason else ''))
        st.setWordWrap(True)
        st.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        st.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        srow.addWidget(st, 1)
        lay.addLayout(srow)
        folder = self.window.folder
        build = getattr(self, f'_page_{key}', None)
        if draft is not None:
            self.window.draft_pages.build(key, lay)
        elif folder is not None and key in ANALYSIS_KEYS:
            self.window.analysis_pages.build(key, folder, lay)
        elif folder is None and key != 'session':
            lay.addWidget(note('Open a session folder to see this.'))
        elif build is not None:
            build(folder, lay)
        lay.addStretch(1)
        self.setWidget(body)

    # -- pages ------------------------------------------------------------------
    def _page_session(self, folder, lay):
        if folder is None:
            sec = Section('New session')
            for kind, ic, text in (('video', 'video', 'From a video...'),
                                   ('image', 'image', 'From an image...'),
                                   ('camera', 'camera', 'From the camera...')):
                b = QPushButton(icon(ic), text)
                b.clicked.connect(lambda _=False, k=kind: self.window.new_session(k))
                sec.widget(b)
            lay.addWidget(sec)
            b = QPushButton(icon('open'), 'Open session folder...')
            b.setObjectName('Primary')
            b.clicked.connect(self.window.open_session_dialog)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
            return
        s = folder.session
        sec = Section('Session')
        sec.field('Folder', str(folder.path))
        sec.field('Created', s.created or '-')
        sec.field('Source', f"{s.source.get('kind', '')}: {s.source.get('name', '')}")
        sec.field('Frames logged', len(folder.log))
        sec.field('Format version', s.schema_version)
        lay.addWidget(sec)
        for p in folder.problems:
            lay.addWidget(callout(p.message))
        b = QPushButton(icon('open'), 'Show folder in Explorer')
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.path))))
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)

    def _children_page(self, key, lay):
        sec = Section('Contents')
        for n in children(key):
            status, reason = self.window.statuses.get(n.key, ('off', ''))
            b = QPushButton(status_icon(status), f"{n.label}")
            b.setToolTip(reason)
            b.setStyleSheet('text-align:left; padding-left:8px;')
            b.clicked.connect(lambda _=False, k=n.key: self.window.select_node(k))
            sec.widget(b)
        lay.addWidget(sec)

    def _page_global(self, folder, lay):
        self._children_page('global', lay)

    def _page_analyses(self, folder, lay):
        self._children_page('analyses', lay)

    def _page_report(self, folder, lay):
        lay.addWidget(note('A printable report of the session and its analyses (settings, '
                           'results, plots) is the next step of the overhaul.'))

    def _page_fluids(self, folder, lay):
        s = folder.session
        f, n = s.fluids, s.needle
        sec = Section('Fluids')
        sec.field('Name', f.get('name') or '-')
        sec.field('Inner density', fmt(f['density_inner'], 6), 'kg/m3')
        sec.field('Outer density', fmt(f['density_outer'], 6), 'kg/m3')
        sec.field('Inner density tolerance', fmt(100 * f.get('rho_in_tol_rel', 0.0005), 3), '%')
        sec.field('Outer density tolerance', fmt(100 * f.get('rho_out_tol_rel', 0.02), 3), '%')
        if f.get('pressure_MPa') is not None:
            sec.field('Pressure', fmt(f['pressure_MPa']), 'MPa')
        if f.get('temperature_C') is not None:
            sec.field('Temperature', fmt(f['temperature_C']), 'C')
        lay.addWidget(sec)
        sec = Section('Needle')
        sec.field('Name', n.get('name') or '-')
        sec.field('Outer diameter', fmt(n['od_mm'], 6), 'mm')
        sec.field('Diameter tolerance', fmt(100 * n.get('tol_rel', 0.01), 3), '%')
        lay.addWidget(sec)
        lay.addWidget(note(LATER_LOGGING))

    def _page_calibration(self, folder, lay):
        sec = Section('Calibration curve')
        p = folder.calibration_path
        if p is not None and p.is_file():
            try:
                from core.calibration import parse_calibration_template
                d = parse_calibration_template(p)
                sec.field('Points', len(d['c']))
                sec.field('Concentration range', f"{fmt(min(d['c']))} - {fmt(max(d['c']))}",
                          d.get('unit') or '')
                sec.field('IFT range', f"{fmt(min(d['gamma']))} - {fmt(max(d['gamma']))}", 'mN/m')
                sec.field('c_sat', fmt(d.get('c_sat')), d.get('unit') or '')
            except Exception as e:                      # shown, not fatal
                sec.widget(callout(f"The stored calibration can't be read: {e}"))
        else:
            sec.widget(note('No calibration stored in this session. Diffusivity (Yang) '
                            'needs one; the other analyses do not.'))
        lay.addWidget(sec)
        b = QPushButton(icon('calcurve'), 'Load calibration file...')
        b.setToolTip('Validate a gamma(c) template and copy it into the session folder')
        b.clicked.connect(self.window.load_calibration)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)

    def _page_source(self, folder, lay):
        src = folder.session.source
        sec = Section('Source')
        sec.field('Kind', src.get('kind', '-'))
        sec.field('Name', src.get('name') or '-')
        if src.get('path'):
            sec.field('Path', src['path'])
        if src.get('fps'):
            sec.field('Frame rate', fmt(src['fps']), 'fps')
        if src.get('n_frames'):
            sec.field('Frames in source', src['n_frames'])
        if src.get('width'):
            sec.field('Frame size', f"{src['width']} x {src.get('height', '?')}", 'px')
        if src.get('kind') == 'video':
            sec.field('Drop formed at', fmt(src.get('t_start_s', 0.0)), 's')
        lay.addWidget(sec)

    def _page_crop(self, folder, lay):
        crops = [a for op, a in folder.session.prep_ops if op == 'crop']
        sec = Section('Crop')
        if crops:
            x, y, w, h = crops[-1]
            sec.field('Top-left corner', f"x {x}, y {y}", 'px')
            sec.field('Size', f"{w} x {h}", 'px')
        else:
            sec.widget(note('No crop: frames are stored whole.'))
        lay.addWidget(sec)
        lay.addWidget(note('Stored frames are already cropped. ' + LATER_LOGGING))

    def _page_logger(self, folder, lay):
        s = folder.session
        if self.window.logging:
            lay.addWidget(callout('Logging is running; press Stop to end it. Frames logged '
                                  'so far are kept.', 'info'))
        sec = Section('Sampling')
        for label, value, unit in sampling_rows(s.source.get('kind'), s.sampling):
            sec.field(label, value, unit)
        sec.field('Frames logged', len(folder.log))
        lay.addWidget(sec)
        lay.addWidget(note('To log more frames or change the sampling, start a new '
                           'session from the same source (File > New session).'))

    def _page_scale(self, folder, lay):
        sc = folder.session.scale
        sec = Section('Scale')
        sec.field('Scale', fmt(sc['pixel_to_m'] * 1e6, 6), 'um/px', result=True)
        sec.field('Method', sc.get('method', '-'))
        if sc.get('reference'):
            sec.field('Reference', sc['reference'])
        lay.addWidget(sec)
        lay.addWidget(note(LATER_LOGGING))

    def _page_baseline(self, folder, lay):
        b = folder.session.baseline
        sec = Section('Baseline')
        sec.field('Row', fmt(b['y_px'], 6), 'px', result=True)
        sec.field('Method', b.get('method', '-'))
        lay.addWidget(sec)
        lay.addWidget(note('The row is in the coordinates of the stored (cropped) frames. '
                           + LATER_LOGGING))

    def _page_log(self, folder, lay):
        log = folder.log
        sec = Section('Log')
        sec.field('Rows', len(log))
        sec.field('Fit converged', sum(1 for r in log.rows if r['fit_ok']))
        sec.field('Passed QC', sum(1 for r in log.rows if r['qc_pass']))
        lay.addWidget(sec)
        for p in folder.problems:
            if p.code != 'calibration_missing':
                lay.addWidget(callout(p.message))

        sec = Section('Columns')
        locked = set(fl.required_columns(folder.session.analyses))
        self.columns_list = QListWidget()
        for c in fl.OPTIONAL:
            it = QListWidgetItem(f"{c.label} ({c.unit})" if c.unit else c.label)
            it.setData(Qt.ItemDataRole.UserRole, c.key)
            on = c.key in log.columns or c.key in locked
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            if c.key in locked:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                it.setToolTip('Needed by an enabled analysis, so always logged.')
            else:
                it.setToolTip(c.description)
            self.columns_list.addItem(it)
        self.columns_list.setMinimumHeight(min(22 * len(fl.OPTIONAL), 240))
        sec.widget(self.columns_list)
        sec.widget(note('Every frame is fitted again from the stored images; the old '
                        'log is replaced only when all frames are done.'))
        lay.addWidget(sec)
        b = QPushButton(icon('rebuild'), 'Rebuild log')
        b.setObjectName('Primary')
        b.setEnabled(bool(folder.frame_indices()) and not self.window.job_running())
        b.clicked.connect(self._rebuild)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)

    def _rebuild(self):
        chosen = [self.columns_list.item(i).data(Qt.ItemDataRole.UserRole)
                  for i in range(self.columns_list.count())
                  if self.columns_list.item(i).checkState() == Qt.CheckState.Checked]
        self.window.rebuild_log(fl.column_keys(
            chosen, analyses=self.window.folder.session.analyses,
            camera=self.window.folder.session.source.get('kind') == 'camera'))
