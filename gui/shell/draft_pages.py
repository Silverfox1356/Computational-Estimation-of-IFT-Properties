"""
gui.shell.draft_pages
=====================

Settings pages for a session being set up (core.session_draft.SessionDraft).
Each builder fills a page for one Analysis Builder node with editable
controls; edits go straight into the draft and the tree updates.  Tools that
need the image (crop box, scale reference, baseline row) ask the window to
start a pick on the Frame tab.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QRadioButton, QSlider, QSpinBox, QWidget)

from core import frame_log as fl
from core import presets as P
from core.session_draft import SECONDS_PER_FRAME, DraftError
from gui.shell.icons import icon
from gui.shell.nodes import children
from gui.shell.pages import Section, callout, note
from gui.shell.panels import fmt


def _spin(value, lo, hi, decimals=3, step=None, suffix=''):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(float(value))
    s.setSingleStep(step or 10 ** -decimals * 10)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    return s


def _row(*widgets, stretch=True):
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    for x in widgets:
        h.addWidget(x)
    if stretch:
        h.addStretch(1)
    return w


def _button(text, ic, slot, primary=False, tip=''):
    b = QPushButton(icon(ic), text)
    if primary:
        b.setObjectName('Primary')
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(slot)
    return b


def _optional_number(text, name):
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise DraftError(f"{name}: '{text}' is not a number") from None


class DraftPages:
    """Builds the draft page for ``key`` into ``lay``.  ``w`` is the window."""

    def __init__(self, window):
        self.w = window

    def build(self, key, lay):
        fn = getattr(self, f'page_{key}', None)
        if fn is not None:
            fn(self.w.draft, lay)
        else:
            lay.addWidget(note('Available once the frames are logged.'))

    def _apply(self, fn, rebuild=False):
        """Run an edit on the draft; report a DraftError instead of raising."""
        try:
            fn()
        except DraftError as e:
            self.w.messages.add(str(e), 'warn')
            self.w.statusBar().showMessage(str(e), 6000)
            self.w.draft_changed(rebuild=True)
            return False
        self.w.draft_changed(rebuild=rebuild)
        return True

    # -- session -----------------------------------------------------------------
    def page_session(self, d, lay):
        sec = Section('New session')
        sec.field('Source', f"{d.kind}: {d.source.get('name', '')}")
        sec.field('Saved', 'Not yet: the folder is created when logging starts')
        lay.addWidget(sec)
        missing = d.missing()
        sec = Section('Still needed' if missing else 'Ready')
        if missing:
            for key, msg in missing:
                b = QPushButton(icon('next'), msg)
                b.setStyleSheet('text-align:left; padding-left:8px;')
                b.clicked.connect(lambda _=False, k=key: self.w.select_node(k))
                sec.widget(b)
        else:
            sec.widget(note('Everything is set. Go to Frame Logger to check the '
                            'sampling and start logging.'))
            sec.widget(_button('Go to Frame Logger', 'next', lambda: self.w.select_node('logger')))
        lay.addWidget(sec)
        lay.addWidget(_button('Discard this session', 'close', self.w.discard_draft))

    def page_global(self, d, lay):
        self.w.settings._children_page('global', lay)

    # -- fluids and needle ------------------------------------------------------------
    def page_fluids(self, d, lay):
        needles, _ = P.list_presets('needle')
        fluids, _ = P.list_presets('fluid')
        n, f = d.needle, d.fluids

        sec = Section('Needle')
        combo_n = QComboBox()
        combo_n.addItem('(custom)', None)
        for p in needles:
            combo_n.addItem(p.name, p)
        i = combo_n.findText(n.get('name') or '')
        combo_n.setCurrentIndex(max(i, 0))
        sec.form.addRow('Preset', combo_n)
        od = _spin(n['od_mm'], 0.001, 100, 4, 0.01)
        tol = _spin(100 * n.get('tol_rel', 0.01), 0, 100, 3, 0.1)
        sec.form.addRow('Outer diameter (mm)', od)
        sec.form.addRow('Tolerance (%)', tol)
        lay.addWidget(sec)

        def set_needle(name=None):
            self._apply(lambda: d.set_needle(od.value(), tol.value() / 100.0,
                                             name or '(custom)'))
        od.valueChanged.connect(lambda _: set_needle())
        tol.valueChanged.connect(lambda _: set_needle())

        def pick_needle(_i):
            p = combo_n.currentData()
            if p is None:
                return
            for s, v in ((od, p.values['od_mm']), (tol, p.values['od_tol_pct'])):
                s.blockSignals(True)
                s.setValue(float(v))
                s.blockSignals(False)
            set_needle(p.name)
        combo_n.currentIndexChanged.connect(pick_needle)

        sec = Section('Fluids')
        combo_f = QComboBox()
        combo_f.addItem('(custom)', None)
        for p in fluids:
            combo_f.addItem(p.name, p)
        i = combo_f.findText(f.get('name') or '')
        combo_f.setCurrentIndex(max(i, 0))
        sec.form.addRow('Preset', combo_f)
        rin = _spin(f['density_inner'], 0.001, 30000, 3, 1)
        rout = _spin(f['density_outer'], 0.001, 30000, 3, 1)
        tin = _spin(100 * f.get('rho_in_tol_rel', 0.0005), 0, 100, 4, 0.01)
        tout = _spin(100 * f.get('rho_out_tol_rel', 0.02), 0, 100, 4, 0.1)
        pres = QLineEdit('' if f.get('pressure_MPa') is None else f"{f['pressure_MPa']:g}")
        temp = QLineEdit('' if f.get('temperature_C') is None else f"{f['temperature_C']:g}")
        pres.setPlaceholderText('optional')
        temp.setPlaceholderText('optional')
        for label, wdg in (('Drop density (kg/m3)', rin), ('Surrounding density (kg/m3)', rout),
                           ('Drop density tolerance (%)', tin),
                           ('Surrounding tolerance (%)', tout),
                           ('Pressure (MPa)', pres), ('Temperature (C)', temp)):
            sec.form.addRow(label, wdg)
        lay.addWidget(sec)

        def set_fluids(name=None):
            def go():
                d.set_fluids(density_inner=rin.value(), density_outer=rout.value(),
                             rho_in_tol_rel=tin.value() / 100.0,
                             rho_out_tol_rel=tout.value() / 100.0,
                             pressure_MPa=_optional_number(pres.text(), 'Pressure'),
                             temperature_C=_optional_number(temp.text(), 'Temperature'),
                             name=name or '(custom)')
            self._apply(go)
        for s in (rin, rout, tin, tout):
            s.valueChanged.connect(lambda _: set_fluids())
        pres.editingFinished.connect(set_fluids)
        temp.editingFinished.connect(set_fluids)

        def pick_fluid(_i):
            p = combo_f.currentData()
            if p is None:
                return
            v = p.values
            for s, x in ((rin, v['rho_in']), (rout, v['rho_out']),
                         (tin, v['rho_in_tol_pct']), (tout, v['rho_out_tol_pct'])):
                s.blockSignals(True)
                s.setValue(float(x))
                s.blockSignals(False)
            pres.setText('' if v.get('P_MPa') is None else f"{v['P_MPa']:g}")
            temp.setText('' if v.get('T_C') is None else f"{v['T_C']:g}")
            set_fluids(p.name)
        combo_f.currentIndexChanged.connect(pick_fluid)
        lay.addWidget(note('Presets are managed in the Pendant window\'s Settings. The '
                           'values are stored in the session when logging starts.'))

    def page_calibration(self, d, lay):
        sec = Section('Calibration curve')
        if d.calibration_path:
            sec.field('File', d.calibration_path)
            sec.widget(note('Copied into the session folder when logging starts.'))
        else:
            sec.widget(note('Optional. Diffusivity (Yang) needs a gamma(c) curve; it can '
                            'also be loaded after logging.'))
        lay.addWidget(sec)
        row = [_button('Load calibration file...', 'calcurve', self.w.load_calibration)]
        if d.calibration_path:
            row.append(_button('Remove', 'close', lambda: self._apply(
                lambda: setattr(d, 'calibration_path', None), rebuild=True)))
        lay.addWidget(_row(*row))

    # -- source and preparation ---------------------------------------------------------
    def page_source(self, d, lay):
        s = d.source
        sec = Section('Source')
        sec.field('Kind', d.kind)
        sec.field('Name', s.get('name', ''))
        if s.get('path'):
            sec.field('Path', s['path'])
        if s.get('width'):
            sec.field('Frame size', f"{s['width']} x {s.get('height')}", 'px')
        if d.kind == 'video':
            sec.field('Frame rate', fmt(s['fps']), 'fps')
            sec.field('Length', f"{fmt(s['duration_s'])} s, {s['n_frames']} frames")
        lay.addWidget(sec)
        if d.kind == 'video':
            sec = Section('Drop formed at')
            t0 = _spin(d.sampling.get('t_start_s', 0.0), 0, max(s['duration_s'] - 1e-3, 0), 3,
                       1.0 / s['fps'], ' s')
            sec.form.addRow('Time in the video', t0)
            sec.widget(note('Frames before this are not logged; times in the log are '
                            'counted from here. The frame at this time is the one you '
                            'prepare on.'))
            t0.valueChanged.connect(lambda v: self._apply(lambda: d.set_t_start(v)))
            lay.addWidget(sec)
        elif d.kind == 'camera':
            sec = Section('Reference frame')
            sec.widget(note('The live picture shows on the Frame tab. Capture the frame '
                            'to prepare on (crop, scale, baseline); logging uses the '
                            'same settings for every frame.'))
            sec.widget(_button('Capture reference frame', 'camera', self.w.capture_reference,
                               primary=True))
            lay.addWidget(sec)

    def page_crop(self, d, lay):
        sec = Section('Crop')
        box = d.crop_box()
        sec.field('Region', f"x {box[0]}, y {box[1]}, {box[2]} x {box[3]} px" if box
                  else 'Whole frame')
        sec.widget(note('Crop to the drop and needle: smaller frames are faster to fit '
                        'and to store. Changing the crop clears the baseline.'))
        lay.addWidget(sec)
        row = [_button('Draw crop box', 'crop', lambda: self.w.pick(
            'rect', 'Drag a box around the needle and drop.',
            lambda x, y, w, h: self._apply(lambda: d.set_crop(x, y, w, h), rebuild=True)),
            primary=True)]
        if box:
            row.append(_button('Remove crop', 'close',
                               lambda: self._apply(d.clear_crop, rebuild=True)))
        lay.addWidget(_row(*row))

    def page_threshold(self, d, lay):
        sec = Section('Threshold')
        level = d.threshold()
        auto = QRadioButton('Automatic (Otsu) on every frame')
        fixed = QRadioButton('Fixed level')
        grp = QButtonGroup(sec)
        grp.addButton(auto)
        grp.addButton(fixed)
        (auto if level is None else fixed).setChecked(True)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        spin = QSpinBox()
        spin.setRange(0, 255)
        for s in (slider, spin):
            s.setValue(128 if level is None else level)
            s.setEnabled(level is not None)
        sec.widget(auto)
        sec.widget(fixed)
        sec.widget(_row(slider, spin, stretch=False))
        sec.widget(note('Frames are stored before the threshold, so it can be changed '
                        'later and the log rebuilt. Tick "Show thresholded" on the Frame '
                        'tab to see what the fit sees.'))
        lay.addWidget(sec)

        def apply_level():
            lvl = None if auto.isChecked() else spin.value()
            slider.setEnabled(lvl is not None)
            spin.setEnabled(lvl is not None)
            self.w.graphics.frames.chk_prepared.setChecked(lvl is not None)
            self._apply(lambda: d.set_threshold(lvl))
            self.w.graphics.frames.redraw()
        slider.valueChanged.connect(lambda v: (spin.blockSignals(True), spin.setValue(v),
                                               spin.blockSignals(False), apply_level()))
        spin.valueChanged.connect(lambda v: (slider.blockSignals(True), slider.setValue(v),
                                             slider.blockSignals(False), apply_level()))
        auto.toggled.connect(lambda _: apply_level())

    # -- scale and baseline --------------------------------------------------------------
    def page_scale(self, d, lay):
        sec = Section('Scale')
        if d.scale:
            sec.field('Scale', fmt(d.scale['pixel_to_m'] * 1e6, 6), 'um/px', result=True)
            sec.field('From', f"{d.scale.get('method')}: {d.scale.get('reference')}")
        else:
            sec.widget(note('Not calibrated yet.'))
        lay.addWidget(sec)

        sec = Section('Calibrate')
        method = QComboBox()
        method.addItem('Needle width: draw a box around the needle', 'needle')
        method.addItem('Known length: click its two ends', 'points')
        method.addItem('Enter the scale', 'direct')
        method.setCurrentIndex(max(method.findData(self.w.scale_method), 0))
        sec.form.addRow('Method', method)
        size = _spin(d.needle['od_mm'], 0.0001, 1000, 4, 0.01, ' mm')
        size.setToolTip('Real size of the reference; the needle diameter by default')
        sec.form.addRow('Reference size', size)
        direct = _spin((d.scale or {}).get('pixel_to_m', 1e-5) * 1e6, 0.001, 10000, 4, 0.1,
                       ' um/px')
        sec.form.addRow('Scale', direct)
        lay.addWidget(sec)

        def show_rows(_=None):
            m = method.currentData()
            self.w.scale_method = m
            sec.form.setRowVisible(size, m != 'direct')
            sec.form.setRowVisible(direct, m == 'direct')
            go.setText({'needle': 'Draw box on needle', 'points': 'Click two points',
                        'direct': 'Apply scale'}[m])

        def go_clicked():
            m = method.currentData()
            if m == 'direct':
                self._apply(lambda: d.set_scale_direct(direct.value()), rebuild=True)
            elif m == 'needle':
                def boxed(x, y, w, h):
                    def do():
                        px, marker = d.measure_box(x, y, w, h, 'needle')
                        self.w.graphics.frames.scale_marker = marker
                        d.set_scale_from_length(px, size.value(), 'needle width',
                                                f"{px:.1f} px = {size.value():g} mm")
                    self._apply(do, rebuild=True)
                self.w.pick('rect', 'Drag a tight box around a straight part of the needle.',
                            boxed)
            else:
                def clicked(x1, y1, x2, y2):
                    import math
                    px = math.hypot(x2 - x1, y2 - y1)

                    def do():
                        d.set_scale_from_length(px, size.value(), 'two points',
                                                f"{px:.1f} px = {size.value():g} mm")
                        self.w.graphics.frames.scale_marker = ('line', (x1, y1), (x2, y2))
                    self._apply(do, rebuild=True)
                self.w.pick('points2', 'Click the two ends of the known length.', clicked)

        go = _button('Apply', 'ruler', go_clicked, primary=True)
        method.currentIndexChanged.connect(show_rows)
        show_rows()
        lay.addWidget(_row(go))

    def page_baseline(self, d, lay):
        sec = Section('Baseline')
        if d.baseline:
            sec.field('Row', fmt(d.baseline['y_px'], 6), 'px', result=True)
            sec.field('Method', d.baseline.get('method'))
        else:
            sec.widget(note('Not set yet.'))
        sec.widget(note('The row where the drop leaves the needle. Only the outline below '
                        'it is fitted. Automatic search looks for where the outline first '
                        'widens beyond the needle.'))
        lay.addWidget(sec)
        lay.addWidget(_row(
            _button('Find automatically', 'baseline',
                    lambda: self._apply(d.auto_baseline, rebuild=True), primary=True),
            _button('Click on the image', 'baseline', lambda: self.w.pick(
                'hline', 'Click the row where the drop leaves the needle.',
                lambda y: self._apply(lambda: d.set_baseline(y, 'manual'), rebuild=True)))))

    # -- logger -------------------------------------------------------------------------
    def page_logger(self, d, lay):
        s = d.sampling
        plan_label = QLabel('')
        plan_label.setWordWrap(True)
        plan_label.setObjectName('Note')

        def show_plan():
            try:
                p = d.sample_plan()
            except DraftError as e:
                plan_label.setText(f"Check the sampling: {e}")
                return
            if d.kind == 'video':
                mins = max(1, round(p['n'] * SECONDS_PER_FRAME / 60))
                plan_label.setText(
                    f"{p['n']} of {d.source['n_frames']} frames will be logged (t = 0 to "
                    f"{p['t_last']:.1f} s; steps {p['min_step']:.3g} to {p['max_step']:.3g} s). "
                    f"Roughly {mins} min, in the background.")
            elif d.kind == 'camera':
                plan_label.setText(
                    f"A frame every {p['interval_s']:g} s" +
                    (f" for {p['duration_s'] / 60:g} min ({p['n']} frames)." if p['duration_s']
                     else " until you press Stop.") +
                    " If a fit takes longer than the interval, the next frame is taken "
                    "as soon as it finishes; the log records the real times.")
            else:
                plan_label.setText('The image is logged as one frame at t = 0.')

        def changed():
            self._apply(lambda: None)
            show_plan()

        sec = Section('Sampling')
        if d.kind == 'video':
            mode = QComboBox()
            mode.addItem('Dense early, then sparser', 'dense_early')
            mode.addItem('One frame every N seconds', 'interval')
            mode.setCurrentIndex(max(mode.findData(s.get('mode')), 0))
            n = QSpinBox()
            n.setRange(2, 5000)
            n.setValue(int(s.get('n_frames') or 120))
            first = QLineEdit('' if not s.get('first_step_s') else f"{s['first_step_s']:g}")
            first.setPlaceholderText('automatic')
            iv = _spin(s.get('interval_s') or 5.0, 0.001, 1e6, 3, 1, ' s')
            until = QLineEdit('' if s.get('t_end_s') is None else f"{s['t_end_s']:g}")
            until.setPlaceholderText('end of the video')
            sec.form.addRow('Plan', mode)
            sec.form.addRow('Frames', n)
            sec.form.addRow('First step (s)', first)
            sec.form.addRow('Interval', iv)
            sec.form.addRow('Log until (s)', until)

            def rows():
                dense = mode.currentData() == 'dense_early'
                sec.form.setRowVisible(n, dense)
                sec.form.setRowVisible(first, dense)
                sec.form.setRowVisible(iv, not dense)

            def read():
                try:
                    s['mode'] = mode.currentData()
                    s['n_frames'] = n.value()
                    s['first_step_s'] = _optional_number(first.text(), 'First step')
                    s['interval_s'] = iv.value()
                    s['t_end_s'] = _optional_number(until.text(), 'Log until')
                except DraftError as e:
                    self.w.messages.add(str(e), 'warn')
                rows()
                changed()
            mode.currentIndexChanged.connect(lambda _: read())
            n.valueChanged.connect(lambda _: read())
            iv.valueChanged.connect(lambda _: read())
            first.editingFinished.connect(read)
            until.editingFinished.connect(read)
            rows()
        elif d.kind == 'camera':
            iv = _spin(s.get('interval_s') or 1.0, 0.05, 3600, 2, 0.5, ' s')
            dur = _spin((s.get('duration_s') or 0) / 60.0, 0, 100000, 1, 1, ' min')
            dur.setSpecialValueText('until stopped')
            sec.form.addRow('Interval', iv)
            sec.form.addRow('Stop after', dur)

            def read_cam():
                s['interval_s'] = iv.value()
                s['duration_s'] = dur.value() * 60.0 or None
                changed()
            iv.valueChanged.connect(lambda _: read_cam())
            dur.valueChanged.connect(lambda _: read_cam())
        sec.widget(plan_label)
        lay.addWidget(sec)
        show_plan()

        sec = Section('Columns to log')
        locked = set(fl.required_columns(d.analyses))
        lst = QListWidget()
        for c in fl.OPTIONAL:
            it = QListWidgetItem(f"{c.label} ({c.unit})" if c.unit else c.label)
            it.setData(Qt.ItemDataRole.UserRole, c.key)
            on = c.key in d.columns or c.key in locked
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            if c.key in locked:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                it.setToolTip('Needed by the swelling and diffusivity analyses, so always logged.')
            else:
                it.setToolTip(c.description)
            lst.addItem(it)
        lst.setMinimumHeight(min(22 * len(fl.OPTIONAL), 200))

        def cols_changed(_item):
            d.columns = [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())
                         if lst.item(i).checkState() == Qt.CheckState.Checked]
        lst.itemChanged.connect(cols_changed)
        sec.widget(lst)
        sec.widget(note('One fit gives every column, so leaving one out saves almost no '
                        'time. A column left out can be added later by rebuilding the log.'))
        lay.addWidget(sec)

        sec = Section('Check on the reference frame')
        if d.preview and d.preview.get('fit_ok'):
            r = d.preview
            sec.field('IFT', f"{fmt(r['ift_mNm'])} +/- {fmt(r['ift_unc_mNm'], 2)}", 'mN/m',
                      result=True)
            sec.field('Shape factor beta', fmt(r['beta']))
            sec.field('Apex radius R0', fmt(r['R0_mm']), 'mm')
            sec.field('Volume', fmt(r['volume_uL']), 'uL')
            sec.field('Quality checks', 'passed' if r['qc_pass'] else 'failed (see Messages)')
        elif d.preview:
            sec.widget(callout(f"The fit failed: {d.preview.get('error', '')}"))
        else:
            sec.widget(note('Fit the reference frame once to see that the settings work '
                            '(this is what the old Static mode did).'))
        sec.widget(_button('Measure reference frame', 'drop', self.w.measure_reference))
        lay.addWidget(sec)

        missing = d.missing()
        if missing:
            lay.addWidget(callout('Before logging: ' + ' '.join(m for _, m in missing)))
        start = _button('Start logging...', 'compute', self.w.start_logging, primary=True,
                        tip='Choose where the session folder goes, then log the frames')
        start.setEnabled(not missing and not self.w.job_running())
        lay.addWidget(_row(start))

    # -- everything after logging --------------------------------------------------------
    def page_analyses(self, d, lay):
        lay.addWidget(note('The analyses run on the logged frames. Log the frames first.'))
        self.w.settings._children_page('analyses', lay)
