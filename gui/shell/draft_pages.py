"""
gui.shell.draft_pages
=====================

Settings pages for a session being set up (core.session_draft.SessionDraft).
Each builder fills a page for one Analysis Builder node with editable
controls; most edits go straight into the draft and the tree updates (the
Fluids & Needle values wait for Apply).  Tools that need the image (crop
box, scale reference, baseline row) ask the window to start a pick on the
Frame tab.  What each page does is explained in the Guide (gui.shell.guide),
so the pages hold controls and results only.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                             QSizePolicy, QSlider, QSpinBox)

from core import frame_log as fl
from core import presets as P
from core.session_draft import SECONDS_PER_FRAME, DraftError
from gui.shell.pages import Section, actions, flow_row, ift_result_section, note
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


def id_spin(id_mm):
    """Needle inner diameter in mm; 0 shows 'automatic' (70 % of the outer
    diameter, as core.domain_builder assumes)."""
    s = _spin(id_mm or 0.0, 0.0, 100, 4, 0.01)
    s.setSpecialValueText('automatic (70 % of OD)')
    s.setToolTip('Used only by the diffusion model (Yang, simulation)')
    return s


def id_value(spin):
    return spin.value() or None


def _row(*widgets, uniform=True):
    """Widgets side by side, wrapping onto a new line when the panel is
    narrow; buttons get one width."""
    return flow_row(*widgets, uniform=uniform)


def _button(text, slot, primary=False, tip=''):
    """A page button: text only, so every button on a page looks alike."""
    b = QPushButton(text)
    if primary:
        b.setObjectName('Primary')
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(lambda: slot())        # not slot(checked): see StudioWindow._act
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
        sec.field('Saved', 'when logging starts')
        lay.addWidget(sec)
        missing = d.missing()
        sec = Section('Still needed' if missing else 'Ready')
        if missing:
            for key, msg in missing:
                b = QPushButton(msg)
                b.setStyleSheet('text-align:left; padding-left:8px;')
                b.setToolTip(msg)
                b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                b.clicked.connect(lambda _=False, k=key: self.w.select_node(k))
                sec.widget(b)
        else:
            sec.widget(note('Everything is set.'))
        lay.addWidget(sec)
        buttons = [_button('Discard', self.w.discard_draft, tip='Discard this session')]
        if not missing:
            buttons.append(_button('Frame Logger', lambda: self.w.select_node('logger'),
                                   primary=True))
        lay.addWidget(actions(*buttons))

    def page_global(self, d, lay):
        self.w.settings._children_page('global', lay)

    # -- fluids and needle ------------------------------------------------------------
    def page_fluids(self, d, lay):
        """The values wait for Apply: picking a preset or editing a box
        changes nothing in the draft until then."""
        needles, _ = P.list_presets('needle')
        fluids, _ = P.list_presets('fluid')
        n, f = d.needle, d.fluids
        names = dict(needle=n.get('name'), fluid=f.get('name'))

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
        idd = id_spin(n.get('id_mm'))
        sec.form.addRow('Outer diameter (mm)', od)
        sec.form.addRow('Tolerance (%)', tol)
        sec.form.addRow('Inner diameter (mm)', idd)
        lay.addWidget(sec)
        needle_sec = sec

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

        pending = note('')
        pending.setVisible(False)

        def go():
            d.set_needle(od.value(), tol.value() / 100.0, names['needle'] or '(custom)',
                         id_mm=id_value(idd))
            d.set_fluids(density_inner=rin.value(), density_outer=rout.value(),
                         rho_in_tol_rel=tin.value() / 100.0,
                         rho_out_tol_rel=tout.value() / 100.0,
                         pressure_MPa=_optional_number(pres.text(), 'Pressure'),
                         temperature_C=_optional_number(temp.text(), 'Temperature'),
                         name=names['fluid'] or '(custom)')

        def apply():
            if self._apply(go, rebuild=True):
                self.w.messages.add('Fluids and needle applied.', 'ok')
        apply_b = _button('Apply', apply, primary=True, tip='Use these values for the session')
        reset_b = _button('Reset', lambda: self.w.draft_changed(rebuild=True),
                          tip='Back to the values in use')

        def edited(kind=None):
            if kind is not None:
                names[kind] = None               # edited by hand: no longer the preset
            pending.setText('Not applied yet.')
            pending.setVisible(True)
            apply_b.setEnabled(True)
            reset_b.setEnabled(True)
        apply_b.setEnabled(False)
        reset_b.setEnabled(False)
        for wdg in (od, tol, idd):
            wdg.valueChanged.connect(lambda _: edited('needle'))
        for wdg in (rin, rout, tin, tout):
            wdg.valueChanged.connect(lambda _: edited('fluid'))
        pres.textEdited.connect(lambda _: edited('fluid'))
        temp.textEdited.connect(lambda _: edited('fluid'))

        def fill(boxes, name, kind):
            for s, v in boxes:
                s.blockSignals(True)
                s.setValue(float(v))
                s.blockSignals(False)
            names[kind] = name
            edited()

        def pick_needle(_i):
            p = combo_n.currentData()
            if p is not None:
                fill(((od, p.values['od_mm']), (tol, p.values['od_tol_pct']),
                      (idd, p.values.get('id_mm') or 0.0)), p.name, 'needle')
        combo_n.currentIndexChanged.connect(pick_needle)

        def pick_fluid(_i):
            p = combo_f.currentData()
            if p is None:
                return
            v = p.values
            pres.setText('' if v.get('P_MPa') is None else f"{v['P_MPa']:g}")
            temp.setText('' if v.get('T_C') is None else f"{v['T_C']:g}")
            fill(((rin, v['rho_in']), (rout, v['rho_out']), (tin, v['rho_in_tol_pct']),
                  (tout, v['rho_out_tol_pct'])), p.name, 'fluid')
        combo_f.currentIndexChanged.connect(pick_fluid)

        # Saving a preset applies its values too (the page is rebuilt from the draft).
        needle_sec.widget(self._preset_buttons(
            'needle', combo_n, lambda: dict(od_mm=od.value(), id_mm=id_value(idd),
                                            od_tol_pct=tol.value()), go))
        sec.widget(self._preset_buttons(
            'fluid', combo_f,
            lambda: dict(rho_in=rin.value(), rho_out=rout.value(), rho_in_tol_pct=tin.value(),
                         rho_out_tol_pct=tout.value(),
                         P_MPa=_optional_number(pres.text(), 'Pressure'),
                         T_C=_optional_number(temp.text(), 'Temperature')), go))
        lay.addWidget(pending)
        lay.addWidget(actions(reset_b, apply_b))

    def _preset_buttons(self, kind, combo, values, commit):
        """Save / Save as / Delete / Import for the presets of ``kind``
        (the Pendant window's Settings dialog).  ``commit`` applies the
        page's values to the draft before a save."""
        def chosen():
            return combo.currentData()

        def save(preset):
            try:
                v = values()
                commit()
            except DraftError as e:
                self.w.messages.add(str(e), 'warn')
                return
            self.w.preset_save(kind, v, preset)

        sv = _button('Save', lambda: save(chosen()),
                     tip='Save these values into the chosen preset (not for built-in presets)')
        sa = _button('Save as...', lambda: save(None), tip='Save these values as a new preset')
        de = _button('Delete', lambda: self.w.preset_delete(kind, chosen()),
                     tip='Delete the chosen preset (not for built-in presets)')
        im = _button('Import...', self.w.preset_import,
                     tip='Add a preset file (.json) to your presets')

        def update(_=None):
            p = chosen()
            own = p is not None and not p.builtin
            sv.setEnabled(own)
            de.setEnabled(own)
        combo.currentIndexChanged.connect(update)
        update()
        return _row(sv, sa, de, im)

    def page_calibration(self, d, lay):
        sec = Section('Calibration curve')
        sec.field('File', Path(d.calibration_path).name if d.calibration_path else 'not loaded')
        lay.addWidget(sec)
        row = [_button('Load...', self.w.load_calibration,
                       tip='Load a gamma(c) calibration file')]
        if d.calibration_path:
            row.insert(0, _button('Remove', lambda: self._apply(
                lambda: setattr(d, 'calibration_path', None), rebuild=True)))
        lay.addWidget(actions(*row))

    # -- Yang from a single frame --------------------------------------------------------
    def page_yang(self, d, lay):
        """Single frame: this picture's outline is the model geometry; the
        gamma(c) calibration and IFT(t) come from CSV files (as the Pendant
        window's Static mode with Upload calibration and Load IFT curve)."""
        if d.kind != 'image':
            lay.addWidget(note('Available once the frames are logged.'))
            return
        from core.ift_data import IFTDataError, load_ift_curve
        from gui.shell.analysis_pages import yang_search
        missing = dict(d.missing())
        sec = Section('Inputs (single frame)')
        pic = [k for k in ('scale', 'baseline') if k in missing]
        sec.field('Picture', None if pic else 'ready',
                  buttons=[_button(f"Set {k}", lambda k=k: self.w.select_node(k),
                                   tip=f"Go to {k.capitalize()}") for k in pic])
        sec.field('gamma(c)', Path(d.calibration_path).name if d.calibration_path
                  else 'not loaded',
                  buttons=[_button('Load...', self.w.load_calibration,
                                   tip='Load the gamma(c) calibration')])
        if d.ift_curve_path:
            try:
                c = load_ift_curve(d.ift_curve_path)
                info = (f"{Path(d.ift_curve_path).name}: {len(c['t'])} points, "
                        f"{fmt(c['t'][0])} to {fmt(c['t'][-1])} s")
            except (IFTDataError, OSError) as e:
                info = f"can't be read: {e}"
        else:
            info = 'not loaded'
        sec.field('IFT(t)', info, buttons=[_button('Load...', self.w.load_ift_curve,
                                                   tip='Load IFT against time (CSV)')])
        sec.field('Templates', None, buttons=[
            _button('gamma(c)', lambda: self.w.save_curve_template('calibration'),
                    tip='Save a gamma(c) template file'),
            _button('IFT(t)', lambda: self.w.save_curve_template('ift'),
                    tip='Save an IFT-against-time template file')])
        lay.addWidget(sec)
        search = yang_search({}, lay)
        ready = not pic and bool(d.calibration_path and d.ift_curve_path)

        def go():
            try:
                kw = search()
            except ValueError as e:
                self.w.messages.add(str(e), 'warn')
                return
            self.w.continue_single_frame(kw)
        b = _button('Run...', go, primary=True,
                    tip='Saves the session (asks where), then builds the model from this '
                        'picture and fits D and k to the IFT curve')
        b.setEnabled(ready and not self.w.job_running())
        lay.addWidget(actions(b))

    # -- single-frame IFT ----------------------------------------------------------------
    def page_frame_ift(self, d, lay):
        ready = d.scale is not None and d.baseline is not None
        b = _button('Measure IFT', self.w.measure_reference, primary=True,
                    tip='Fit the drop shape of the reference frame')
        b.setEnabled(ready and not self.w.job_running())
        lay.addWidget(actions(b))
        if d.preview:
            lay.addWidget(ift_result_section(d.preview, 'Result: reference frame'))

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
            t0.setToolTip('Frames before this are not logged; times count from here')
            sec.form.addRow('Time in the video', t0)
            t0.valueChanged.connect(lambda v: self._apply(lambda: d.set_t_start(v)))
            lay.addWidget(sec)
        elif d.kind == 'camera':
            lay.addWidget(actions(_button('Capture reference frame', self.w.capture_reference,
                                          primary=True)))

    def page_crop(self, d, lay):
        sec = Section('Crop')
        box = d.crop_box()
        sec.field('Region', f"x {box[0]}, y {box[1]}, {box[2]} x {box[3]} px" if box
                  else 'whole frame')
        lay.addWidget(sec)
        row = [_button('Draw crop box', lambda: self.w.pick(
            'rect', 'Drag a box around the needle and drop.',
            lambda x, y, w, h: self._apply(lambda: d.set_crop(x, y, w, h), rebuild=True)),
            primary=True, tip='Changing the crop clears the baseline')]
        if box:
            row.insert(0, _button('Remove crop', lambda: self._apply(d.clear_crop, rebuild=True)))
        lay.addWidget(actions(*row))

    def page_threshold(self, d, lay):
        sec = Section('Threshold')
        level = d.threshold()
        auto = QRadioButton('Automatic (Otsu)')
        auto.setToolTip("Otsu's threshold, computed again on every frame")
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
        sec.widget(_row(slider, spin, uniform=False))
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
            sec.field('Scale', 'not set')
        lay.addWidget(sec)

        sec = Section('Calibrate')
        method = QComboBox()
        method.addItem('Needle width', 'needle')
        method.addItem('Reference sphere', 'sphere')
        method.addItem('Known length', 'points')
        method.addItem('Enter the scale', 'direct')
        method.setCurrentIndex(max(method.findData(self.w.scale_method), 0))
        sec.form.addRow('Method', method)
        size = _spin(d.needle['od_mm'], 0.0001, 1000, 4, 0.01, ' mm')
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
            go.setText({'needle': 'Draw box on needle', 'sphere': 'Draw box on sphere',
                        'points': 'Click two points', 'direct': 'Apply scale'}[m])
            size.setToolTip('Real diameter of the sphere' if m == 'sphere' else
                            'Real size of the reference; the needle diameter by default')

        def go_clicked():
            m = method.currentData()
            if m == 'direct':
                self._apply(lambda: d.set_scale_direct(direct.value()), rebuild=True)
            elif m in ('needle', 'sphere'):
                what = 'needle width' if m == 'needle' else 'sphere diameter'

                def boxed(x, y, w, h):
                    def do():
                        px, marker = d.measure_box(x, y, w, h, m)
                        self.w.graphics.frames.scale_marker = marker
                        d.set_scale_from_length(px, size.value(), what,
                                                f"{px:.1f} px = {size.value():g} mm")
                    self._apply(do, rebuild=True)
                self.w.pick('rect', 'Drag a tight box around a straight part of the needle.'
                            if m == 'needle' else 'Drag a tight box around the sphere.', boxed)
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

        go = _button('Apply', go_clicked, primary=True)
        method.currentIndexChanged.connect(show_rows)
        show_rows()
        lay.addWidget(actions(go))

    def page_baseline(self, d, lay):
        sec = Section('Baseline')
        if d.baseline:
            sec.field('Row', fmt(d.baseline['y_px'], 6), 'px', result=True)
            sec.field('Method', d.baseline.get('method'))
        else:
            sec.field('Row', 'not set')
        lay.addWidget(sec)
        lay.addWidget(actions(
            _button('Click on the image', lambda: self.w.pick(
                'hline', 'Click the row where the drop leaves the needle.',
                lambda y: self._apply(lambda: d.set_baseline(y, 'manual'), rebuild=True))),
            _button('Find automatically', lambda: self._apply(d.auto_baseline, rebuild=True),
                    primary=True)))

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
                    f"{p['n']} of {d.source['n_frames']} frames, t = 0 to {p['t_last']:.1f} s "
                    f"(steps {p['min_step']:.3g} to {p['max_step']:.3g} s), about {mins} min.")
            elif d.kind == 'camera':
                plan_label.setText(
                    f"A frame every {p['interval_s']:g} s" +
                    (f" for {p['duration_s'] / 60:g} min ({p['n']} frames)." if p['duration_s']
                     else " until stopped."))
            else:
                plan_label.setText('One frame at t = 0.')

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
            sec.field('Fit', f"failed: {d.preview.get('error', '')}")
        else:
            sec.field('Fit', 'not measured yet')
        sec.widget(_row(_button('Measure reference frame', self.w.measure_reference,
                                tip='Fit the reference frame once to check the set-up')))
        lay.addWidget(sec)

        missing = d.missing()
        if missing:
            lay.addWidget(note('Before logging: ' + ' '.join(m for _, m in missing)))
        start = _button('Start logging...', self.w.start_logging, primary=True,
                        tip='Choose where the session folder goes, then log the frames')
        start.setEnabled(not missing and not self.w.job_running())
        lay.addWidget(actions(start))

    # -- everything after logging --------------------------------------------------------
    def page_analyses(self, d, lay):
        lay.addWidget(note('Available once the frames are logged.'))
        self.w.settings._children_page('analyses', lay)
