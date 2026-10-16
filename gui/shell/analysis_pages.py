"""
gui.shell.analysis_pages
========================

Settings pages for the analyses of an open session (core.analyses), for
measuring one frame again (Single-Frame IFT) and for changing the threshold
of its frames: settings, a Run button (bottom-right), the result, and its
warnings as a list ("No warnings" when there are none).  Runs go through the
window's job runner, so they show progress, can be stopped, and refresh the
tree and the result tabs when they finish.  What each analysis does is in
the Guide (gui.shell.guide).
"""

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
                             QLineEdit, QPushButton, QRadioButton, QSpinBox)

from core import frame_log as fl
from core.analyses import SWELLING_DEFAULT_THRESHOLD
from core.session import YANG_MIN_POINTS, SessionError
from gui.shell.draft_pages import _button, _row, _spin
from gui.shell.pages import Section, actions, ift_result_section, note, warnings_section
from gui.shell.panels import fmt

KEYS = ('swelling', 'bagalkot', 'yang', 'yang_model', 'simulation', 'results', 'threshold',
        'frame_ift')


def _sci(text, name):
    try:
        v = float(text.strip().replace(',', '.'))
    except ValueError:
        raise ValueError(f"{name}: '{text}' is not a number") from None
    if not (v > 0 and math.isfinite(v)):
        raise ValueError(f"{name} must be a positive number")
    return v


def _sci_edit(value):
    e = QLineEdit(f"{value:.3g}")
    e.setToolTip('A number such as 1e-9')
    return e


def _parse_list(text, name):
    out = []
    for part in text.replace(';', ',').split(','):
        if part.strip():
            out.append(_sci(part, name))
    return out


SWELLING_VERDICTS = dict(negligible='negligible', significant='significant',
                         not_measurable='not measurable', shrinking='shrinking',
                         insufficient='too few frames')


def result_warnings(folder, name, doc):
    """The warnings of a stored result, as sentences (empty: none)."""
    if doc is None:
        return []
    if 'error' in doc:
        return [f"The stored result can't be read: {doc['error']}"]
    out = []
    why = folder.out_of_date_reason(name)
    if why:
        out.append(f"{why} This result is out of date; run it again.")
    r = doc['data']['result']
    if name == 'swelling':
        if r.get('verdict') in ('not_measurable', 'shrinking', 'insufficient'):
            out.append(r.get('message', ''))
        elif r.get('still_rising'):
            out.append('The volume is still rising at the last frame, so the swelling is a '
                       'lower bound.')
    elif name == 'bagalkot':
        if r.get('at_edge'):
            out.append('The best D is at the edge of the search range; widen it and run again.')
    elif name == 'yang':
        if not r.get('D_well_determined'):
            out.append('D is not well determined: equally good fits span a wide range of D.')
        if r.get('message'):
            m = r['message']
            out.append(m[:1].upper() + m[1:])
    elif name == 'simulation':
        if r.get('has_nan'):
            out.append('The concentration has NaN values: the run is not usable. Use '
                       'automatic or smaller time steps.')
        if r.get('has_negative'):
            out.append('The final concentration field has negative values.')
    return [w for w in out if w]


def yang_search(st, lay):
    """The search settings of Yang's fit (on the session's and on the single
    frame's Diffusivity (Yang) page).  Returns a function giving run_yang's
    keyword arguments, or raising ValueError with a message for the user.

        kD grid     Yang's Fig 7a: an E-vs-D curve for each of your kD; the
                    exact best D on each, the lowest of all is the answer
        Automatic   the nested search; widens its range by itself
        Manual      the nested search inside ranges you give
    """
    from PyQt6.QtWidgets import QSpinBox
    from core.analyses import YANG_GRID_D, kD_grid, AnalysisError
    sec = Section('Search for D and kD')
    grid_r = QRadioButton('kD grid')
    grid_r.setToolTip("Yang et al. 2006, Fig. 7a: the error E against D for each kD; "
                      "the lowest point of all curves gives D and kD")
    auto_r = QRadioButton('Automatic')
    auto_r.setToolTip('Searches kD and D by itself and widens the range until the minimum '
                      'is inside it')
    man_r = QRadioButton('Manual range')
    man_r.setToolTip('The automatic search, inside the D and kD ranges you give')
    grp = QButtonGroup(sec)
    for r in (grid_r, auto_r, man_r):
        grp.addButton(r)
    sec.widget(_row(grid_r, auto_r, man_r, uniform=False))
    mode = st.get('search') or ('grid' if not st else 'auto')
    if mode == 'grid':
        grid_r.setChecked(True)
    else:
        (auto_r if st.get('auto', True) else man_r).setChecked(True)

    spec = st.get('kD_spec') or dict(start=1.0, stop=5.0, step=1.0, extras=[0.1, 0.5])
    kd_from = _spin(spec['start'], 1e-6, 1e6, 4, 0.5)
    kd_to = _spin(spec['stop'], 1e-6, 1e6, 4, 0.5)
    kd_step = _spin(spec['step'], 1e-6, 1e6, 4, 0.5)
    extras = QLineEdit(', '.join(f"{v:g}" for v in spec.get('extras') or []))
    extras.setPlaceholderText('e.g. 0.1, 0.5')
    extras.setToolTip('More kD values outside the even steps, separated by commas')
    dg = st.get('D_grid') or YANG_GRID_D
    g_lo, g_hi = _sci_edit(dg[0]), _sci_edit(dg[1])
    g_n = QSpinBox()
    g_n.setRange(5, 400)
    g_n.setValue(int(dg[2]))
    g_n.setToolTip('Points on each E-vs-D curve (log-spaced). The best D on each curve is '
                   'found exactly, between the points')
    grid_rows = (('kD from', kd_from), ('kD to', kd_to), ('kD step', kd_step),
                 ('Extra kD values', extras), ('D from (m2/s)', g_lo), ('D to (m2/s)', g_hi),
                 ('Points per curve', g_n))
    db = st.get('D_bounds') or (1e-11, 1e-7)
    kb = st.get('kD_range') or (1e-2, 1e3)
    d_lo, d_hi, k_lo, k_hi = (_sci_edit(v) for v in (*db, *kb))
    man_rows = (('D from (m2/s)', d_lo), ('D to (m2/s)', d_hi), ('kD from', k_lo),
                ('kD to', k_hi))
    for label, wdg in grid_rows + man_rows:
        sec.form.addRow(label, wdg)
    count = note('')
    sec.widget(count)
    offset = _spin(st.get('t_offset', 0.0), -1e5, 1e5, 2, 1, ' s')
    offset.setToolTip('Added to the times of the IFT curve, if the drop formed before t = 0 '
                      'of the curve (Yang et al. added 2 s)')
    sec.form.addRow('Time offset', offset)

    def values():
        try:
            return kD_grid(kd_from.value(), kd_to.value(), kd_step.value(),
                           _parse_list(extras.text(), 'Extra kD value'))
        except AnalysisError as e:
            raise ValueError(str(e)) from None

    def refresh():
        for _, wdg in grid_rows:
            sec.form.setRowVisible(wdg, grid_r.isChecked())
        for _, wdg in man_rows:
            sec.form.setRowVisible(wdg, man_r.isChecked())
        count.setVisible(grid_r.isChecked())
        if grid_r.isChecked():
            try:
                v = values()
                count.setText(f"{len(v)} values: {', '.join(f'{x:g}' for x in v)}")
            except ValueError as e:
                count.setText(str(e))
    for r in (grid_r, auto_r, man_r):
        r.toggled.connect(lambda _: refresh())
    for wdg in (kd_from, kd_to, kd_step):
        wdg.valueChanged.connect(lambda _: refresh())
    extras.editingFinished.connect(refresh)
    refresh()
    lay.addWidget(sec)

    def kwargs():
        kw = dict(t_offset=offset.value())
        if grid_r.isChecked():
            kw.update(kD_values=values(),
                      D_grid=(_sci(g_lo.text(), 'D from'), _sci(g_hi.text(), 'D to'),
                              g_n.value()),
                      kD_spec=dict(start=kd_from.value(), stop=kd_to.value(),
                                   step=kd_step.value(),
                                   extras=_parse_list(extras.text(), 'Extra kD value')))
            if not kw['D_grid'][0] < kw['D_grid'][1]:
                raise ValueError('D from must be below D to')
        elif man_r.isChecked():
            kw.update(D_bounds=(_sci(d_lo.text(), 'D from'), _sci(d_hi.text(), 'D to')),
                      kD_range=(_sci(k_lo.text(), 'kD from'), _sci(k_hi.text(), 'kD to')))
        return kw
    return kwargs


class AnalysisPages:
    def __init__(self, window):
        self.w = window

    def build(self, key, folder, lay):
        getattr(self, f'page_{key}')(folder, lay)

    # -- shared pieces --------------------------------------------------------------
    def _doc(self, folder, name):
        try:
            return folder.load_analysis(name)
        except SessionError as e:
            return dict(error=str(e))

    def _header(self, folder, name, reads, lay):
        ok, why = folder.availability()[name]
        sec = Section('Inputs')
        sec.field('Reads', reads)
        sec.field('Can run', 'Yes' if ok else f"No: {why}")
        lay.addWidget(sec)
        return ok

    def _stored(self, folder, name, doc, lay):
        """True when ``doc`` holds a readable result; a stored result that
        can't be read shows as a warning."""
        if doc is None:
            return False
        if 'error' in doc:
            lay.addWidget(warnings_section(result_warnings(folder, name, doc)))
            return False
        return True

    def _warnings(self, folder, name, doc, lay):
        lay.addWidget(warnings_section(result_warnings(folder, name, doc)))

    # -- single-frame IFT -------------------------------------------------------------
    def page_frame_ift(self, folder, lay):
        rows = {r['frame_index']: r for r in folder.log.rows}
        order = sorted((i for i in folder.frame_indices()),
                       key=lambda i: (rows.get(i, {}).get('time_s', 0.0), i))
        sec = Section('Frame')
        combo = QComboBox()
        for i in order:
            r = rows.get(i, {})
            ift = r.get('ift_mNm')
            combo.addItem(f"Frame {i}, t = {fmt(r.get('time_s'))} s"
                          + (f", IFT {fmt(ift)} mN/m" if ift == ift and ift is not None else ''), i)
        cur = self.w.graphics.frames.current()
        if cur in order:
            combo.setCurrentIndex(order.index(cur))
        combo.currentIndexChanged.connect(
            lambda k: self.w._show_frame(combo.itemData(k)) if k >= 0 else None)
        sec.form.addRow('Frame', combo)
        lay.addWidget(sec)
        busy = self.w.job_running()
        run = self._run_button(bool(order), lambda: self.w.measure_frame(combo.currentData()),
                               'Measure IFT')
        other = _button('Image file...', self.w.measure_image_file,
                        tip='Measure an image taken with the same camera, needle and '
                            'magnification (not stored)')
        other.setEnabled(not busy)
        lay.addWidget(actions(other, run))
        fit = self.w.frame_fit
        if fit is not None and fit.get('folder') is folder:
            lay.addWidget(ift_result_section(fit['rec'], f"Result: {fit['label']}"))

    def _export_button(self, name):
        return _button('Export CSV...', lambda: self.w.export_result(name),
                       tip='Save the settings, the result and its curves as CSV')

    def _run_button(self, ok, slot, text='Run'):
        b = _button(text, slot, primary=True)
        b.setEnabled(ok and not self.w.job_running())
        return b

    # -- swelling ------------------------------------------------------------------
    def page_swelling(self, folder, lay):
        ok = self._header(folder, 'swelling', 'Drop volume against time from the log', lay)
        doc = self._doc(folder, 'swelling')
        thr0 = (doc or {}).get('data', {}).get('settings', {}).get(
            'threshold', SWELLING_DEFAULT_THRESHOLD)
        sec = Section('Settings')
        thr = _spin(100 * thr0, 0.1, 99, 2, 0.5, ' %')
        thr.setToolTip('Volume change above which the drop counts as swelling '
                       '(Bagalkot: ignoring 3 % swelling shifts D by about 2 %)')
        sec.form.addRow('Swelling threshold', thr)
        lay.addWidget(sec)
        lay.addWidget(actions(self._run_button(
            ok, lambda: self.w.run_analysis('swelling', threshold=thr.value() / 100.0))))
        if not self._stored(folder, 'swelling', doc, lay):
            return
        r = doc['data']['result']
        sec = Section('Result')
        sec.field('Verdict', SWELLING_VERDICTS.get(r.get('verdict'), r.get('verdict')))
        if r.get('S') is not None:
            sec.field('Swelling', f"{fmt(100 * r['S'], 3)} +/- {fmt(100 * (r.get('S_err') or 0), 2)}",
                      '%', result=True)
            sec.field('Volume at start / end', f"{fmt(r['V0'])} / {fmt(r['V_end'])}", 'uL')
        sec.field('Frames with a volume', r.get('n_valid'))
        sec.field('Still rising at the end', 'yes' if r.get('still_rising') else 'no')
        rec = r.get('recommend')
        sec.field('Recommended method', {'yang': 'Yang (IFT; D and k)',
                                         'bagalkot': 'Bagalkot (volume; D only)'}.get(rec, '-'))
        if r.get('message') and r.get('verdict') in ('negligible', 'significant'):
            sec.widget(note(r['message']))
        lay.addWidget(sec)
        self._warnings(folder, 'swelling', doc, lay)
        buttons = [self._export_button('swelling')]
        if rec in ('yang', 'bagalkot'):
            buttons.insert(0, _button(f"Go to Diffusivity ({rec.capitalize()})",
                                      lambda: self.w.select_node(rec)))
        lay.addWidget(_row(*buttons))

    # -- Bagalkot ---------------------------------------------------------------------
    def page_bagalkot(self, folder, lay):
        from core.bagalkot_estimator import DEFAULT_D_RANGE
        doc = self._doc(folder, 'bagalkot')
        st = (doc or {}).get('data', {}).get('settings', {})
        from_log, from_file, can_run = self._volume_inputs(folder, st, lay)

        def curve():
            return 'file' if from_file.isChecked() else 'log'
        lo0, hi0 = st.get('D_range') or DEFAULT_D_RANGE
        sec = Section('Settings')
        lo, hi = _sci_edit(lo0), _sci_edit(hi0)
        sec.form.addRow('D from (m2/s)', lo)
        sec.form.addRow('D to (m2/s)', hi)
        moving = QCheckBox('Growing interface')
        moving.setToolTip('The interface moves outward as the drop swells '
                          '(Bagalkot thesis, section 4.2.4)')
        moving.setChecked(bool(st.get('moving')))
        sec.widget(moving)
        lay.addWidget(sec)

        def run():
            try:
                rng = (_sci(lo.text(), 'D from'), _sci(hi.text(), 'D to'))
            except ValueError as e:
                self.w.messages.add(str(e), 'warn')
                return
            self.w.run_analysis('bagalkot', D_range=rng, moving=moving.isChecked(),
                                curve=curve())
        run_b = self._run_button(True, run)
        busy = self.w.job_running()

        def update():
            ok, why = folder.bagalkot_availability(curve())
            run_b.setEnabled(ok and not busy)
            can_run.setText('Yes' if ok else f"No: {why}")
        for r in (from_log, from_file):
            r.toggled.connect(lambda _: update())
        update()
        lay.addWidget(actions(run_b))
        if not self._stored(folder, 'bagalkot', doc, lay):
            return
        r = doc['data']['result']
        sec = Section('Result')
        sec.field('D', fmt(r['D'], 4), 'm2/s', result=True)
        sec.field('D', fmt(r['D_per_min'], 4), 'm2/min')
        lo_, hi_ = r['D_near']
        sec.field(f"Equally good (F within {fmt(100 * r['near_tol'], 2)} %)",
                  f"{fmt(lo_, 3)} to {fmt(hi_, 3)}" + (' or beyond' if r['near_open'] else ''),
                  'm2/s')
        sec.field('Swelling factor SF', fmt(r['SF'], 4))
        sec.field('Misfit F', fmt(r['F_min'], 3))
        sec.field('Simulations', r['n_evaluations'])
        sec.field('Volume curve from', 'the file volume_curve.csv'
                  if doc['data']['settings'].get('curve') == 'file' else 'the frame log')
        lay.addWidget(sec)
        self._warnings(folder, 'bagalkot', doc, lay)
        lay.addWidget(_row(_button('Show the fit', lambda: self.w.graphics.show_result('bagalkot')),
                           self._export_button('bagalkot')))

    # -- Yang -------------------------------------------------------------------------
    def page_yang(self, folder, lay):
        doc = self._doc(folder, 'yang')
        st = (doc or {}).get('data', {}).get('settings', {})
        frame, from_log, from_file, can_run = self._yang_inputs(folder, st, lay)

        def curve():
            return 'file' if from_file.isChecked() else 'log'
        search = yang_search(st, lay)

        def run():
            try:
                kw = search()
            except ValueError as e:
                self.w.messages.add(str(e), 'warn')
                return
            kw.update(curve=curve(), frame_index=frame.currentData())
            self.w.run_analysis('yang', **kw)
        run_b = self._run_button(True, run)
        busy = self.w.job_running()

        def update():
            ok, why = folder.yang_availability(curve())
            run_b.setEnabled(ok and not busy)
            can_run.setText('Yes' if ok else f"No: {why}")
        for r in (from_log, from_file):
            r.toggled.connect(lambda _: update())
        update()
        lay.addWidget(actions(run_b))
        if not self._stored(folder, 'yang', doc, lay):
            return
        r = doc['data']['result']
        sec = Section('Result')
        sec.field('D', fmt(r['D'], 4), 'm2/s', result=True)
        sec.field('k', fmt(r['k'], 4), 'm/s', result=True)
        sec.field('kD (Biot number)', fmt(r['kD'], 4))
        sec.field('Fit error E (eq. 14)', fmt(r['E_percent'], 3), '%')
        sec.field('RMS misfit', fmt(r['residual_rms'], 3), 'mN/m')
        tol = fmt(100 * r['near_tol'], 2)
        more = ' or beyond' if r['near_open'] else ''
        sec.field(f"Equally good D (E within {tol} %)",
                  f"{fmt(r['D_near'][0], 3)} to {fmt(r['D_near'][1], 3)}{more}", 'm2/s')
        sec.field('Equally good kD', f"{fmt(r['kD_near'][0], 3)} to {fmt(r['kD_near'][1], 3)}")
        sec.field('Simulations', r['n_evaluations'] + r.get('n_scan_evaluations', 0))
        st_ = doc['data'].get('settings', {})
        m = doc['data'].get('model', {})
        sec.field('IFT curve from', 'the file ift_curve.csv' if st_.get('curve') == 'file'
                  else 'the frame log')
        if st_.get('search') == 'grid':
            sec.field('Search', f"kD grid: {', '.join(f'{v:g}' for v in st_['kD_values'])}")
        else:
            sec.field('Search', 'automatic' if st_.get('auto', True) else 'manual range')
        sec.field('Drop outline from', f"frame {m.get('frame_index')}")
        lay.addWidget(sec)
        self._warnings(folder, 'yang', doc, lay)
        lay.addWidget(_row(_button('Show the fit', lambda: self.w.graphics.show_result('yang')),
                           _button('Show the model', lambda: self.w.graphics.show_result('mesh')),
                           self._export_button('yang')))

    def _yang_inputs(self, folder, st, lay):
        """The three inputs of Yang's method: the drop outline (one frame), the
        gamma(c) calibration and IFT against time (the log, or a file)."""
        from PyQt6.QtWidgets import QLabel
        sec = Section('Inputs')
        combo = self._frame_combo(folder, st.get('frame_index'))
        sec.form.addRow('Drop outline from', combo)
        has_cal = bool(folder.calibration_path and folder.calibration_path.is_file())
        sec.field('gamma(c)', 'stored in the session' if has_cal else 'not loaded',
                  buttons=[] if has_cal else [_button('Load...', self.w.load_calibration,
                                                      tip='Load the gamma(c) calibration')])
        lay.addWidget(sec)

        sec = Section('IFT against time')
        n_log = sum(1 for r in folder.log.rows if r['qc_pass'] and r['ift_mNm'] == r['ift_mNm']
                    and (r['ift_mNm'] or 0) > 0) if folder.log.has('ift_mNm') else 0
        from_log = QRadioButton(f"Frame log ({n_log} frame{'s' if n_log != 1 else ''} passed QC)")
        from_log.setToolTip(f"The IFT of the logged frames; needs at least {YANG_MIN_POINTS}")
        from_file = QRadioButton('IFT curve file (CSV)')
        from_file.setToolTip('IFT against time measured for this drop by another instrument or '
                             'program, for example when only one image of the drop was logged')
        grp = QButtonGroup(sec)
        grp.addButton(from_log)
        grp.addButton(from_file)
        curve = st.get('curve') or ('log' if folder.yang_availability('log')[0] else 'file')
        (from_file if curve == 'file' else from_log).setChecked(True)
        sec.widget(from_log)
        sec.widget(from_file)
        try:
            c = folder.ift_curve()
            info = ('none loaded' if c is None else
                    f"{len(c['t'])} points, t = {fmt(c['t'][0])} to {fmt(c['t'][-1])} s, "
                    f"IFT {fmt(c['gamma'].min())} to {fmt(c['gamma'].max())} mN/m")
        except SessionError as e:
            info = str(e)
        sec.field('Curve file', info, buttons=[
            _button('Load...', self.w.load_ift_curve,
                    tip='Load IFT against time (CSV); copied into the session folder'),
            _button('Template...', self.w.save_ift_template,
                    tip='Save a CSV with the expected columns (time_s, ift_mNm)')])
        can = QLabel()
        can.setWordWrap(True)
        can.setObjectName('FieldValue')
        sec.form.addRow('Can run', can)
        lay.addWidget(sec)
        return combo, from_log, from_file, can

    def _volume_inputs(self, folder, st, lay):
        """Where Bagalkot's V(t) comes from: the frame log or a volume CSV."""
        from PyQt6.QtWidgets import QLabel
        sec = Section('Drop volume against time')
        n_log = (sum(1 for r in folder.log.rows if r['fit_ok'] and r['volume_uL'] == r['volume_uL'])
                 if folder.log.has('volume_uL') else 0)
        from_log = QRadioButton(f"Frame log ({n_log} frame{'s' if n_log != 1 else ''} with a "
                                f"volume)")
        from_file = QRadioButton('Volume curve file (CSV)')
        from_file.setToolTip('Volume against time from another program or instrument, as in '
                             'the Bagalkot window')
        grp = QButtonGroup(sec)
        grp.addButton(from_log)
        grp.addButton(from_file)
        curve = st.get('curve') or ('log' if folder.bagalkot_availability('log')[0]
                                    or folder.volume_curve_path is None else 'file')
        (from_file if curve == 'file' else from_log).setChecked(True)
        sec.widget(from_log)
        sec.widget(from_file)
        try:
            c = folder.volume_curve()
            info = ('none loaded' if c is None else
                    f"{len(c['t'])} points, t = {fmt(c['t'][0])} to {fmt(c['t'][-1])} s, "
                    f"V {fmt(c['volume'].min())} to {fmt(c['volume'].max())} uL")
        except SessionError as e:
            info = str(e)
        sec.field('Curve file', info, buttons=[
            _button('Load...', self.w.load_volume_curve,
                    tip='Load volume against time (CSV); copied into the session folder'),
            _button('Template...', lambda: self.w.save_curve_template('volume'),
                    tip='Save a CSV with the expected columns (time_s, volume_uL)')])
        can = QLabel()
        can.setWordWrap(True)
        can.setObjectName('FieldValue')
        sec.form.addRow('Can run', can)
        lay.addWidget(sec)
        return from_log, from_file, can

    def _frame_combo(self, folder, want=None):
        """The measured frames; the chosen one's outline becomes the model."""
        frames = folder.measured_frames()
        combo = QComboBox()
        for r in frames:
            combo.addItem(f"Frame {r['frame_index']}, t = {fmt(r['time_s'])} s", r['frame_index'])
        if not frames:
            combo.addItem('No frame with a successful fit', None)
        combo.setEnabled(len(frames) > 1)
        combo.setToolTip('The frame whose outline becomes the model geometry (mesh, FEM). '
                         'Yang et al. used the drop just after it formed.')
        k = combo.findData(want) if want is not None else -1
        combo.setCurrentIndex(max(k, 0))
        combo.currentIndexChanged.connect(
            lambda i: self.w._show_frame(combo.itemData(i)) if combo.itemData(i) is not None
            else None)
        return combo

    def page_yang_model(self, folder, lay):
        doc = next((d for d in (self._doc(folder, 'yang'), self._doc(folder, 'simulation'))
                    if d and 'error' not in d), None)
        if doc is None:
            lay.addWidget(note('Built when Diffusivity (Yang) or the simulation runs.'))
            return
        m = doc['data']['model']
        sec = Section('Model')
        sec.field('Drop outline from', f"frame {m['frame_index']} (t = {fmt(m['frame_time_s'], 4)} s)")
        dom = m.get('domain')
        if dom:
            sec.field('Drop height', fmt(dom['drop_height_m'] * 1e3, 4), 'mm')
            sec.field('Max radius', fmt(dom['r_max_m'] * 1e3, 4), 'mm')
            sec.field('Needle ID / OD', f"{fmt(2 * dom['r_inner_m'] * 1e3, 4)} / "
                                        f"{fmt(2 * dom['r_outer_m'] * 1e3, 4)}", 'mm')
        sec.field('Mesh', f"{m['n_nodes']} nodes, {m['n_elements']} elements")
        sec.field('Smallest angle', fmt(m['min_angle'], 3), 'degrees')
        c = doc['data'].get('calibration') or {}
        if c:
            sec.field('Calibration', f"{c.get('n_points')} points, c_sat {fmt(c.get('c_sat'))}")
        lay.addWidget(sec)
        fem = m.get('fem')
        if fem:
            sec = Section('FEM checks')
            sec.field('System size', f"{fem['size']} x {fem['size']}")
            for name, ok, detail in fem['checks']:
                sec.field(name, 'pass' if ok else 'FAIL')
            lay.addWidget(sec)
        lay.addWidget(warnings_section(
            ['Some FEM checks fail: the model may not conserve the solute. Check the drop '
             'outline and the mesh.'] if fem and not fem['all_pass'] else []))
        lay.addWidget(_row(_button('Show the model', lambda: self.w.graphics.show_result('mesh'))))

    # -- forward simulation ------------------------------------------------------------
    def page_simulation(self, folder, lay):
        from core.analyses import simulation_defaults
        ok, why = folder.availability()['simulation']
        doc = self._doc(folder, 'simulation')
        st = (doc or {}).get('data', {}).get('settings', {}) if doc and 'error' not in doc else {}
        D0, k0, dur0, frame0 = simulation_defaults(folder)
        has_yang = frame0 is not None
        sec = Section('Inputs')
        frame = self._frame_combo(folder, st.get('frame_index', frame0))
        sec.form.addRow('Drop outline from', frame)
        has_cal = bool(folder.calibration_path and folder.calibration_path.is_file())
        sec.field('gamma(c)', 'stored in the session' if has_cal
                  else 'not loaded (Cs(t) only)',
                  buttons=[] if has_cal else [_button('Load...', self.w.load_calibration,
                                                      tip='Load the gamma(c) calibration')])
        sec.field('Can run', 'Yes' if ok else f"No: {why}")
        lay.addWidget(sec)

        sec = Section('Parameters')
        d_e, k_e = _sci_edit(st.get('D', D0)), _sci_edit(st.get('k', k0))
        sec.form.addRow('D (m2/s)', d_e)
        sec.form.addRow('k (m/s)', k_e)
        dur = _spin(st.get('duration_s', dur0), 0.01, 1e7, 2, 10, ' s')
        dur.setToolTip('Simulated time from the moment the drop formed')
        sec.form.addRow('Duration', dur)
        if has_yang:
            def use_yang():
                d_e.setText(f"{D0:.4g}")
                k_e.setText(f"{k0:.4g}")
                dur.setValue(dur0)
            sec.widget(_row(_button("Use Yang's best fit", use_yang,
                                    tip=f"D = {D0:.3g} m2/s, k = {k0:.3g} m/s, "
                                        f"{dur0:g} s")))
        auto = QRadioButton('Automatic time steps')
        auto.setToolTip('Fine steps at first, growing later (as the Yang fit uses)')
        fixed = QRadioButton('Fixed time step')
        fixed.setToolTip('Uniform steps, theta = 0.7, as the Pendant window\'s Step 7')
        grp = QButtonGroup(sec)
        grp.addButton(auto)
        grp.addButton(fixed)
        (fixed if st.get('dt_s') else auto).setChecked(True)
        sec.widget(auto)
        sec.widget(fixed)
        dt = _spin(st.get('dt_s') or 0.01, 1e-5, 1e5, 5, 0.01, ' s')
        sec.form.addRow('Time step', dt)
        fixed.toggled.connect(lambda on: sec.form.setRowVisible(dt, on))
        sec.form.setRowVisible(dt, fixed.isChecked())
        lay.addWidget(sec)

        from core.analyses import (SIM_DEFAULT_SNAPSHOTS, SIM_MAX_SNAPSHOTS,
                                   SIM_SNAPSHOT_SPACINGS)
        sec = Section('Snapshots')
        n_snap = QSpinBox()
        n_snap.setRange(1, SIM_MAX_SNAPSHOTS)
        n_snap.setValue(int(st.get('n_snapshots') or SIM_DEFAULT_SNAPSHOTS))
        n_snap.setToolTip('How many concentration fields are stored over the run')
        sec.form.addRow('Number', n_snap)
        spacing = QComboBox()
        for key, label in SIM_SNAPSHOT_SPACINGS.items():
            spacing.addItem(label, key)
        spacing.setCurrentIndex(max(0, spacing.findData(st.get('snapshot_spacing', 'early'))))
        spacing.setToolTip('Dense early: more snapshots at the start, where the '
                           'concentration changes fastest')
        sec.form.addRow('Spacing', spacing)
        times = QLineEdit(', '.join(f"{x:g}" for x in st.get('snapshot_times') or ()))
        times.setPlaceholderText('e.g. 60, 600, 1200')
        times.setToolTip('Times (s) that always get a snapshot, separated by commas')
        sec.form.addRow('Also at (s)', times)
        lay.addWidget(sec)

        def run():
            try:
                kw = dict(D=_sci(d_e.text(), 'D'), k=_sci(k_e.text(), 'k'),
                          duration_s=dur.value(), frame_index=frame.currentData(),
                          dt_s=dt.value() if fixed.isChecked() else None,
                          n_snapshots=n_snap.value(), snapshot_spacing=spacing.currentData(),
                          snapshot_times=_parse_list(times.text(), 'Snapshot times'))
            except ValueError as e:
                self.w.messages.add(str(e), 'warn')
                return
            self.w.run_analysis('simulation', **kw)
        lay.addWidget(actions(self._run_button(ok, run)))
        if not self._stored(folder, 'simulation', doc, lay):
            return
        r, st = doc['data']['result'], doc['data']['settings']
        sec = Section('Result')
        sec.field('D, k from', "Yang's fit" if st.get('source') == 'yang' else 'entered')
        sec.field('D', fmt(st['D'], 4), 'm2/s')
        sec.field('k', fmt(st['k'], 4), 'm/s')
        sec.field('kD (Biot number)', fmt(r['kD'], 4))
        sec.field('Simulated', f"{fmt(r['simulated_s'], 5)} s in {r['n_steps']} steps")
        sec.field('Snapshots stored', str(len(doc['data'].get('snapshots') or ())))
        sec.field('Final Cs', fmt(r['Cs_final'], 4), 'c/c_sat', result=True)
        if r.get('gamma_final') is not None:
            sec.field('Final IFT', fmt(r['gamma_final'], 4), 'mN/m', result=True)
            sec.field('IFT range', f"{fmt(r['gamma_range'][0], 4)} to "
                                   f"{fmt(r['gamma_range'][1], 4)}", 'mN/m')
        sec.field('C over the run', f"{fmt(r['C_min'], 3)} to {fmt(r['C_max'], 3)}")
        sec.field('NaN / negative', f"{'yes' if r['has_nan'] else 'no'} / "
                                    f"{'yes' if r['has_negative'] else 'no'}")
        lay.addWidget(sec)
        self._warnings(folder, 'simulation', doc, lay)
        lay.addWidget(_row(_button('Show the simulation',
                                   lambda: self.w.graphics.show_result('simulation')),
                           _button('Show the model', lambda: self.w.graphics.show_result('mesh')),
                           self._export_button('simulation')))

    # -- results -------------------------------------------------------------------------
    def page_results(self, folder, lay):
        from core.analyses import LABELS
        sec = Section('Stored results')
        any_ = False
        warnings = []
        for name, label in LABELS.items():
            doc = self._doc(folder, name)
            if not doc or 'error' in doc:
                continue
            any_ = True
            r = doc['data']['result']
            summary = {'swelling': lambda: (f"{fmt(100 * r['S'], 3)} % swelling, {r.get('verdict')}"
                                            if r.get('S') is not None else str(r.get('verdict'))),
                       'bagalkot': lambda: f"D = {fmt(r['D'], 3)} m2/s",
                       'yang': lambda: f"D = {fmt(r['D'], 3)} m2/s, k = {fmt(r['k'], 3)} m/s",
                       'simulation': lambda: (
                           f"Cs {fmt(r['Cs_final'], 3)}" + (
                               f", IFT {fmt(r['gamma_final'], 4)} mN/m"
                               if r.get('gamma_final') is not None else '')
                           + f" after {fmt(r['simulated_s'], 4)} s")}[name]()
            stale = ' (out of date)' if folder.analysis_out_of_date(name) else ''
            sec.field(label + stale, summary)
            sec.widget(_row(_button('Open', lambda n=name: self.w.select_node(n),
                                    tip='Settings and result'),
                            _button('Plot', lambda n=name: self.w.graphics.show_result(n)),
                            self._export_button(name)))
            warnings += [f"{label}: {w}" for w in result_warnings(folder, name, doc)]
        if not any_:
            sec.widget(note('No analysis has run on this session yet.'))
        lay.addWidget(sec)
        if any_:
            lay.addWidget(warnings_section(warnings))

    # -- threshold (rebuild the log with another level) -------------------------------------
    def page_threshold(self, folder, lay):
        level = None
        for op, a in folder.session.prep_ops:
            if op == 'threshold':
                level = a
        sec = Section('Threshold')
        sec.field('Used for the log', level if level is not None else 'automatic (Otsu)')
        lay.addWidget(sec)
        sec = Section('Change and rebuild')
        auto = QRadioButton('Automatic (Otsu)')
        auto.setToolTip("Otsu's threshold, computed again on every frame")
        fixed = QRadioButton('Fixed level')
        grp = QButtonGroup(sec)
        grp.addButton(auto)
        grp.addButton(fixed)
        (auto if level is None else fixed).setChecked(True)
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.setValue(128 if level is None else level)
        sec.widget(auto)
        sec.widget(_row(fixed, spin, uniform=False))
        lay.addWidget(sec)

        def rebuild():
            ops = [] if auto.isChecked() else [('threshold', spin.value())]
            self.w.rebuild_log(None, prep_ops=ops)
        b = _button('Rebuild log', rebuild, primary=True,
                    tip='Fit every stored frame again with this threshold; stored results '
                        'become out of date')
        b.setEnabled(bool(folder.frame_indices()) and not self.w.job_running())
        lay.addWidget(actions(b))
