"""
gui.shell.analysis_pages
========================

Settings pages for the analyses of an open session (core.analyses) and for
changing the threshold of its frames: settings, a Run button, and the
stored result in plain words.  Runs go through the window's job runner, so
they show progress, can be stopped, and refresh the tree and the result
tabs when they finish.
"""

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QDoubleSpinBox, QLineEdit,
                             QPushButton, QRadioButton, QSpinBox)

from core import frame_log as fl
from core.analyses import SWELLING_DEFAULT_THRESHOLD
from core.session import YANG_MIN_POINTS, SessionError
from gui.shell.draft_pages import _button, _row, _spin
from gui.shell.pages import Section, callout, note
from gui.shell.panels import fmt

KEYS = ('swelling', 'bagalkot', 'yang', 'yang_model', 'results', 'threshold')


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

    def _stored_notes(self, folder, name, doc, lay):
        if doc is None:
            return False
        if 'error' in doc:
            lay.addWidget(callout(f"The stored result can't be read: {doc['error']}"))
            return False
        if folder.analysis_out_of_date(name):
            lay.addWidget(callout('The frame log changed since this ran (for example it was '
                                  'rebuilt), so this result is out of date. Run it again.'))
        return True

    def _run_button(self, ok, slot, text='Run'):
        b = _button(text, 'compute', slot, primary=True)
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
        lay.addWidget(_row(self._run_button(
            ok, lambda: self.w.run_analysis('swelling', threshold=thr.value() / 100.0))))
        if not self._stored_notes(folder, 'swelling', doc, lay):
            return
        r = doc['data']['result']
        good = r.get('verdict') in ('negligible', 'not_measurable')
        lay.addWidget(callout(r.get('message', ''), 'good' if good else 'warn'))
        sec = Section('Result')
        if r.get('S') is not None:
            sec.field('Swelling', f"{fmt(100 * r['S'], 3)} +/- {fmt(100 * (r.get('S_err') or 0), 2)}",
                      '%', result=True)
            sec.field('Volume at start / end', f"{fmt(r['V0'])} / {fmt(r['V_end'])}", 'uL')
        sec.field('Frames with a volume', r.get('n_valid'))
        sec.field('Still rising at the end', 'yes' if r.get('still_rising') else 'no')
        rec = r.get('recommend')
        sec.field('Recommended method', {'yang': 'Yang (IFT; D and k)',
                                         'bagalkot': 'Bagalkot (volume; D only)'}.get(rec, '-'))
        lay.addWidget(sec)
        if rec in ('yang', 'bagalkot'):
            lay.addWidget(_row(_button(f"Go to Diffusivity ({rec.capitalize()})", 'next',
                                       lambda: self.w.select_node(rec))))

    # -- Bagalkot ---------------------------------------------------------------------
    def page_bagalkot(self, folder, lay):
        from core.bagalkot_estimator import DEFAULT_D_RANGE
        ok = self._header(folder, 'bagalkot', 'Drop volume against time from the log', lay)
        doc = self._doc(folder, 'bagalkot')
        st = (doc or {}).get('data', {}).get('settings', {})
        lo0, hi0 = st.get('D_range') or DEFAULT_D_RANGE
        sec = Section('Settings')
        lo, hi = _sci_edit(lo0), _sci_edit(hi0)
        sec.form.addRow('D from (m2/s)', lo)
        sec.form.addRow('D to (m2/s)', hi)
        moving = QCheckBox('Growing interface (thesis section 4.2.4)')
        moving.setChecked(bool(st.get('moving')))
        sec.widget(moving)
        sec.widget(note("Bagalkot's sphere model fitted to V(t): equally spaced D values, "
                        "then refined around the best. Gives D only (no k)."))
        lay.addWidget(sec)

        def run():
            try:
                rng = (_sci(lo.text(), 'D from'), _sci(hi.text(), 'D to'))
            except ValueError as e:
                self.w.messages.add(str(e), 'warn')
                return
            self.w.run_analysis('bagalkot', D_range=rng, moving=moving.isChecked())
        lay.addWidget(_row(self._run_button(ok, run)))
        if not self._stored_notes(folder, 'bagalkot', doc, lay):
            return
        r = doc['data']['result']
        if r['at_edge']:
            lay.addWidget(callout('The best D is at the edge of the search range; widen it '
                                  'and run again.'))
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
        lay.addWidget(sec)
        lay.addWidget(_row(_button('Show the fit', 'plot',
                                   lambda: self.w.graphics.show_result('bagalkot'))))

    # -- Yang -------------------------------------------------------------------------
    def page_yang(self, folder, lay):
        ok = self._header(folder, 'yang', f"IFT against time from the log (frames that passed "
                          f"QC, at least {YANG_MIN_POINTS}), the first measured frame (drop "
                          f"geometry) and the gamma(c) calibration", lay)
        if not (folder.calibration_path and folder.calibration_path.is_file()):
            lay.addWidget(_row(_button('Load calibration...', 'calcurve',
                                       lambda: self.w.select_node('calibration'))))
        doc = self._doc(folder, 'yang')
        st = (doc or {}).get('data', {}).get('settings', {})
        sec = Section('Search range')
        auto = QRadioButton('Automatic (widens until the minimum is inside)')
        manual = QRadioButton('Manual')
        grp = QButtonGroup(sec)
        grp.addButton(auto)
        grp.addButton(manual)
        (auto if st.get('auto', True) else manual).setChecked(True)
        sec.widget(auto)
        sec.widget(manual)
        db = st.get('D_bounds') or (1e-11, 1e-7)
        kb = st.get('kD_range') or (1e-2, 1e3)
        d_lo, d_hi, k_lo, k_hi = (_sci_edit(v) for v in (*db, *kb))
        for label, wdg in (('D from (m2/s)', d_lo), ('D to (m2/s)', d_hi),
                           ('kD from', k_lo), ('kD to', k_hi)):
            sec.form.addRow(label, wdg)
        offset = _spin(st.get('t_offset', 0.0), -1e5, 1e5, 2, 1, ' s')
        offset.setToolTip('Shift of the log times, if the drop formed before t = 0 of the log')
        sec.form.addRow('Time offset', offset)

        def toggle():
            for wdg in (d_lo, d_hi, k_lo, k_hi):
                sec.form.setRowVisible(wdg, manual.isChecked())
        manual.toggled.connect(lambda _: toggle())
        toggle()
        lay.addWidget(sec)
        lay.addWidget(note("Yang et al. 2006: for each kD (= k rn / D) the best D is found by "
                           "forward simulations of the diffusion model; this takes minutes."))

        def run():
            kw = dict(t_offset=offset.value())
            if manual.isChecked():
                try:
                    kw.update(D_bounds=(_sci(d_lo.text(), 'D from'), _sci(d_hi.text(), 'D to')),
                              kD_range=(_sci(k_lo.text(), 'kD from'), _sci(k_hi.text(), 'kD to')))
                except ValueError as e:
                    self.w.messages.add(str(e), 'warn')
                    return
            self.w.run_analysis('yang', **kw)
        lay.addWidget(_row(self._run_button(ok, run)))
        if not self._stored_notes(folder, 'yang', doc, lay):
            return
        r = doc['data']['result']
        good = r['success'] and r['D_well_determined']
        msg = ("D is well determined by this curve." if r['D_well_determined'] else
               "D is not well determined: fits this good span a wide range of D, so this "
               "curve alone cannot tell them apart.")
        lay.addWidget(callout(msg + (f" {r['message']}" if r.get('message') else ''),
                              'good' if good else 'warn'))
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
        lay.addWidget(sec)
        lay.addWidget(_row(_button('Show the fit', 'plot', lambda: self.w.graphics.show_result('yang')),
                           _button('Show the mesh', 'mesh', lambda: self.w.graphics.show_result('mesh'))))

    def page_yang_model(self, folder, lay):
        doc = self._doc(folder, 'yang')
        if not doc or 'error' in doc:
            lay.addWidget(note('The diffusion model (geometry, mesh, FEM system) is built from '
                               'the first measured frame when Diffusivity (Yang) runs.'))
            return
        m = doc['data']['model']
        sec = Section('Model')
        sec.field('Drop outline from', f"frame {m['frame_index']} (t = {fmt(m['frame_time_s'], 4)} s)")
        sec.field('Mesh', f"{m['n_nodes']} nodes, {m['n_elements']} elements")
        sec.field('Smallest angle', fmt(m['min_angle'], 3), 'degrees')
        c = doc['data'].get('calibration') or {}
        if c:
            sec.field('Calibration', f"{c.get('n_points')} points, c_sat {fmt(c.get('c_sat'))}")
        lay.addWidget(sec)
        lay.addWidget(_row(_button('Show the mesh', 'mesh', lambda: self.w.graphics.show_result('mesh'))))

    # -- results -------------------------------------------------------------------------
    def page_results(self, folder, lay):
        from core.analyses import LABELS
        sec = Section('Stored results')
        any_ = False
        for name, label in LABELS.items():
            doc = self._doc(folder, name)
            if not doc or 'error' in doc:
                continue
            any_ = True
            r = doc['data']['result']
            summary = {'swelling': lambda: (f"{fmt(100 * r['S'], 3)} % swelling, {r.get('verdict')}"
                                            if r.get('S') is not None else str(r.get('verdict'))),
                       'bagalkot': lambda: f"D = {fmt(r['D'], 3)} m2/s",
                       'yang': lambda: f"D = {fmt(r['D'], 3)} m2/s, k = {fmt(r['k'], 3)} m/s"}[name]()
            stale = ' (out of date)' if folder.analysis_out_of_date(name) else ''
            sec.field(label + stale, summary)
            sec.widget(_row(_button('Settings and result', 'next',
                                    lambda _=False, n=name: self.w.select_node(n)),
                            _button('Plot', 'plot', lambda _=False, n=name: self.w.graphics.show_result(n))))
        if not any_:
            sec.widget(note('No analysis has run on this session yet.'))
        lay.addWidget(sec)
        lay.addWidget(note('Each result is stored in the session folder (analyses/), with the '
                           'settings it used and the log it was computed from.'))

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
        auto = QRadioButton('Automatic (Otsu) on every frame')
        fixed = QRadioButton('Fixed level')
        grp = QButtonGroup(sec)
        grp.addButton(auto)
        grp.addButton(fixed)
        (auto if level is None else fixed).setChecked(True)
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.setValue(128 if level is None else level)
        sec.widget(auto)
        sec.widget(_row(fixed, spin))
        sec.widget(note('The frames are stored before the threshold, so every frame can be '
                        'fitted again with another level. Stored analysis results become '
                        'out of date.'))
        lay.addWidget(sec)

        def rebuild():
            ops = [] if auto.isChecked() else [('threshold', spin.value())]
            self.w.rebuild_log(None, prep_ops=ops)
        b = _button('Rebuild log with this threshold', 'rebuild', rebuild, primary=True)
        b.setEnabled(bool(folder.frame_indices()) and not self.w.job_running())
        lay.addWidget(_row(b))
