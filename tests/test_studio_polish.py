"""
Drop Analysis Studio: the settings-page polish (2026-10-16).

    1  Fluids & Needle: edits and presets wait for Apply; Reset goes back
    2  the breadcrumb links go to their page; the Guide opens at the page's topic
    3  the Settings panel always shows its vertical scroll bar
    4  page buttons carry no icons
    5  every ribbon page paints the light background
    8  Yang Fit: the kD picker shows / hides curves; with many kD only the best
       and its neighbours show at first; Best only
    9  the Yang Fit tab shows no warnings; a Warnings section lists them
    10 the frame rate can be typed as seconds per frame or frames per second
    11 the Log Plot shows IFT and volume on one graph (two axes) by default
    12 Simulation: a typed time shows its snapshot, or offers to compute it
    13 log axes keep their labels apart when zoomed
    14 result tabs: tiles on a grid of equal cells, layout presets (remembered),
       double-click to enlarge; Yang kD colour scale and hover instead of a legend

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_polish.py      (a few seconds)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from _synthetic_video import render_drop

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def yang_data(kds, message=''):
    curves = []
    for kd in kds:
        D = np.logspace(-11, -8, 30)
        E = 2 + 3 * (np.log10(D) + 9.6 - 0.15 * kd) ** 2 + 0.01 * kd
        curves.append(dict(kD=kd, D=list(D), E=list(E), D_min=float(D[E.argmin()]),
                           E_min=float(E.min())))
    best = min(curves, key=lambda c: c['E_min'])
    t = np.linspace(0, 100, 20)
    return dict(result=dict(E_percent=best['E_min'], near_tol=0.1, kD=best['kD'], D=4e-10,
                            k=1.4e-5, D_well_determined=False, message=message),
                curve=dict(t=list(t), gamma_exp=list(41 + 30 * np.exp(-t / 20)),
                           gamma_fit=list(40.5 + 30 * np.exp(-t / 20))),
                sweep=dict(kD=kds, E=[c['E_min'] for c in curves]), curves=curves)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='studio polish '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    from PyQt6.QtCore import QSettings, Qt
    from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QLabel, QMessageBox,
                                 QPushButton)
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    from gui.shell.main_window import StudioWindow
    from gui.shell.kit import warnings_section
    from gui.shell.result_views import MAX_SHOWN, YangView

    w = StudioWindow(settings_store=QSettings(str(tmp / 's.ini'), QSettings.Format.IniFormat))
    w.resize(1280, 800)
    w.show()
    png = tmp / 'drop.png'
    cv2.imwrite(str(png), render_drop(0.30, 70)[0])
    w.new_session('image', path=str(png))

    def button(text):
        return next((b for b in w.settings.widget().findChildren(QPushButton)
                     if b.text() == text), None)

    print("── 1: Fluids & Needle apply ──")
    w.select_node('fluids')
    rho0 = w.draft.fluids['density_inner']
    spin = next(s for s in w.settings.widget().findChildren(QDoubleSpinBox)
                if abs(s.value() - rho0) < 1e-9)
    check("Apply disabled until something changes", not button('Apply').isEnabled())
    spin.setValue(rho0 + 5)
    check("an edit waits for Apply", w.draft.fluids['density_inner'] == rho0
          and button('Apply').isEnabled())
    button('Reset').click()
    check("Reset goes back", w.draft.fluids['density_inner'] == rho0)
    spin = next(s for s in w.settings.widget().findChildren(QDoubleSpinBox)
                if abs(s.value() - rho0) < 1e-9)
    spin.setValue(rho0 + 5)
    button('Apply').click()
    check("Apply stores it", w.draft.fluids['density_inner'] == rho0 + 5
          and w.draft.fluids.get('name') == '(custom)')

    print("── 2, 3, 4: page header, scroll bar, buttons ──")
    w.select_node('fluids')
    crumb = next(l for l in w.settings.widget().findChildren(QLabel)
                 if l.objectName() == 'PageCrumb')
    check("breadcrumb has links", 'href="global"' in crumb.text() and 'href="session"' in crumb.text())
    crumb.linkActivated.emit('global')
    check("a link goes to its page", w.settings.key == 'global')
    w.show_guide('yang')
    g = w._guide
    check("Guide opens at the page's topic", g is not None and g.isVisible()
          and 'Yang' in g.text.toPlainText())
    g.close()
    check("Settings scroll bar always on",
          w.settings.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    icons = []
    for key in ('session', 'fluids', 'yang', 'scale', 'baseline', 'logger', 'crop'):
        w.select_node(key)
        icons += [f"{key}: {b.text()}" for b in w.settings.widget().findChildren(QPushButton)
                  if not b.icon().isNull() and key != 'global']
    check("no icons on page buttons", not icons, '; '.join(icons))

    print("── 5: ribbon ──")
    pages = list(w.ribbon.pages.values())
    check("every ribbon page paints its background",
          all(p.testAttribute(Qt.WidgetAttribute.WA_StyledBackground) for p in pages))

    print("── 8, 9: Yang Fit ──")
    v = YangView()
    v.set_data(yang_data([0.1, 0.5, 1, 2, 3], message='the best fit is at the edge'))
    shown = [k for k, (line, _l, _n) in v.items.items() if line.isVisible()]
    check("few kD: all shown", len(shown) == 5 and v.pick_button.text().endswith('5 of 5'))
    v.list.item(1).setCheckState(Qt.CheckState.Unchecked)
    check("unticking hides that curve", not v.items[v.list.item(1).data(
        Qt.ItemDataRole.UserRole)][0].isVisible() and v.pick_button.text().endswith('4 of 5'))
    v._best_only()
    shown = [k for k, (line, _l, _n) in v.items.items() if line.isVisible()]
    check("Best only", shown == [v.best_kD], str(shown))
    check("no warning text on the tab", 'edge' not in v.note.text() and 'NOT' not in v.note.text())
    many = list(np.round(np.linspace(0.2, 12, 30), 2))
    v.set_data(yang_data(many))
    shown = sorted(k for k, (line, _l, _n) in v.items.items() if line.isVisible())
    check("many kD: only the best and its neighbours at first",
          len(shown) == MAX_SHOWN and v.best_kD in shown and v.list.count() == 30, str(shown))
    v._set_all(True)
    check("All shows every curve", v.pick_button.text().endswith('30 of 30'))
    sec = warnings_section([])
    check("'No warnings' when clear", any(l.text() == 'No warnings'
                                          for l in sec.findChildren(QLabel)))
    sec = warnings_section(['a < b', 'second'])
    txt = ' '.join(l.text() for l in sec.findChildren(QLabel))
    check("warnings listed as bullets", txt.count('<li') == 2 and 'a &lt; b' in txt)

    print("── 10: frame rate ──")
    from types import SimpleNamespace
    from gui.shell.draft_pages import RateBox
    got = []
    win = SimpleNamespace(rate_unit='spf')
    rb = RateBox(win, 0.5, 0.05, 3600, got.append)
    check("seconds per frame shown", rb.spin.suffix() == ' s' and rb.spin.value() == 0.5)
    rb.unit.setCurrentIndex(rb.unit.findData('fps'))
    check("switches to frames per second", rb.spin.suffix() == ' fps'
          and abs(rb.spin.value() - 2.0) < 1e-9 and win.rate_unit == 'fps')
    rb.spin.setValue(4.0)
    check("4 fps stored as 0.25 s per frame", abs(got[-1] - 0.25) < 1e-9
          and abs(rb.interval - 0.25) < 1e-9)
    check("no faster than the limit", abs(rb.spin.maximum() - 20.0) < 1e-9)

    print("── 11: Log Plot ──")
    from core import frame_log as fl
    from gui.shell.graphics import LogPlot
    rows = [dict(time_s=float(i), frame_index=i, qc_pass=i != 2, ift_mNm=50.0 - i,
                 volume_uL=10.0 + 0.1 * i) for i in range(5)]
    log = SimpleNamespace(columns=['time_s', 'frame_index', 'ift_mNm', 'volume_uL'], rows=rows)
    lp = LogPlot()
    lp.set_log(log)
    check("IFT and volume chosen first", lp.combo.currentData() == LogPlot.BOTH)
    check("both drawn, volume on the right axis", len(lp.pts.points()) == 5
          and len(lp.pts2.points()) == 5 and lp.plot.plotItem.getAxis('right').isVisible())
    lp.combo.setCurrentIndex(lp.combo.findData('volume_uL'))
    check("a single column hides the right axis", len(lp.pts2.points()) == 0
          and not lp.plot.plotItem.getAxis('right').isVisible())
    lp.set_log(SimpleNamespace(columns=['time_s', 'volume_uL'], rows=rows))
    check("no IFT: no combined choice", lp.combo.findData(LogPlot.BOTH) < 0)

    print("── 12: Simulation snapshot at a typed time ──")
    from gui.shell.result_views import SimulationView
    steps = list(np.arange(1, 101) * 10.0)              # 100 steps of 10 s
    stored = (100.0, 500.0, 1000.0)
    P = [[0, 0], [1e-3, 0], [0, 1e-3], [1e-3, 1e-3]]
    sim = dict(snapshots=[dict(step=int(t // 10), t=t, C=[t / 1000] * 4) for t in stored],
               model=dict(points=P, triangles=[[0, 1, 2], [1, 3, 2]]),
               curve=dict(t=steps, Cs=[0.5] * 100, gamma=None), measured=None,
               settings=dict(D=1e-9, k=1e-5, source='manual', stepping='graded'),
               result=dict(kD=1.0, simulated_s=1000.0, n_steps=100, Cs_final=0.9,
                           C_min=0.0, C_max=1.0, gamma_final=None), diagnostics=[])
    sv = SimulationView()
    sv.set_data(sim)
    asked = []
    sv.compute_requested.connect(asked.append)
    sv.time_edit.setText('500')
    sv.go_to_time()
    check("a stored time is shown", sv.slider.value() == 1 and sv.compute_btn.isHidden()
          and sv.time_info.text() == '')
    sv.time_edit.setText('503')
    sv.go_to_time()
    check("a time near a stored step snaps to it", sv.slider.value() == 1
          and 'Nearest solver step' in sv.time_info.text())
    sv.time_edit.setText('720')
    sv.go_to_time()
    check("an unstored time shows the nearest and offers Compute", sv.slider.value() == 1
          and not sv.compute_btn.isHidden() and 'Not stored' in sv.time_info.text())
    sv.compute_btn.click()
    check("Compute asks for a re-run at that time", asked == [720.0])
    sim2 = dict(sim, snapshots=sorted(sim['snapshots'] + [dict(step=72, t=720.0, C=[0.7] * 4)],
                                      key=lambda s: s['t']))
    sv.set_data(sim2)
    check("after the re-run the asked time is shown", sv.slider.value() == 2
          and sv.time_edit.text() == '720' and sv.compute_btn.isHidden())
    sv.time_edit.setText('5000')
    sv.go_to_time()
    check("a time outside the run is refused", 'Outside the run' in sv.time_info.text())
    sv.time_edit.setText('720')
    sv.go_to_time()
    sv._asked_t = 300.0
    sv.compute_btn.click()
    sv.compute_ended()
    check("a failed re-run says so", sv.time_info.text() == 'Not computed.')

    print("── 14: result tabs on a grid of equal cells; kD colour scale ──")
    from PyQt6.QtCore import QPointF
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QToolTip
    from gui.shell.graphics import GraphicsArea
    yv = YangView()
    yv.resize(950, 640)
    yv.show()
    yv.set_data(yang_data([0.1, 0.5, 1, 2, 3, 4, 5, 6]))
    app.processEvents()

    def cell(view, name):
        i = view.grid.indexOf(view.tiles[name])
        return view.grid.getItemPosition(i) if i >= 0 else None
    check("Yang default: E-D across the bottom row", cell(yv, 'curves') == (1, 0, 1, 2)
          and cell(yv, 'fit') == (0, 0, 1, 1) and cell(yv, 'err') == (0, 1, 1, 1))
    check("equal cells", [yv.grid.rowStretch(r) for r in (0, 1)] == [1, 1]
          and [yv.grid.columnStretch(c) for c in (0, 1)] == [1, 1])
    yv.set_layout('3 in a row')
    check("a layout preset moves the tiles", cell(yv, 'curves') == (0, 2, 1, 1)
          and yv.grid.rowStretch(1) == 0 and yv.grid.columnStretch(2) == 1)
    yv.set_layout('2 + 1 wide')
    QTest.mouseDClick(yv.curves.viewport(), Qt.MouseButton.LeftButton)
    app.processEvents()
    check("double-click fills the tab with that plot", yv.maxed == 'curves'
          and yv.tiles['fit'].isHidden() and yv.tiles['err'].isHidden()
          and cell(yv, 'curves') == (0, 0, 1, 1))
    QTest.mouseDClick(yv.curves.viewport(), Qt.MouseButton.LeftButton)
    app.processEvents()
    check("double-click again puts it back", yv.maxed is None
          and not yv.tiles['fit'].isHidden() and cell(yv, 'curves') == (1, 0, 1, 2))
    check("no legend, a kD colour scale", not yv.curves.plotItem.legend.isVisible()
          and yv.scale.values == [0.1, 0.5, 1, 2, 3, 4, 5, 6] and yv.scale.best == yv.best_kD)
    yv.list.item(2).setCheckState(Qt.CheckState.Unchecked)
    check("hidden curve greyed on the scale", yv.scale.hidden == {1.0})
    yv._set_all(True)
    lx, E, Dm, Em = yv.data_of[3.0]
    k = int(np.nanargmin(E))
    vb = yv.curves.plotItem.vb
    yv._hover(vb.mapViewToScene(QPointF(lx[k], E[k])))
    tip = QToolTip.text()
    check("hovering a curve names its kD", tip.startswith('kD = 3') and 'lowest E' in tip, tip)
    ini = QSettings(str(tmp / 'layouts.ini'), QSettings.Format.IniFormat)
    ga = GraphicsArea()
    ga.use_store(ini)
    yg = ga.results['yang']
    yg._layout_picked(yg.layout_names().index('Stacked'))
    ga2 = GraphicsArea()
    ga2.use_store(ini)
    check("layout choice remembered per tab", ga2.results['yang'].layout_box.currentText()
          == 'Stacked' and ga2.results['bagalkot'].layout_box.currentText() == 'Side by side')
    check("one-plot tab shows no Layout box",
          ga.results['swelling'].layout_box.isHidden() and len(ga.results['simulation'].LAYOUTS) == 3)

    lv = ga.live
    lv.attach(ga.frames, ga.logplot)
    check("Live tab: frame and plot are tiles with a Layout box",
          lv.tiles['frame'].isAncestorOf(ga.frames) and lv.tiles['plot'].isAncestorOf(ga.logplot)
          and lv.layout_names() == ['Side by side', 'Stacked'] and not lv.layout_box.isHidden())
    ga.addTab(ga.frames, 'Frame')               # the Frame tab takes the viewer back
    QTest.mouseDClick(ga.frames.view.viewport(), Qt.MouseButton.LeftButton)
    check("a double-click on the Frame tab doesn't touch the Live layout", lv.maxed is None
          and not lv.tiles['frame'].isAncestorOf(ga.frames))
    lv.attach(ga.frames, ga.logplot)
    QTest.mouseDClick(ga.logplot.plot.viewport(), Qt.MouseButton.LeftButton)
    check("back in Live; double-click enlarges the plot", lv.maxed == 'plot'
          and lv.tiles['frame'].isAncestorOf(ga.frames))

    print("── 15: one warning source for page and tree; shared building blocks ──")
    from gui.shell import nodes as N

    def ydoc(**r):
        return dict(data=dict(result=dict(dict(success=True, D_well_determined=True), **r)))
    ok_doc = ydoc(message='global minimum found in the interior of the search box')
    check("a success note is not a warning", N.result_warnings(None, 'yang', ok_doc) == []
          and N.result_caveat('yang', ok_doc) == '')
    bad = ydoc(success=False, D_well_determined=False,
               message='the best fit is at the edge of your kD range')
    ws = N.result_warnings(None, 'yang', bad)
    check("page lists every warning; tree gives the first (+ how many more)", len(ws) == 2
          and ws[0].startswith('The best fit is at the edge')
          and N.result_caveat('yang', bad) == ws[0] + ' (+1 more on the page)')
    sim = dict(data=dict(result=dict(has_nan=False, has_negative=False, calibrated=False)))
    check("simulation without gamma(c) warns on page and tree alike",
          N.result_warnings(None, 'simulation', sim) == [N.result_caveat('simulation', sim)])
    import gui.shell.kit as kit
    check("page blocks live in kit", all(hasattr(kit, n) for n in (
        'Section', 'actions', 'button_row', 'page_button', 'number_box', 'sci_edit',
        'parse_sci', 'parse_sci_list', 'warnings_section', 'callout', 'note')))
    from gui.shell.graphics import LogPlot as LP
    lp2 = LP()
    check("Log Plot made with make_plot (no empty title row, shared style)",
          not lp2.plot.plotItem.titleLabel.isVisible()
          and lp2.plot.getAxis('left').pen().color().name() == '#555555'
          and lp2.plot.minimumWidth() == 220)

    print("── 13: log axes keep their labels apart when zoomed ──")
    from gui.shell.result_views import LogAxis, sci_label
    check("labels in powers of ten", sci_label(1e-9) == '10⁻⁹'
          and sci_label(3.5e-9) == '3.5×10⁻⁹' and sci_label(2.0) == '2')
    la = LogAxis('bottom')
    la.setLogMode(True)
    for lo, hi, width in ((1e-11, 1e-8, 700), (1e-9, 1e-8, 700), (1.5e-9, 7e-9, 700),
                          (3.0e-9, 3.6e-9, 700), (3.0e-9, 3.6e-9, 250), (1e-12, 1e-6, 250)):
        levels = la.tickValues(np.log10(lo), np.log10(hi), width)
        pos = sorted(v for sp, vals in levels
                     for v, s in zip(vals, la.tickStrings(vals, 1, sp)) if s)
        # Labels sit at least LABEL_PX apart on screen (the old axis put ~9 in 20 px).
        px = np.diff(pos) / (np.log10(hi) - np.log10(lo)) * width
        check(f"{lo:.2g} to {hi:.2g} on {width} px: labels apart", len(pos) >= 2
              and (not len(px) or px.min() >= 0.9 * LogAxis.LABEL_PX),
              f"{len(pos)} labels, closest {px.min() if len(px) else 0:.0f} px")
    w.discard_draft()
    w.close()


if __name__ == "__main__":
    main()
