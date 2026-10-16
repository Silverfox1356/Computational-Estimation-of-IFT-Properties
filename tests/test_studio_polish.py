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
    from gui.shell.pages import warnings_section
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
    w.discard_draft()
    w.close()


if __name__ == "__main__":
    main()
