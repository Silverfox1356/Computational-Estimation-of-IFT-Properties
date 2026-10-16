"""
Drop Analysis Studio: the usability fixes asked for after step 5, on Qt's
offscreen platform (set QT_QPA_PLATFORM=windows to also check the spin-box
arrows under the real Windows style).

    1, 2  each side panel hides and comes back on its own (activity bar);
          dragging a divider resizes only the two panels beside it, and the
          Settings panel can't grow over the Graphics area
    3     every Settings page fits a narrow panel without a horizontal
          scroll bar; Messages wrap
    4     dialogs and popups stay light when the application palette is dark
          (Windows dark mode)
    5, 6  dropdowns and spin boxes use Qt's Fusion style; arrows click where
          they are drawn; dropdown items highlight on hover
    7     Next walks the steps in order
    8     Load Calibration from the ribbon and from the page opens the file
    9     Single-Frame IFT: the reference frame before logging, any logged
          frame, and an image file taken with the same set-up

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_usability.py      (about 30 s)
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from _synthetic_video import render_drop
from test_analyses import CALIB

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


def main():
    tmp = Path(tempfile.mkdtemp(prefix='studio usability '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    from PyQt6.QtCore import QPoint, QSettings, Qt
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFileDialog,
                                 QMessageBox, QStyle, QStyleOptionSpinBox)
    app = QApplication.instance() or QApplication(sys.argv)
    warnings = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    from gui.shell import nodes
    from gui.shell.main_window import PANELS, StudioWindow

    ini = QSettings(str(tmp / 'studio.ini'), QSettings.Format.IniFormat)
    w = StudioWindow(settings_store=ini)
    w.resize(1280, 800)
    w.show()

    def pump(seconds=0.05):
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)

    def wait_job(timeout=120):
        t = time.time()
        while w.job_running() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.02)
        pump(0.1)
        return not w.job_running()

    def sizes():
        return w.split.sizes() + w.right.sizes()

    pump(0.2)
    # ------------------------------------------------------------ 1, 2: panels
    print("── panels ──")
    t0, s0, r0, g0, o0 = sizes()
    check("default layout", abs(t0 - 250) <= 2 and abs(s0 - 340) <= 2, str(sizes()))
    check("three panel buttons on the activity bar", set(w.panel_actions) == set(PANELS))
    w.panel_actions['tree'].trigger()
    pump()
    t1, s1, r1, *_ = sizes()
    check("Analysis Builder hides on its own", w.panels['tree'].isHidden()
          and not w.panels['settings'].isHidden() and s1 == s0, str(sizes()))
    w.panel_actions['tree'].trigger()
    pump()
    check("and comes back at its size", not w.panels['tree'].isHidden()
          and abs(w.split.sizes()[0] - t0) <= 2 and abs(w.split.sizes()[1] - s0) <= 2, str(sizes()))
    w.panel_actions['settings'].trigger()
    pump()
    check("Settings hides on its own", w.panels['settings'].isHidden()
          and abs(w.split.sizes()[0] - t0) <= 2)
    w.panel_actions['settings'].trigger()
    w.panel_actions['output'].trigger()
    pump()
    check("Output hides on its own", w.panels['output'].isHidden()
          and not w.panels['settings'].isHidden())
    w.panel_actions['output'].trigger()
    pump()
    check("Output back", not w.panels['output'].isHidden() and abs(w.right.sizes()[1] - o0) <= 2,
          str(sizes()))

    pos = sum(w.split.sizes()[:2]) + w.split.handleWidth()   # right edge of Settings
    w.split.moveSplitter(pos + 120, 2)
    pump()
    t2, s2, r2, *_ = sizes()
    check("widening Settings leaves the Analysis Builder alone", t2 == t0 and s2 > s0,
          f"{t0}->{t2}, {s0}->{s2}")
    w.split.moveSplitter(w.split.width(), 2)
    pump()
    t3, s3, r3, *_ = sizes()
    check("Settings stops at its maximum", s3 <= PANELS['settings'][4] and t3 == t0, str(sizes()))
    check("Graphics keeps its room", w.graphics.width() >= 340, f"{w.graphics.width()} px")
    w.split.moveSplitter(t0 + 60, 1)                         # divider Builder | Settings
    pump()
    t4, s4, r4, *_ = sizes()
    check("Builder divider moves only Builder and Settings", r4 == r3 and t4 > t0, str(sizes()))
    w.reset_desktop()
    pump()

    # ------------------------------------------------------------ 3: narrow pages
    print("── narrow Settings ──")
    img, _, _ = render_drop(0.30, 70)
    png = tmp / 'drop.png'
    cv2.imwrite(str(png), img)
    w.new_session('image', path=str(png))
    w.split.moveSplitter(t0 + 1, 2)                          # Settings as narrow as it goes
    pump()
    narrow = w.split.sizes()[1]
    check("Settings can be narrow", narrow <= PANELS['settings'][3] + 2, f"{narrow} px")

    def wide_pages():
        out = []
        for n in nodes.NODES:
            w.select_node(n.key)
            pump(0.02)
            sb = w.settings.horizontalScrollBar()
            if sb.maximum() > 0:
                out.append(f"{n.key} +{sb.maximum()}px")
        return out
    if app.platformName() == 'windows':
        wide = wide_pages()
        check("every set-up page fits without sideways scrolling", not wide, ', '.join(wide))
    else:
        # The offscreen platform has no Segoe UI: its fallback font is about
        # twice as wide, so page widths only mean something on Windows.
        print("  (page widths checked only with QT_QPA_PLATFORM=windows)")
    check("Messages wrap long lines", w.messages.lineWrapMode() == w.messages.LineWrapMode.WidgetWidth)

    # ------------------------------------------------------------ 5, 6: controls
    print("── spin boxes and dropdowns ──")
    w.select_node('fluids')
    pump()
    page = w.settings.widget()
    spins = page.findChildren(QAbstractSpinBox)
    combos = page.findChildren(QComboBox)
    check("spin boxes and dropdowns use the Fusion style",
          spins and combos and all(x.testAttribute(Qt.WidgetAttribute.WA_SetStyle)
                                   for x in spins + combos), f"{len(spins)} spin, {len(combos)} combo")
    misses = []
    for s in spins:
        opt = QStyleOptionSpinBox()
        s.initStyleOption(opt)
        for sc, sign in ((QStyle.SubControl.SC_SpinBoxUp, 1), (QStyle.SubControl.SC_SpinBoxDown, -1)):
            r = s.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, opt, sc, s)
            before = s.value()
            QTest.mouseClick(s, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, r.center())
            if (s.value() - before) * sign <= 0 and not (s.value() in (s.minimum(), s.maximum())):
                misses.append(f"{s.objectName() or type(s).__name__} {'up' if sign > 0 else 'down'}")
            s.setValue(before)
    check("arrow buttons click where they are drawn", not misses, ', '.join(misses))
    combo = combos[0]
    combo.setCurrentIndex(0)
    combo.showPopup()
    pump()
    view = combo.view()
    rect = view.visualRect(view.model().index(1, 0))
    QTest.mouseMove(view.viewport(), rect.center())
    pump()
    hovered = view.grab().toImage().pixelColor(rect.left() + 3, rect.center().y())
    combo.hidePopup()
    pump()
    check("dropdown items highlight on hover", hovered.name() != '#ffffff'
          and hovered.lightness() > 200, hovered.name())
    check("dropdowns can shrink below their longest entry",
          all(c.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
              for c in combos))

    # ------------------------------------------------------------ 4: dark mode
    print("── dialogs under a dark palette ──")
    dark = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        dark.setColor(role, QColor('#202020'))
    old_pal = app.palette()
    app.setPalette(dark)
    box = QMessageBox(QMessageBox.Icon.Warning, 'Test', 'Black text must be readable.',
                      QMessageBox.StandardButton.Ok, w)
    box.show()
    pump()
    bg = box.grab().toImage().pixelColor(5, 5)
    check("a message box is light", bg.lightness() > 200, bg.name())
    box.close()
    menu = w.ribbon.file_menu
    menu.popup(w.mapToGlobal(QPoint(10, 10)))
    pump()
    bg = menu.grab().toImage().pixelColor(menu.width() // 2, menu.height() - 3)
    check("the File menu is light", bg.lightness() > 200, bg.name())
    menu.close()
    combo = combos[0]
    combo.showPopup()
    pump()
    view = combo.view()
    bg = view.grab().toImage().pixelColor(view.width() - 4, view.height() - 4)
    check("a dropdown list is light", bg.lightness() > 200, bg.name())
    combo.hidePopup()
    app.setPalette(old_pal)
    w.reset_desktop()
    pump()

    # ------------------------------------------------------------ 7: Next
    print("── Next ──")
    w.select_node('session')
    seen = []
    for _ in range(6):
        w.a_next.trigger()
        pump(0.02)
        seen.append(w.settings.key)
    check("Next walks the set-up steps in order",
          seen == ['fluids', 'calibration', 'crop', 'threshold', 'scale', 'baseline'], str(seen))

    # ------------------------------------------------------------ 8: calibration
    print("── Load Calibration ──")
    calib = tmp / 'calib.csv'
    calib.write_text(CALIB, encoding='utf-8')
    asked = []
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (asked.append(1), (str(calib), ''))[1])
    w.a_calib.trigger()
    pump()
    check("ribbon button opens the file dialog and loads", asked and w.draft.calibration_path == str(calib),
          str(w.draft.calibration_path))
    w.draft.calibration_path = None
    w.select_node('calibration')
    pump()
    next(b for b in w.settings.widget().findChildren(type(w.settings.widget().findChildren(
        __import__('PyQt6.QtWidgets', fromlist=['QPushButton']).QPushButton)[0]))
         if b.text() == 'Load...').click()
    pump()
    check("page button loads too", w.draft.calibration_path == str(calib) and len(asked) == 2)

    # ------------------------------------------------------------ 9: single frame
    print("── Single-Frame IFT ──")
    d = w.draft
    px, _ = d.measure_box(110, 5, 100, 40, 'needle')
    d.set_scale_from_length(px, 0.8, 'needle width', '')
    d.auto_baseline()
    w.draft_changed()
    check("node ready once scale and baseline are set", w.statuses['frame_ift'][0] == 'ready',
          w.statuses['frame_ift'][1])
    w.select_node('frame_ift')
    pump()
    from PyQt6.QtWidgets import QPushButton
    next(b for b in w.settings.widget().findChildren(QPushButton) if b.text() == 'Measure IFT').click()
    check("reference frame measured", wait_job() and d.preview is not None and d.preview['fit_ok'])
    check("full result shown on the page", any(
        'Interfacial tension' in lbl.text() for lbl in w.settings.widget().findChildren(
            __import__('PyQt6.QtWidgets', fromlist=['QLabel']).QLabel)))
    check("tree marks it complete", w.statuses['frame_ift'][0] == 'done')

    dest = tmp / 'image session'
    check("logged", w.start_logging(path=dest) and wait_job() and w.folder is not None)
    row = w.folder.log.rows[0]
    w.select_node('frame_ift')
    pump()
    check("ribbon Measure Frame enabled for a session", w.a_measure.isEnabled())
    next(b for b in w.settings.widget().findChildren(QPushButton) if b.text() == 'Measure IFT').click()
    check("logged frame measured again", wait_job() and w.frame_fit is not None)
    rec = w.frame_fit['rec']
    check("same IFT as its log row", abs(rec['ift_mNm'] - row['ift_mNm']) <= 1e-5 * row['ift_mNm'],
          f"{rec['ift_mNm']:.6g} vs {row['ift_mNm']:.6g}")
    check("fitted outline drawn on the frame", len(w.graphics.frames.view._overlays) >= 4)
    check("result section on the page", any(
        lbl.text().startswith('Result: Frame') for lbl in w.settings.widget().findChildren(
            __import__('PyQt6.QtWidgets', fromlist=['QToolButton']).QToolButton)))
    n_rows = len(w.folder.log)

    other = tmp / 'other.png'
    cv2.imwrite(str(other), render_drop(0.32, 72)[0])
    w.measure_image_file(str(other))
    check("image file measured", wait_job() and w.frame_fit['label'] == 'other.png'
          and w.frame_fit['rec']['fit_ok'])
    check("log unchanged", len(w.folder.log) == n_rows)
    small = tmp / 'small.png'
    cv2.imwrite(str(small), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    n_warn = len(warnings)
    w.measure_image_file(str(small))
    wait_job()
    check("an image of another size is refused", len(warnings) == n_warn + 1
          and 'px' in str(warnings[-1]), str(warnings[-1:]))
    w.close()


if __name__ == "__main__":
    main()
