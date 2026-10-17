"""
Drop Analysis Studio, step 4 (gui.shell): run the analyses of a session
from their pages on Qt's offscreen platform.  The swelling check and
Bagalkot run from their Run buttons, their result tabs appear, the tree
marks them complete, a threshold rebuild marks them out of date, and a Yang
run can be stopped.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_analyses.py      (about 1 min)
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

from test_analyses import CALIB, series_folder

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
    tmp = Path(tempfile.mkdtemp(prefix='studio analyses '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton
    app = QApplication.instance() or QApplication(sys.argv)
    warnings = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    from gui.shell import nodes
    from gui.shell.main_window import StudioWindow

    folder = series_folder(tmp)
    ini = QSettings(str(tmp / 'studio.ini'), QSettings.Format.IniFormat)
    w = StudioWindow(settings_store=ini)
    w.resize(1280, 800)
    w.show()

    def pump(seconds=0.05):
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)

    def wait_job(timeout=300):
        t = time.time()
        while w.job_running() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.02)
        pump(0.1)
        return not w.job_running()

    def button(text):
        for b in w.settings.widget().findChildren(QPushButton):
            if b.text().startswith(text):
                return b
        raise AssertionError(f"no button '{text}' on the {w.settings.key} page")

    def tabs():
        return [w.graphics.tabText(i) for i in range(w.graphics.count())]

    check("session opens", w.open_session(str(folder.path)))
    check("analyses ready", all(w.statuses[k][0] == 'ready' for k in ('swelling', 'bagalkot')))
    check("yang waits for a calibration", w.statuses['yang'][0] == 'input'
          and 'calibration' in w.statuses['yang'][1])
    check("ribbon analysis buttons enabled", w.a_swell.isEnabled() and w.a_bagalkot.isEnabled())
    errors = []
    for n in nodes.NODES:
        try:
            w.select_node(n.key)
            pump(0.01)
        except Exception as e:
            errors.append(f"{n.key}: {e}")
    check("every page builds before any run", not errors, '; '.join(errors))

    print("── swelling ──")
    w.select_node('swelling')
    button('Run').click()
    check("swelling finished", wait_job())
    sw_doc = w.folder.load_analysis('swelling')
    sw_why = nodes.result_caveat('swelling', sw_doc)
    check("swelling in the tree: complete, or a warning matching the page",
          w.statuses['swelling'] == (('warn', sw_why) if sw_why
                                     else ('done', 'Results stored in the session.')),
          f"{w.statuses['swelling']}")
    check("Swelling tab added", 'Swelling' in tabs(), str(tabs()))
    check("result shown on the page", any(b.text().startswith('Go to Diffusivity (Bagalkot)')
                                          for b in w.settings.widget().findChildren(QPushButton)))
    check("results node complete", w.statuses['results'][0] == 'done')

    print("── Bagalkot ──")
    w.select_node('bagalkot')
    button('Run').click()
    check("Bagalkot finished", wait_job())
    check("Bagalkot complete", w.statuses['bagalkot'][0] == 'done')
    check("Bagalkot Fit tab shown", w.graphics.currentWidget() is w.graphics.results['bagalkot'])
    w.select_node('results')
    check("results page lists both", sum(b.text() == 'Plot' for b in
                                         w.settings.widget().findChildren(QPushButton)) == 2)

    print("── reopen ──")
    w.close_current()
    w.open_session(str(folder.path))
    check("result tabs come back from the folder", {'Swelling', 'Bagalkot Fit'} <= set(tabs()),
          str(tabs()))

    print("── threshold rebuild ──")
    w.select_node('threshold')
    from PyQt6.QtWidgets import QRadioButton, QSpinBox
    page = w.settings.widget()
    next(r for r in page.findChildren(QRadioButton) if r.text() == 'Fixed level').setChecked(True)
    page.findChildren(QSpinBox)[0].setValue(110)
    button('Rebuild log').click()
    check("rebuild finished", wait_job())
    check("results marked out of date", w.statuses['swelling'][0] == 'stale'
          and w.statuses['bagalkot'][0] == 'stale' and w.statuses['results'][0] == 'stale',
          f"{w.statuses['swelling'][0]}, {w.statuses['bagalkot'][0]}")

    print("── Yang ──")
    calib = tmp / 'calib.csv'
    calib.write_text(CALIB, encoding='utf-8')
    w.load_calibration(str(calib))
    n_qc = sum(r['qc_pass'] for r in w.folder.log.rows)
    if w.statuses['yang'][0] in ('off', 'input'):
        check("yang explains why it can't run", 'QC' in w.statuses['yang'][1] or n_qc < 10,
              w.statuses['yang'][1])
    else:
        from PyQt6.QtWidgets import QLineEdit, QRadioButton
        w.select_node('yang')
        page = w.settings.widget()
        next(r for r in page.findChildren(QRadioButton) if r.text() == 'Manual range').setChecked(True)
        edits = page.findChildren(QLineEdit)[-4:]           # the manual range's four boxes
        for e, v in zip(edits, ('1e-10', '1e-8', '0.5', '50')):   # D from, D to, kD from, kD to
            e.setText(v)
        button('Run').click()
        check("Yang ran in the background (model built off the main thread)", wait_job(),
              f"{w.folder.load_analysis('yang') is not None}")
        doc = w.folder.load_analysis('yang')
        check("Yang result stored", doc is not None and doc['data']['settings']['auto'] is False)
        caveat = nodes.result_caveat('yang', doc)
        check("Yang in the tree: complete, or a warning when the fit flags its own result",
              w.statuses['yang'] == (('warn', caveat) if caveat else ('done', 'Results stored in the session.'))
              and w.statuses['yang_model'][0] == 'done', f"{w.statuses['yang'][0]}: {caveat}")
        check("Yang Fit, Simulation and Model tabs",
              {'Yang Fit', 'Simulation', 'Model'} <= set(tabs()), str(tabs()))
        D_before = doc['data']['result']['D'] if doc else None
        w.select_node('yang')
        button('Run').click()
        pump(2.0)
        w.cancel_job()
        check("second Yang run stopped", wait_job())
        doc = w.folder.load_analysis('yang')
        check("stopped run kept the earlier result", doc is not None
              and doc['data']['result']['D'] == D_before)

    print("── Yang on a single image + IFT curve file ──")
    from PyQt6.QtWidgets import QDoubleSpinBox, QLabel, QLineEdit, QRadioButton
    from _synthetic_video import render_drop
    from core.session import SessionFolder
    from test_analyses import ROOT, session
    one = SessionFolder.create(tmp / 'one', session(kind='image'))
    one.log_image(render_drop(0.30, 70)[0])
    one.finish()
    w.close_current()
    check("one-image session opens", w.open_session(str(one.path)))
    check("Yang asks for input (not greyed out)", w.statuses['yang'][0] == 'input',
          str(w.statuses['yang']))
    check("the session itself is not flagged", w.statuses['session'][0] != 'input',
          str(w.statuses['session']))
    w.select_node('yang')

    def radio(text):
        return next(r for r in w.settings.widget().findChildren(QRadioButton)
                    if r.text().startswith(text))
    check("IFT curve file chosen by default", radio('IFT curve file').isChecked())
    check("Run disabled without calibration and curve", not button('Run').isEnabled())
    w.load_calibration(str(calib))
    w.load_ift_curve(str(ROOT / 'data' / 'ift_curve_5.67MPa.csv'))
    check("curve copied into the session", w.folder.ift_curve_path is not None)
    check("Yang ready", w.statuses['yang'][0] == 'ready', str(w.statuses['yang']))
    w.select_node('yang')
    check("Run enabled", button('Run').isEnabled())
    radio('Frame log').setChecked(True)
    check("Run disabled for the frame log route", not button('Run').isEnabled())
    radio('IFT curve file').setChecked(True)
    radio('Manual').setChecked(True)
    for e, v in zip(w.settings.widget().findChildren(QLineEdit)[-4:],
                    ('1e-10', '1e-8', '0.5', '50')):
        e.setText(v)
    button('Run').click()
    check("Yang ran", wait_job())
    doc = w.folder.load_analysis('yang')
    check("stored with the curve file as source", doc is not None
          and doc['data']['settings']['curve'] == 'file'
          and doc['data']['settings']['frame_index'] == 0)
    check("Yang Fit, Simulation and Model tabs",
          {'Yang Fit', 'Simulation', 'Model'} <= set(tabs()), str(tabs()))
    w.select_node('yang')
    labels = [l.text() for l in w.settings.widget().findChildren(QLabel)]
    check("result page names the curve file", 'the file ift_curve.csv' in labels)

    print("── forward simulation ──")
    check("simulation node complete after Yang", w.statuses['simulation'][0] == 'done',
          str(w.statuses['simulation']))
    w.select_node('yang_model')
    labels = [l.text() for l in w.settings.widget().findChildren(QLabel)]
    check("model page lists the FEM checks", labels.count('pass') >= 8, str(labels.count('pass')))
    w.select_node('simulation')
    page = w.settings.widget()
    edits = page.findChildren(QLineEdit)
    edits[0].setText('2e-9')
    edits[1].setText('3e-5')
    radio('Fixed time step').setChecked(True)
    spins = [sp for sp in page.findChildren(QDoubleSpinBox) if sp.suffix() == ' s']
    spins[0].setValue(120)                    # duration
    spins[1].setValue(0.5)                    # time step
    button('Run').click()
    check("simulation ran", wait_job())
    doc = w.folder.load_analysis('simulation')
    st = doc['data']['settings'] if doc else {}
    check("entered values used", st.get('source') == 'manual' and st.get('D') == 2e-9
          and st.get('dt_s') == 0.5 and st.get('duration_s') == 120, str(st))
    check("Simulation tab shown", w.graphics.currentWidget() is w.graphics.results['simulation'])
    sv = w.graphics.results['simulation']
    check("concentration field drawn", sv.canvas is not None and len(sv.canvas.figure.axes) >= 1
          and sv.slider.maximum() >= 5)
    sv.slider.setValue(0)
    check("slider shows an early snapshot", sv.slider_label.text().startswith('1 of'))
    w.select_node('simulation')
    check("result on the page", button('Show the simulation') is not None)

    print("── Bagalkot on a single image + volume curve file ──")
    check("Bagalkot not available without a curve", w.statuses['bagalkot'][0] == 'off',
          str(w.statuses['bagalkot']))
    w.select_node('bagalkot')
    check("Run disabled", not button('Run').isEnabled())
    w.load_volume_curve(str(ROOT / 'data' / 'volume_curve_template.csv'))
    check("Bagalkot ready with the file", w.statuses['bagalkot'][0] == 'ready',
          str(w.statuses['bagalkot']))
    w.select_node('bagalkot')
    check("volume file chosen", radio('Volume curve file').isChecked())
    button('Run').click()
    check("Bagalkot ran", wait_job())
    doc = w.folder.load_analysis('bagalkot')
    check("stored with the file as source", doc is not None
          and doc['data']['settings']['curve'] == 'file')
    check("Bagalkot Fit tab", 'Bagalkot Fit' in tabs(), str(tabs()))
    check("no unexpected warnings", not warnings, '; '.join(map(str, warnings)))
    w.close()


if __name__ == "__main__":
    main()
