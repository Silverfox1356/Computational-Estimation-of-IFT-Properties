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
    check("yang waits for a calibration", w.statuses['yang'][0] == 'off'
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
    check("swelling complete in the tree", w.statuses['swelling'][0] == 'done')
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
    button('Rebuild log with this threshold').click()
    check("rebuild finished", wait_job())
    check("results marked out of date", w.statuses['swelling'][0] == 'stale'
          and w.statuses['bagalkot'][0] == 'stale' and w.statuses['results'][0] == 'stale',
          f"{w.statuses['swelling'][0]}, {w.statuses['bagalkot'][0]}")

    print("── Yang ──")
    calib = tmp / 'calib.csv'
    calib.write_text(CALIB, encoding='utf-8')
    w.load_calibration(str(calib))
    n_qc = sum(r['qc_pass'] for r in w.folder.log.rows)
    if w.statuses['yang'][0] == 'off':
        check("yang explains why it can't run", 'QC' in w.statuses['yang'][1] or n_qc < 10,
              w.statuses['yang'][1])
    else:
        from PyQt6.QtWidgets import QLineEdit, QRadioButton
        w.select_node('yang')
        page = w.settings.widget()
        next(r for r in page.findChildren(QRadioButton) if r.text() == 'Manual').setChecked(True)
        edits = page.findChildren(QLineEdit)
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
        check("Yang Fit and Mesh tabs", {'Yang Fit', 'Mesh'} <= set(tabs()), str(tabs()))
        D_before = doc['data']['result']['D'] if doc else None
        w.select_node('yang')
        button('Run').click()
        pump(2.0)
        w.cancel_job()
        check("second Yang run stopped", wait_job())
        doc = w.folder.load_analysis('yang')
        check("stopped run kept the earlier result", doc is not None
              and doc['data']['result']['D'] == D_before)
    check("no unexpected warnings", not warnings, '; '.join(map(str, warnings)))
    w.close()


if __name__ == "__main__":
    main()
