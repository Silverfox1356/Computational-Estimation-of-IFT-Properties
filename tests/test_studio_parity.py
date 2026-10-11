"""
Drop Analysis Studio: features ported from the old windows that are not
analyses of their own (the gap list of 2026-10-12):

    needle inner diameter   draft field, stored in session.json, changeable
                            after logging; marks model results out of date
    presets                 Save / Save as / Delete / Import from the Fluids &
                            Needle page (the Pendant window's Settings dialog)
    export                  every stored analysis as CSV (+ sweep), as the
                            Bagalkot window's Export results
    calibration             Remove from an open session (the Pendant Clear)
    sphere scale            box around a reference sphere (the Pendant
                            window's Sphere calibration)

Runs against a throwaway preset folder (IFT_PRESETS_HOME), never the real one.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_parity.py      (about 30 s)
"""
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
TMP = tempfile.mkdtemp(prefix='studio parity ')
os.environ['IFT_PRESETS_HOME'] = str(Path(TMP) / 'presets home')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from _synthetic_video import render_drop
from core import analyses as A
from core import presets as P
from core.image_processing import measure_reference
from core.result_export import export_result
from core.session import SessionError, SessionFolder
from core.session_draft import DraftError, SessionDraft
from test_analyses import CALIB, ROOT, session

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


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        print(f"       (raised {type(e).__name__}: {e})")
    return False


def main():
    tmp = Path(TMP)
    try:
        core_checks(tmp)
        studio_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def one_image(tmp, name):
    f = SessionFolder.create(tmp / name, session(kind='image'))
    f.log_image(render_drop(0.30, 70)[0])
    f.finish()
    calib = tmp / f'{name}_calib.csv'
    calib.write_text(CALIB, encoding='utf-8')
    f.set_calibration(calib)
    return f


def core_checks(tmp):
    print("── needle inner diameter ──")
    img = tmp / 'drop.png'
    cv2.imwrite(str(img), render_drop(0.30, 70)[0])
    d = SessionDraft.from_image(str(img))
    d.set_needle(0.8, id_mm=0.5)
    check("draft keeps the inner diameter", d.needle.get('id_mm') == 0.5)
    d.set_needle(0.9)
    check("left out = kept", d.needle.get('id_mm') == 0.5)
    d.set_needle(0.9, id_mm=None)
    check("None = automatic", 'id_mm' not in d.needle)
    check("ID >= OD refused", raises(lambda: d.set_needle(0.8, id_mm=0.8), DraftError))

    f = one_image(tmp, 'id')
    check("session refuses ID >= OD", raises(lambda: f.set_needle_id(0.8), SessionError))
    A.run_simulation(f, D=2e-9, k=3e-5, duration_s=60)
    r_auto = f.load_analysis('simulation')['data']['model']['domain']['r_inner_m']
    check("automatic ID is 70 % of OD", abs(2 * r_auto * 1e3 - 0.7 * 0.8) < 1e-9,
          f"{2 * r_auto * 1e3:.4f} mm")
    f.set_needle_id(0.6)
    check("stored in session.json", SessionFolder.open(f.path).session.needle['id_mm'] == 0.6)
    check("simulation now out of date", f.analysis_out_of_date('simulation'))
    A.run_simulation(f, D=2e-9, k=3e-5, duration_s=60)
    r = f.load_analysis('simulation')['data']['model']['domain']['r_inner_m']
    check("model built with the new ID", abs(2 * r * 1e3 - 0.6) < 1e-9
          and not f.analysis_out_of_date('simulation'), f"{2 * r * 1e3:.4f} mm")

    print("── export ──")
    sim = f.load_analysis('simulation')['data']
    files = export_result('simulation', sim, tmp / 'sim.csv', source='id')
    rows = list(csv.reader(l for l in open(files[0], encoding='utf-8') if not l.startswith('#')))
    head = [l for l in open(files[0], encoding='utf-8') if l.startswith('#')]
    check("simulation CSV: curve table", rows[0] == ['time_s', 'Cs_over_csat', 'ift_mNm']
          and len(rows) - 1 == len(sim['curve']['t']), f"{len(rows) - 1} rows")
    check("settings and results in the header", any(l.startswith('# setting.D,') for l in head)
          and any(l.startswith('# kD,') for l in head))
    vol = tmp / 'v.csv'
    shutil.copy(ROOT / 'data' / 'volume_curve_template.csv', vol)
    f.set_volume_curve(vol)
    bag = A.run_bagalkot(f, curve='file')
    files = export_result('bagalkot', bag, tmp / 'bag.csv')
    check("Bagalkot: curve file and sweep file", len(files) == 2 and files[1].name == 'bag_sweep.csv'
          and open(files[1], encoding='utf-8').readline().strip() == 'D_m2_per_s,F_percent,SF')

    print("── calibration removal ──")
    f.clear_calibration()
    g = SessionFolder.open(f.path)
    check("calibration gone from folder and session.json",
          g.calibration_path is None and not (f.path / 'calibration.csv').exists())
    check("simulation still runs (Cs only)", A.run_simulation(g, D=2e-9, k=3e-5, duration_s=30)
          ['result']['calibrated'] is False)

    print("── sphere scale ──")
    roi = np.full((120, 120, 3), 230, np.uint8)
    cv2.circle(roi, (60, 60), 40, (20, 20, 20), -1)
    px, marker = measure_reference(roi, 'sphere')
    check("sphere diameter measured", px is not None and abs(px - 80) < 3 and marker[0] == 'circle',
          f"{px:.1f} px")


def studio_checks(tmp):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QInputDialog,
                                 QMessageBox, QPushButton)
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning = staticmethod(lambda *a, **k: print('       WARNING', a[2:3]))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    names = iter(['My needle', 'My fluid'])
    QInputDialog.getText = staticmethod(lambda *a, **k: (next(names), True))
    from gui.shell.main_window import StudioWindow

    w = StudioWindow(settings_store=QSettings(str(tmp / 's.ini'), QSettings.Format.IniFormat))
    w.resize(1280, 800)
    w.show()

    def pump(s=0.05):
        t = time.time()
        while time.time() - t < s:
            app.processEvents()
            time.sleep(0.01)

    def button(text):
        for b in w.settings.widget().findChildren(QPushButton):
            if b.text() == text:
                return b
        raise AssertionError(f"no button '{text}' on the {w.settings.key} page")

    print("── Studio: needle inner diameter and presets ──")
    img = tmp / 'drop2.png'
    cv2.imwrite(str(img), render_drop(0.30, 70)[0])
    w.new_session('image', path=str(img))
    w.select_node('fluids')
    page = w.settings.widget()
    idd = next(s for s in page.findChildren(QDoubleSpinBox)
               if s.specialValueText().startswith('automatic'))
    check("inner diameter shows automatic", idd.value() == 0 and 'id_mm' not in w.draft.needle)
    idd.setValue(0.4)
    pump()
    check("inner diameter reaches the draft", w.draft.needle.get('id_mm') == 0.4)
    w.select_node('fluids')
    button('Save as...').click()           # needle section comes first
    pump()
    p = P.find_preset('needle', 'My needle')
    check("needle preset saved with the ID", p is not None and p.values['id_mm'] == 0.4
          and not p.builtin)
    combos = w.settings.widget().findChildren(QComboBox)
    check("new preset chosen on the page", combos[0].currentText() == 'My needle')
    saves = [b for b in w.settings.widget().findChildren(QPushButton) if b.text() == 'Save']
    check("Save enabled for a user preset", saves[0].isEnabled())
    deletes = [b for b in w.settings.widget().findChildren(QPushButton) if b.text() == 'Delete']
    deletes[0].click()
    pump()
    check("preset deleted", P.find_preset('needle', 'My needle') is None)
    exported = tmp / 'needle.json'
    P.export_preset(P.list_presets('needle')[0][0], exported)
    data = exported.read_text(encoding='utf-8').replace(
        P.list_presets('needle')[0][0].name, 'Imported needle')
    exported.write_text(data, encoding='utf-8')
    w.preset_import(str(exported))
    check("preset imported", P.find_preset('needle', 'Imported needle') is not None)
    w.discard_draft() if hasattr(w, 'discard_draft') else w.close_current()

    print("── Studio: open session ──")
    f = one_image(tmp, 'studio')
    A.run_simulation(f, D=2e-9, k=3e-5, duration_s=60)
    check("session opens", w.open_session(str(f.path)))
    w.select_node('fluids')
    idd = next(s for s in w.settings.widget().findChildren(QDoubleSpinBox)
               if s.specialValueText().startswith('automatic'))
    idd.setValue(0.5)
    button('Apply inner diameter').click()
    pump()
    check("ID applied to the session", w.folder.session.needle.get('id_mm') == 0.5)
    check("simulation marked out of date, saying why", w.statuses['simulation'][0] == 'stale'
          and 'inner diameter' in w.statuses['simulation'][1], str(w.statuses['simulation']))
    w.export_result('simulation', str(tmp / 'from_studio.csv'))
    check("export from the Studio", (tmp / 'from_studio.csv').is_file())
    w.select_node('simulation')
    check("Export button on the page", button('Export CSV...') is not None)
    w.select_node('calibration')
    button('Remove').click()
    pump()
    check("calibration removed from the open session", w.folder.calibration_path is None
          and w.statuses['calibration'][0] == 'ready')
    w.close()


if __name__ == "__main__":
    main()
