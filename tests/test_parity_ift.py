"""
UI overhaul, step 5: the Drop Analysis Studio and the old Pendant window
give the same IFT for the same image.

Both apps call the same fit (core.physics_calcs), so any difference comes
from the code around it: how the image is loaded and stored, the needle
box, the automatic baseline, the settings handed to the fit, and the log.
This test drives the real windows on Qt's offscreen platform:

    old Pendant window: load image -> auto calibration on a needle box ->
                        auto baseline -> Calculate IFT
    Studio:             new session from the image -> needle box ->
                        auto baseline -> Start logging -> log.csv row

and compares scale, baseline, IFT, beta, R0, volume and the DS/DE check.
The Studio's log keeps 6 significant digits, so "same" means equal to
1e-5 relative.

Images: a synthetic drop (always), plus every image in
Academic/parity-images/ that has an entry in REAL_IMAGES, plus any image
paths given on the command line (with the synthetic settings).

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_parity_ift.py [image ...]
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

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


# Settings for each image: the needle box (x, y, w, h) on a straight part of
# the needle, the needle OD and the two densities.  Both apps get the same.
SYNTHETIC = dict(box=(110, 5, 100, 40), od_mm=0.8, rho_in=998.2, rho_out=1.204)

# Real images (gitignored folder).  The CO2/brine picture shows no needle,
# only the neck at the top edge, so its "needle" box is the neck and its
# scale, like the densities (brine in dense CO2), is assumed: the IFT is
# not a physical result, only the same input to both apps.
REAL_IMAGES = {
    'co2-brine-drop.png': dict(box=(190, 0, 430, 40), od_mm=1.6, rho_in=1050.0, rho_out=630.0),
}

# (old results key, Studio log column)
COMPARED = [('ift_mNm', 'ift_mNm'), ('beta', 'beta'), ('R0_mm', 'R0_mm'),
            ('volume_ul', 'volume_uL'), ('ift_dsde_mNm', 'ift_dsde_mNm'),
            ('rmse_px', 'rmse_px'), ('Wo', 'Wo')]
REL_TOL = 1e-5


def main():
    tmp = Path(tempfile.mkdtemp(prefix='parity ift '))
    try:
        run(tmp, sys.argv[1:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp, extra):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    warnings = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.critical = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    from gui.pendant_window import PendantWindow
    from gui.shell.main_window import StudioWindow

    def pump(seconds=0.05):
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)

    def wait(busy, timeout=300):
        t = time.time()
        while busy() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.02)
        pump(0.1)
        return not busy()

    images = []
    img, _, _ = render_drop(0.30, 70)               # needle 80 px wide at x 120..200
    png = tmp / 'synthetic drop.png'
    cv2.imwrite(str(png), img)
    images.append((png, SYNTHETIC))
    real = ROOT / 'Academic' / 'parity-images'
    for name, cfg in REAL_IMAGES.items():
        if (real / name).is_file():
            images.append((real / name, cfg))
        else:
            print(f"  (skipped {name}: not in Academic/parity-images)")
    images += [(Path(p), SYNTHETIC) for p in extra]

    for path, cfg in images:
        print(f"── {path.name} ──")
        x, y, bw, bh = cfg['box']

        # ------------------------------------------------ old Pendant window
        old = PendantWindow()
        old.density_inner, old.density_outer = cfg['rho_in'], cfg['rho_out']
        old.setup_image_for_analysis(cv2.imread(str(path), cv2.IMREAD_COLOR))
        old.combo_calib.setCurrentText("Vertical Needle")
        QInputDialog.getDouble = staticmethod(lambda *a, **k: (cfg['od_mm'], True))
        old.handle_calib_box(x, y, bw, bh)
        old.combo_base.setCurrentText("Auto")
        old.run_baseline()
        ok_old = old.width_px is not None and old.baseline_y is not None
        check("old window: calibrated and baseline found", ok_old,
              f"{old.width_px} px, baseline {old.baseline_y}")
        if not ok_old:
            old.deleteLater()
            continue
        old.run_static_analysis()
        check("old window: fit finished", wait(lambda: old.static_worker is not None))
        r_old = old.last_results
        check("old window: IFT computed", r_old is not None and r_old.get('fit_success'),
              old.lbl_sigma_unc.text())

        # ------------------------------------------------------------ Studio
        ini = QSettings(str(tmp / 'studio.ini'), QSettings.Format.IniFormat)
        w = StudioWindow(settings_store=ini)
        w.resize(1280, 800)
        w.show()
        check("Studio: session from the image", w.new_session('image', path=str(path)))
        d = w.draft
        d.set_needle(cfg['od_mm'])
        d.set_fluids(density_inner=cfg['rho_in'], density_outer=cfg['rho_out'])
        px, _ = d.measure_box(x, y, bw, bh, 'needle')
        d.set_scale_from_length(px, cfg['od_mm'], 'needle width', f"{px:.1f} px")
        d.auto_baseline()
        w.draft_changed()
        dest = tmp / f"{path.stem} session"
        check("Studio: logging starts", w.start_logging(path=dest))
        check("Studio: logging finished", wait(w.job_running))
        row = w.folder.log.rows[0] if w.folder is not None and len(w.folder.log) else None
        check("Studio: frame logged", row is not None and row['fit_ok'],
              (row or {}).get('error', 'no row'))

        # ----------------------------------------------------------- compare
        s = w.folder.session if w.folder is not None else None
        check("same needle width", s is not None and abs(s.scale['width_px'] - old.width_px) < 1e-9,
              f"old {old.width_px:.3f} px, Studio {s.scale['width_px']:.3f} px" if s else '')
        check("same scale", s is not None and abs(s.scale['pixel_to_m'] / old.pixel_to_m - 1) < 1e-12,
              f"{old.pixel_to_m * 1e6:.4f} um/px")
        check("same baseline", s is not None and s.baseline['y_px'] == old.baseline_y,
              f"old {old.baseline_y}, Studio {s.baseline['y_px'] if s else '?'}")
        if r_old and row:
            for k_old, k_new in COMPARED:
                a, b = r_old.get(k_old), row.get(k_new)
                same = a is not None and b is not None and abs(b - a) <= REL_TOL * max(abs(a), 1e-12)
                check(f"same {k_new}", same, f"old {a:.6g}, Studio {b:.6g}"
                      if a is not None and b is not None else f"old {a}, Studio {b}")
            check("same QC verdict", bool(r_old.get('qc_pass')) == bool(row['qc_pass']),
                  f"{'pass' if row['qc_pass'] else 'issues'}")
            print(f"     IFT {r_old['ift_mNm']:.3f} mN/m, beta {r_old['beta']:.4f}, "
                  f"R0 {r_old['R0_mm']:.4f} mm, V {r_old['volume_ul']:.2f} uL")
        w.close_current()
        w.close()
        old.deleteLater()
        pump(0.1)
    check("no unexpected warnings", not warnings, '; '.join(map(str, warnings)))


if __name__ == "__main__":
    main()
