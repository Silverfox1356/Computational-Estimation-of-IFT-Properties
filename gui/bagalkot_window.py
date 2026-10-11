"""
gui.bagalkot_window
===================

Bagalkot's (2019) volume method: the effective CO₂ diffusion coefficient
D from the measured drop volume V(t).  Opened from the Pendant window's
time-series mode, either with the volume curve of an analysed video or
empty (load a volume CSV).

The method and its numerics live in :mod:`core.bagalkot_estimator` and
:mod:`core.sphere_model`; this window only collects settings, runs the fit
in a worker thread and shows the result.
"""

import csv
import shutil
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core.bagalkot_estimator import (DEFAULT_D_RANGE, DEFAULT_N_GRID,
                                     DEFAULT_N_REFINE, BagalkotCancelled,
                                     estimate_D_bagalkot)
from core.sphere_model import N_CELLS
from core.volume_data import VolumeDataError, load_volume_curve
from gui.styles import BTN_PRIMARY, DARK_STYLESHEET
from gui.window_utils import fit_to_screen, scroll_root

VOLUME_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / 'data' / 'volume_curve_template.csv')


class BagalkotWorker(QThread):
    """Runs :func:`estimate_D_bagalkot` off the GUI thread."""
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, t, V, kwargs):
        super().__init__()
        self.t, self.V, self.kwargs = t, V, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            r = estimate_D_bagalkot(
                self.t, self.V, **self.kwargs,
                progress_cb=lambda i, n, D, F: self.progress.emit(i, n),
                is_cancelled=lambda: self._cancel)
            self.finished_ok.emit(r)
        except BagalkotCancelled:
            self.cancelled.emit()
        except Exception as e:                       # shown to the user
            self.failed.emit(str(e))


class BagalkotWindow(QWidget):
    """Fit D to a volume curve with Bagalkot's method."""

    def __init__(self, t=None, volume=None, source="", swelling=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bagalkot volume method: diffusion coefficient D")
        fit_to_screen(self, 1250, 820)
        self.setStyleSheet(DARK_STYLESHEET)
        self.t = None
        self.V = None
        self.source = ""
        self.result = None
        self.worker = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body_host = QWidget()
        body = scroll_root(body_host, QHBoxLayout)
        body.setContentsMargins(10, 10, 10, 6)
        outer.addWidget(body_host, 1)

        # ---- left: data + settings + result --------------------------
        left = QVBoxLayout()
        left.setSpacing(8)
        body.addLayout(left, 2)

        box_data = QGroupBox("1 · Volume data")
        ld = QVBoxLayout(box_data)
        self.lbl_data = QLabel("No volume curve loaded.")
        self.lbl_data.setWordWrap(True)
        row = QHBoxLayout()
        self.btn_template = QPushButton("Download template")
        self.btn_load = QPushButton("Load volume curve (CSV)…")
        row.addWidget(self.btn_template)
        row.addWidget(self.btn_load, 1)
        ld.addLayout(row)
        ld.addWidget(self.lbl_data)
        self.lbl_swelling = QLabel("")
        self.lbl_swelling.setWordWrap(True)
        self.lbl_swelling.setStyleSheet("color:#9aa3b8;")
        ld.addWidget(self.lbl_swelling)
        left.addWidget(box_data)

        box_model = QGroupBox("2 · Model (Bagalkot 2019, §4.2)")
        lm = QFormLayout(box_model)
        self.combo_interface = QComboBox()
        self.combo_interface.addItems(
            ["Growing sphere (adaptable interface, §4.2.4)",
             "Static sphere"])
        self.combo_interface.setToolTip(
            "Growing: the sphere's radius follows the model volume at every "
            "time step, as in his §4.2.4. Static: the sphere keeps its "
            "starting volume (Bagalkot & Hamouda 2017).")
        fpos = QDoubleValidator(0.0, 1e12, 15)
        self.in_V0 = QLineEdit("")
        self.in_V0.setValidator(fpos)
        self.in_V0.setPlaceholderText("first measured volume")
        self.in_V0.setToolTip(
            "Starting volume V0 (µL). The equivalent sphere has this volume. "
            "Leave blank to use the first measured volume.")
        self.in_D_min = QLineEdit(f"{DEFAULT_D_RANGE[0]:g}")
        self.in_D_max = QLineEdit(f"{DEFAULT_D_RANGE[1]:g}")
        for w in (self.in_D_min, self.in_D_max):
            w.setValidator(fpos)
        self.in_n_grid = QLineEdit(str(DEFAULT_N_GRID))
        self.in_n_grid.setValidator(QIntValidator(3, 1001))
        self.in_n_refine = QLineEdit(str(DEFAULT_N_REFINE))
        self.in_n_refine.setValidator(QIntValidator(0, 10))
        self.in_cells = QLineEdit(str(N_CELLS))
        self.in_cells.setValidator(QIntValidator(20, 5000))
        self.in_cells.setToolTip("Radial grid cells in the starting sphere.")
        lm.addRow("Interface:", self.combo_interface)
        lm.addRow("Starting volume V0 (µL):", self.in_V0)
        lm.addRow("D min (m²/s):", self.in_D_min)
        lm.addRow("D max (m²/s):", self.in_D_max)
        lm.addRow("D values per pass:", self.in_n_grid)
        lm.addRow("Refinement passes:", self.in_n_refine)
        lm.addRow("Grid cells:", self.in_cells)
        left.addWidget(box_model)

        box_fit = QGroupBox("3 · Fit D")
        lf = QVBoxLayout(box_fit)
        self.btn_fit = QPushButton("Fit D to the volume curve")
        self.btn_fit.setStyleSheet(BTN_PRIMARY)
        self.btn_fit.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_result = QLabel(
            "D only: Bagalkot's interface is at equilibrium (C = C0), so "
            "the result is an effective D with no interface resistance.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_result.setStyleSheet("color:#9aa3b8;")
        lf.addWidget(self.btn_fit)
        lf.addWidget(self.progress)
        lf.addWidget(self.lbl_result)
        left.addWidget(box_fit)
        left.addStretch(1)

        # ---- right: plots ---------------------------------------------
        right = QVBoxLayout()
        body.addLayout(right, 3)
        self.plot_V = pg.PlotWidget(title="Drop volume: measured and model")
        self.plot_V.setLabel('left', 'V (µL)')
        self.plot_V.setLabel('bottom', 'Time (s)')
        self.plot_V.showGrid(x=True, y=True, alpha=0.25)
        self.plot_V.addLegend(offset=(10, 10))
        self.pts_V = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None),
                                        brush=pg.mkBrush('#8ab4ff'),
                                        name='measured')
        self.plot_V.addItem(self.pts_V)
        self.line_V = self.plot_V.plot([], [], name='model',
                                       pen=pg.mkPen('#ffd166', width=2))
        self.plot_F = pg.PlotWidget(title="Objective F vs D (his Fig 4.3)")
        self.plot_F.setLabel('left', 'F (%)')
        self.plot_F.setLabel('bottom', 'D (m²/s)')
        self.plot_F.showGrid(x=True, y=True, alpha=0.25)
        self.plot_F.setLogMode(x=False, y=True)
        # plot() items (not raw scatter items) so the log axis applies
        self.pts_F = self.plot_F.plot([], [], pen=None, symbol='o',
                                      symbolSize=5, symbolPen=None,
                                      symbolBrush='#8ab4ff')
        self.best_F = self.plot_F.plot([], [], pen=None, symbol='o',
                                       symbolSize=12,
                                       symbolPen=pg.mkPen('#ffd166', width=2),
                                       symbolBrush=None)
        for p in (self.plot_V, self.plot_F):
            p.setMinimumHeight(240)
            right.addWidget(p, 1)

        # ---- bottom-right buttons -------------------------------------
        btns = QHBoxLayout()
        btns.setContentsMargins(10, 4, 10, 10)
        btns.addStretch(1)
        self.btn_export = QPushButton("Export results…")
        self.btn_export.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_close)
        outer.addLayout(btns)

        self.btn_template.clicked.connect(self.download_template)
        self.btn_load.clicked.connect(lambda: self.load_csv())
        self.btn_fit.clicked.connect(self.run_fit)
        self.btn_export.clicked.connect(self.export_results)
        self.btn_close.clicked.connect(self.close)

        if t is not None and volume is not None:
            self.set_data(t, volume, source)
        if swelling is not None:
            self.show_swelling(swelling)

    # ------------------------------------------------------------------
    def set_data(self, t, V, source=""):
        t = np.asarray(t, float)
        V = np.asarray(V, float)
        ok = np.isfinite(t) & np.isfinite(V) & (V > 0)
        self.t, self.V, self.source = t[ok], V[ok], source
        self.result = None
        self.btn_export.setEnabled(False)
        self.line_V.setData([], [])
        self.pts_F.setData([], [])
        self.best_F.setData([], [])
        self.pts_V.setData(self.t, self.V)
        n = len(self.t)
        if n:
            self.lbl_data.setText(
                f"{n} points from {source or 'the video analysis'}: "
                f"t = {self.t.min():g}–{self.t.max():g} s, "
                f"V = {self.V.min():.4g}–{self.V.max():.4g} µL "
                f"(V_end/V_first = {self.V[-1] / self.V[0]:.4f}).")
        self.btn_fit.setEnabled(n >= 5)
        if n < 5:
            self.lbl_data.setText(f"Only {n} valid volume point(s); at least "
                                  f"5 are needed.")

    def show_swelling(self, sw):
        self.lbl_swelling.setText("Swelling check: " + sw.get('message', ''))
        if sw.get('verdict') == 'significant':
            self.combo_interface.setCurrentIndex(0)

    def download_template(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save volume curve template", "volume_curve_template.csv",
            "CSV files (*.csv)")
        if not dest:
            return
        try:
            shutil.copyfile(VOLUME_TEMPLATE_PATH, dest)
        except OSError as e:
            QMessageBox.critical(self, "Could not save template", str(e))
            return
        self.lbl_data.setText(f"Template saved to {dest}. Replace the example "
                              f"rows with your own data, then load it.")

    def load_csv(self, path=None):
        if self.worker is not None and self.worker.isRunning():
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load volume curve", "", "CSV files (*.csv);;All files (*)")
            if not path:
                return
        try:
            d = load_volume_curve(path)
        except VolumeDataError as e:
            QMessageBox.critical(self, "Cannot load volume curve", str(e))
            return
        self.set_data(d['t'], d['volume'], Path(path).name)
        self.lbl_swelling.setText("")
        if d['warnings']:
            self.lbl_data.setText(self.lbl_data.text() + "".join(
                f"\nWarning: {w}" for w in d['warnings']))

    # ------------------------------------------------------------------
    def _settings(self):
        def num(w, name, cast=float):
            txt = w.text().strip()
            try:
                return cast(txt)
            except ValueError:
                raise ValueError(f"{name}: '{txt}' is not a number") from None
        V0_txt = self.in_V0.text().strip()
        kw = dict(D_range=(num(self.in_D_min, "D min"), num(self.in_D_max, "D max")),
                  n_grid=num(self.in_n_grid, "D values per pass", int),
                  n_refine=num(self.in_n_refine, "Refinement passes", int),
                  n_cells=num(self.in_cells, "Grid cells", int),
                  moving=self.combo_interface.currentIndex() == 0,
                  V0=float(V0_txt) if V0_txt else None)
        if not 0 < kw['D_range'][0] < kw['D_range'][1]:
            raise ValueError("D min must be > 0 and smaller than D max")
        if kw['V0'] is not None and kw['V0'] <= 0:
            raise ValueError("V0 must be > 0")
        return kw

    def run_fit(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.btn_fit.setEnabled(False)
            self.btn_fit.setText("Cancelling…")
            return
        try:
            kw = self._settings()
        except ValueError as e:
            QMessageBox.warning(self, "Check the settings", str(e))
            return
        self.worker = BagalkotWorker(self.t, self.V, kw)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._reset_ui)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_fit.setText("Cancel")
        self.btn_load.setEnabled(False)
        self.lbl_result.setText("Fitting… (growing sphere: about 0.1–1 s per "
                                "D value)" if kw['moving'] else "Fitting…")
        self.lbl_result.setStyleSheet("color:#8ab4ff;")
        self.worker.start()

    def _on_progress(self, i, n):
        self.progress.setRange(0, n)
        self.progress.setValue(i)

    def _reset_ui(self):
        self.progress.setVisible(False)
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("Fit D to the volume curve")
        self.btn_load.setEnabled(True)

    def _on_failed(self, msg):
        self.lbl_result.setText(f"Fit failed: {msg}")
        self.lbl_result.setStyleSheet("color:#ff6a4d;")

    def _on_cancelled(self):
        self.lbl_result.setText("Fit cancelled.")
        self.lbl_result.setStyleSheet("color:#ff9f5a;")

    def _on_done(self, r):
        self.result = r
        self.btn_export.setEnabled(True)
        self.line_V.setData(r['t'], r['V_model'])
        self.pts_F.setData(r['sweep_D'], np.maximum(r['sweep_F'], 1e-30))
        self.best_F.setData([r['D']], [max(r['F_min'], 1e-30)])
        lo, hi = r['D_near']
        lines = [
            f"D = {r['D']:.4g} m²/s  ({r['D_per_min']:.4g} m²/min)",
            f"Swelling factor SF = {r['SF']:.4f}",
            f"F_min = {r['F_min']:.4g} %  (eq 4.8)",
            f"Equally good fits (F within {r['near_tol'] * 100:.0f} % of the "
            f"minimum): D {lo:.3g} to {hi:.3g} m²/s"
            + (" or beyond (edge of the range)" if r['near_open'] else ""),
            f"Sphere: V0 = {r['V0']:.4g} µL, radius {r['radius_m'] * 1e3:.4f} mm; "
            f"{'growing (§4.2.4)' if r['moving'] else 'static'}",
            f"{r['n_evaluations']} D values evaluated",
        ]
        warn = []
        if r['at_edge']:
            warn.append("The first-pass minimum is at the edge of the D range: "
                        "widen the range and fit again.")
        if not r['sf_converged']:
            warn.append("SF iteration did not converge at the best D; the "
                        "result is approximate.")
        text = "\n".join(lines + [f"Warning: {w}" for w in warn])
        self.lbl_result.setText(text)
        self.lbl_result.setStyleSheet(
            "color:#ff9f5a; font-weight:600;" if warn else
            "color:#4fd07b; font-weight:600;")

    # ------------------------------------------------------------------
    def export_results(self):
        r = self.result
        if r is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Bagalkot results", "bagalkot_fit.csv",
            "CSV files (*.csv)")
        if not dest:
            return
        try:
            with open(dest, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# Bagalkot (2019) volume method, source: {self.source}\n")
                f.write(f"# D_m2_per_s,{r['D']:.6g}\n# D_m2_per_min,{r['D_per_min']:.6g}\n")
                f.write(f"# SF,{r['SF']:.6g}\n# F_min_percent,{r['F_min']:.6g}\n")
                f.write(f"# V0_uL,{r['V0']:.6g}\n# sphere_radius_mm,{r['radius_m'] * 1e3:.6g}\n")
                f.write(f"# interface,{'growing' if r['moving'] else 'static'}\n")
                f.write(f"# D_near_m2_per_s,{r['D_near'][0]:.6g},{r['D_near'][1]:.6g}\n")
                w = csv.writer(f)
                w.writerow(['time_s', 'volume_measured_uL', 'volume_model_uL'])
                for a, b, c in zip(r['t'], r['V_exp'], r['V_model']):
                    w.writerow([f"{a:.6g}", f"{b:.6g}", f"{c:.6g}"])
            sweep = Path(dest).with_name(Path(dest).stem + "_sweep.csv")
            with open(sweep, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['D_m2_per_s', 'F_percent', 'SF'])
                for a, b, c in zip(r['sweep_D'], r['sweep_F'], r['sweep_SF']):
                    w.writerow([f"{a:.6g}", f"{b:.6g}", f"{c:.6g}"])
        except OSError as e:
            QMessageBox.critical(self, "Could not export", str(e))
            return
        self.lbl_result.setText(self.lbl_result.text()
                                + f"\nExported to {dest} and {sweep.name}.")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        event.accept()
