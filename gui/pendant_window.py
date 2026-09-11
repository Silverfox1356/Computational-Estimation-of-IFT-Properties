"""
gui.pendant_window
==================

Pendant-drop main window.

Changes in this revision
------------------------
    · Result overlays (contour, YL fit, axis of symmetry, apex R₀
      osculating circle, apex crosshair) are Qt graphics items with
      COSMETIC pens → always visible at every zoom, no more lines
      vanishing at awkward zoom levels.
    · No on-image text / info-box. All numeric readout lives in the
      right-panel Results card.
    · Right panel wrapped in QScrollArea so the full results breakdown
      stays readable even on smaller displays.
    · Compact layout: Calibration and Baseline boxes live side by side.
    · Settings becomes a small ⚙ icon inline with the mode radios.
    · Shared dark theme + pressed-state buttons + visible dropdowns via
      gui.styles.
"""

import cv2
import sys
import time
import shutil
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QComboBox,
                             QGroupBox, QRadioButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsLineItem,
                             QGraphicsEllipseItem, QGraphicsPathItem,
                             QInputDialog, QFrame, QScrollArea,
                             QFormLayout, QLineEdit, QProgressBar,
                             QApplication)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QRectF, QProcess
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
                         QPen, QBrush, QColor, QPainterPath, QDoubleValidator)

from core.image_processing import attempt_auto_baseline
from core.pendant_calcs import extract_pendant_edges
from core.physics_calcs import (calculate_physics, compute_worthington,
                                yl_profile)
from core.domain_builder import contour_to_domain
from core.mesh_generator import generate_mesh
from core.fem_assembly import assemble_fem_system
from core.time_solver import solve_diffusion
from core.estimator import estimate_D_k_yang, EstimationCancelled
from core.calibration import (install_calibration, clear_active_calibration,
                              CalibrationError, ACTIVE_CALIBRATION_PATH)
from core.camera_handler import CameraHandler
from gui.settings_dialog import SettingsDialog
from gui.threshold_dialog import ThresholdDialog
from gui.domain_window import DomainWindow
from gui.mesh_window import MeshWindow
from gui.fem_window import FEMWindow
from gui.solver_window import SolverWindow
from gui.styles import (DARK_STYLESHEET, BTN_PRIMARY, BTN_SUCCESS, BTN_DANGER,
                        BTN_ICON, make_cosmetic_pen, make_pixel_pen)

# Downloadable blank template the user fills in and re-uploads.
CALIB_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / 'data' / 'calibration_template.csv')


# Overlay colors
_COL_CONTOUR  = QColor(100, 200, 255)     # cyan
_COL_BASELINE = QColor(  0, 220,  80)     # green
_COL_FITCURVE = QColor(255, 110, 220)     # magenta
_COL_AXIS     = QColor( 90, 255, 140)     # dashed light-green
_COL_APEX     = QColor(255, 235,  80)     # bright yellow crosshair
_COL_R0CIRC   = QColor(255, 200,  80)     # dashed amber (osculating R₀)


# =============================================================================
# Analysis worker (time series)
# =============================================================================
class AnalysisWorker(QThread):
    result_ready = pyqtSignal(float, dict)

    def __init__(self, pixel_to_m, baseline_y,
                 density_inner, density_outer,
                 needle_tol_rel, rho_in_tol_rel, rho_out_tol_rel,
                 needle_diameter_m):
        super().__init__()
        self.pixel_to_m  = pixel_to_m
        self.baseline_y  = baseline_y
        self.rho_in      = density_inner
        self.rho_out     = density_outer
        self.needle_tol  = needle_tol_rel
        self.rho_in_tol  = rho_in_tol_rel
        self.rho_out_tol = rho_out_tol_rel
        self.d_needle_m  = needle_diameter_m
        self.running     = True
        self.frame_queue = []

    def run(self):
        while self.running:
            if self.frame_queue:
                timestamp, frame = self.frame_queue.pop(0)
                try:
                    edges_df = extract_pendant_edges(frame)
                    if edges_df is not None:
                        results = calculate_physics(
                            edges_df, self.baseline_y, self.pixel_to_m,
                            density_inner=self.rho_in,
                            density_outer=self.rho_out,
                            needle_tol_rel=self.needle_tol,
                            density_inner_tol_rel=self.rho_in_tol,
                            density_outer_tol_rel=self.rho_out_tol,
                        )
                        if results.get('fit_success'):
                            results = compute_worthington(results, self.d_needle_m)
                            self.result_ready.emit(timestamp, results)
                except Exception as e:
                    print(f"[worker] skipped frame: {e}")
            else:
                time.sleep(0.04)

    def stop(self):
        self.running = False


class SimulationCancelled(Exception):
    """Raised inside the progress callback to unwind a cancelled run."""


class SimulationWorker(QThread):
    """Runs solve_diffusion off the UI thread so the window stays responsive."""

    progress    = pyqtSignal(int, int)   # (step, n_steps)
    finished_ok = pyqtSignal(dict)       # sim result dict
    failed      = pyqtSignal(str)        # error message
    cancelled   = pyqtSignal()

    def __init__(self, fem_results, domain_metadata, dtau, total_time,
                 calibration=None):
        super().__init__()
        self.fem_results     = fem_results
        self.domain_metadata = domain_metadata
        self.dtau            = dtau
        self.total_time      = total_time
        self.calibration     = calibration
        self._cancel         = False

    def cancel(self):
        self._cancel = True

    def _on_progress(self, step, n_steps, C, Cs):
        # Runs on the worker thread: only read a flag and emit signals here,
        # never touch Qt widgets directly.
        if self._cancel:
            raise SimulationCancelled()
        # Throttle to ~200 UI updates total, regardless of step count.
        if step % max(1, n_steps // 200) == 0 or step == n_steps:
            self.progress.emit(step, n_steps)

    def run(self):
        try:
            sim = solve_diffusion(
                self.fem_results, self.domain_metadata,
                D=1e-9, k=1e-5, theta=0.7,
                dtau=self.dtau, total_time=self.total_time,
                progress_cb=self._on_progress,
                calibration=self.calibration,
            )
            self.finished_ok.emit(sim)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))


class EstimationWorker(QThread):
    """Runs estimate_D_k_yang off the UI thread — the nested kD sweep runs
    hundreds of forward simulations and must not freeze the window."""

    finished_ok = pyqtSignal(dict)       # estimate_D_k_yang result dict
    failed      = pyqtSignal(str)
    cancelled   = pyqtSignal()

    def __init__(self, fem_results, domain_metadata, t_exp, gamma_exp,
                 calibration=None, kD_range=None, D_bounds=None,
                 auto_widen=True):
        super().__init__()
        self.fem_results     = fem_results
        self.domain_metadata = domain_metadata
        self.t_exp           = t_exp
        self.gamma_exp       = gamma_exp
        self.calibration     = calibration
        self.kD_range        = kD_range
        self.D_bounds        = D_bounds
        self.auto_widen      = auto_widen
        self._cancel         = False

    def cancel(self):
        """Ask the sweep to stop.  Checked before each forward simulation,
        so the thread unwinds within one sim rather than running to
        completion — a full sweep is minutes of work."""
        self._cancel = True

    def run(self):
        try:
            result = estimate_D_k_yang(
                self.fem_results, self.domain_metadata,
                self.t_exp, self.gamma_exp,
                calibration=self.calibration,
                kD_range=self.kD_range, D_bounds=self.D_bounds,
                auto_widen=self.auto_widen,
                should_cancel=lambda: self._cancel)
            self.finished_ok.emit(result)
        except EstimationCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))


# =============================================================================
# Graphics view
# =============================================================================
class ZoomableGraphicsView(QGraphicsView):
    image_clicked   = pyqtSignal(int, int)
    mouse_moved     = pyqtSignal(int, int)
    calib_box_ready = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#0f1018"))
        self.setMouseTracking(True)

        self.interaction_mode = 'idle'
        self.waiting_for_click = False
        self.crop_start = None
        self.crop_rect_item = None
        self.dynamic_items = []
        self._is_panning = False
        self._pan_start = None

    def set_image(self, cv_img, auto_fit=False):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        first_load = self.image_item.pixmap().isNull()
        self.image_item.setPixmap(pixmap)
        if first_load or auto_fit:
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.fitInView(self.scene.itemsBoundingRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def clear_dynamic_drawings(self):
        for item in self.dynamic_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        self.dynamic_items.clear()
        self.crop_rect_item = None

    def set_mode(self, mode):
        self.interaction_mode = mode
        self.clear_dynamic_drawings()

    def get_crop_rect(self):
        if self.crop_rect_item:
            r = self.crop_rect_item.rect()
            return int(r.x()), int(r.y()), int(r.width()), int(r.height())
        return None

    def wheelEvent(self, event: QWheelEvent):
        if self.image_item.pixmap().isNull():
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            if not self.image_item.pixmap().isNull():
                pos = self.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                if self.interaction_mode in ('crop', 'calib_box'):
                    self.crop_start = (x, y)
                    self.crop_rect_item = QGraphicsRectItem()
                    color = (Qt.GlobalColor.cyan if self.interaction_mode == 'crop'
                             else Qt.GlobalColor.yellow)
                    self.crop_rect_item.setPen(make_pixel_pen(color))
                    self.scene.addItem(self.crop_rect_item)
                    self.dynamic_items.append(self.crop_rect_item)
                elif self.waiting_for_click and self.image_item.contains(pos):
                    self.image_clicked.emit(x, y)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_panning:
            dx = event.pos().x() - self._pan_start.x()
            dy = event.pos().y() - self._pan_start.y()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - dy)
            self._pan_start = event.pos()
            event.accept()
        else:
            if not self.image_item.pixmap().isNull():
                pos = self.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                if self.interaction_mode in ('crop', 'calib_box') and self.crop_start is not None:
                    sx, sy = self.crop_start
                    rect = QRectF(min(sx, x) + 0.5, min(sy, y) + 0.5,
                                  abs(x - sx), abs(y - sy))
                    self.crop_rect_item.setRect(rect)
                elif self.waiting_for_click and self.image_item.contains(pos):
                    self.mouse_moved.emit(x, y)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.interaction_mode == 'crop':
                self.crop_start = None
            elif self.interaction_mode == 'calib_box':
                if self.crop_rect_item:
                    r = self.crop_rect_item.rect()
                    self.calib_box_ready.emit(int(r.x()), int(r.y()),
                                              int(r.width()), int(r.height()))
                self.crop_start = None
                self.set_mode('idle')
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# =============================================================================
# Pendant window
# =============================================================================
class PendantWindow(QWidget):
    def __init__(self, camera_type="Basler"):
        super().__init__()
        self.setWindowTitle("Pendant Drop Analysis")
        self.resize(1380, 880)
        self.setStyleSheet(DARK_STYLESHEET)

        # --- state ----------------------------------------------------
        self.capture = None
        self.live_camera = CameraHandler(camera_type)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_raw_image = None
        self.is_video_file = False

        self.needle_mm        = 0.7176
        self.density_inner    = 998.2
        self.density_outer    = 1.204
        self.ts_interval      = 1.0
        self.needle_id_mm     = None
        self.needle_tol_rel   = 0.010
        self.rho_in_tol_rel   = 0.0005
        self.rho_out_tol_rel  = 0.020

        self.edges_df = None
        self.pixel_to_m = None
        self.width_px = None
        self.baseline_y = None
        self.click_mode = None
        self.calib_pts = []
        self.guide_item = None
        self.last_results = None

        self._result_overlays = []
        self._baseline_overlay = None

        self.worker = None
        self.sim_worker = None
        self.est_worker = None
        # Time grid the last fit ran on (QC-filtered), for the overlay plot.
        self._est_t_exp = np.array([])
        # Concentration→IFT calibration (None = placeholder linear model).
        # Set by uploading a filled template; see the Calibration box.
        self.calibration = None
        self.time_data, self.ift_data, self.ift_err, self.qc_flags = [], [], [], []
        self.last_grab_time = -999.0
        self.ts_start_wall_time = 0

        # --- root layout ----------------------------------------------
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.image_label = ZoomableGraphicsView()
        self.image_label.setStyleSheet(
            "QGraphicsView{background-color:#0f1018;border:1px solid #343852;"
            "border-radius:6px;}")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.image_clicked.connect(self.handle_image_click)
        self.image_label.mouse_moved.connect(self.handle_mouse_move)
        self.image_label.calib_box_ready.connect(self.handle_calib_box)
        main_layout.addWidget(self.image_label, stretch=3)

        # --- scrollable right panel -----------------------------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setMinimumWidth(420)
        right_scroll.setMaximumWidth(520)
        right_host = QWidget()
        right_scroll.setWidget(right_host)
        main_layout.addWidget(right_scroll, stretch=2)
        right_panel = QVBoxLayout(right_host)
        right_panel.setContentsMargins(4, 4, 8, 4)
        right_panel.setSpacing(8)

        # Mode + settings icon
        mode_row = QHBoxLayout()
        self.radio_static = QRadioButton("Static")
        self.radio_ts     = QRadioButton("Time series")
        self.radio_static.setChecked(True)
        mode_row.addWidget(self.radio_static)
        mode_row.addWidget(self.radio_ts)
        mode_row.addStretch(1)
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedWidth(40)
        self.btn_settings.setToolTip(
            "Physical parameters and measurement tolerances")
        self.btn_settings.setStyleSheet(BTN_ICON)
        mode_row.addWidget(self.btn_settings)
        self.btn_restart = QPushButton("⟳")
        self.btn_restart.setFixedWidth(40)
        self.btn_restart.setToolTip("Restart the application (fresh state)")
        self.btn_restart.setStyleSheet(BTN_ICON)
        mode_row.addWidget(self.btn_restart)
        mode_box = QGroupBox("Analysis mode")
        mode_box.setLayout(mode_row)
        right_panel.addWidget(mode_box)

        # Source & prep
        box_source = QGroupBox("Source input and preparation")
        l_source = QVBoxLayout()
        h_source1 = QHBoxLayout()
        self.btn_upload_img = QPushButton("Upload image")
        self.btn_upload_vid = QPushButton("Upload video")
        h_source1.addWidget(self.btn_upload_img)
        h_source1.addWidget(self.btn_upload_vid)
        self.btn_camera  = QPushButton("Start live camera")
        self.btn_capture = QPushButton("📸  Capture frame")
        self.btn_capture.hide()
        h_prep = QHBoxLayout()
        self.btn_crop       = QPushButton("✂  Crop")
        self.btn_apply_crop = QPushButton("✓  Apply crop")
        self.btn_apply_crop.setStyleSheet(BTN_SUCCESS)
        self.btn_thresh     = QPushButton("🎛  Threshold…")
        self.btn_crop.setEnabled(False)
        self.btn_thresh.setEnabled(False)
        self.btn_apply_crop.hide()
        h_prep.addWidget(self.btn_crop)
        h_prep.addWidget(self.btn_apply_crop)
        h_prep.addWidget(self.btn_thresh)
        l_source.addLayout(h_source1)
        l_source.addWidget(self.btn_camera)
        l_source.addWidget(self.btn_capture)
        l_source.addLayout(h_prep)
        box_source.setLayout(l_source)
        right_panel.addWidget(box_source)

        # Step 1 | Step 2 side-by-side
        calib_base_row = QHBoxLayout()
        calib_base_row.setSpacing(8)

        self.box_calib = QGroupBox("Step 1 · Calibration")
        l_calib = QVBoxLayout()
        self.combo_calib = QComboBox()
        self.combo_calib.addItems(
            ["Vertical Needle", "Sphere", "Misc", "Direct Input"])
        h_calib_btns = QHBoxLayout()
        self.btn_calib_auto   = QPushButton("Auto")
        self.btn_calib_manual = QPushButton("Manual")
        h_calib_btns.addWidget(self.btn_calib_auto)
        h_calib_btns.addWidget(self.btn_calib_manual)
        self.lbl_calib_res = QLabel("Awaiting input…")
        self.lbl_calib_res.setWordWrap(True)
        l_calib.addWidget(self.combo_calib)
        l_calib.addLayout(h_calib_btns)
        l_calib.addWidget(self.lbl_calib_res)
        self.box_calib.setLayout(l_calib)
        self.box_calib.setEnabled(False)

        self.box_base = QGroupBox("Step 2 · Baseline")
        l_base = QVBoxLayout()
        self.combo_base = QComboBox()
        self.combo_base.addItems(["Auto", "Manual"])
        self.btn_run_base = QPushButton("Run baseline")
        self.lbl_base_res = QLabel("Awaiting calibration…")
        self.lbl_base_res.setWordWrap(True)
        l_base.addWidget(self.combo_base)
        l_base.addWidget(self.btn_run_base)
        l_base.addWidget(self.lbl_base_res)
        self.box_base.setLayout(l_base)
        self.box_base.setEnabled(False)

        calib_base_row.addWidget(self.box_calib, 1)
        calib_base_row.addWidget(self.box_base, 1)
        right_panel.addLayout(calib_base_row)

        # Step 3 · Analyze + results
        self.box_analyze = QGroupBox("Step 3 · Analyze")
        l_analyze = QVBoxLayout()
        l_analyze.setSpacing(6)

        h_analyze_btns = QHBoxLayout()
        self.btn_analyze_static = QPushButton("Calculate IFT")
        self.btn_analyze_static.setStyleSheet(BTN_PRIMARY)
        self.btn_analyze_time = QPushButton("▶  Start time series")
        self.btn_analyze_time.setStyleSheet(BTN_SUCCESS)
        h_analyze_btns.addWidget(self.btn_analyze_static)
        h_analyze_btns.addWidget(self.btn_analyze_time)
        l_analyze.addLayout(h_analyze_btns)

        # Results card
        self.results_card = QFrame(objectName="card")
        rc = QVBoxLayout(self.results_card)
        rc.setContentsMargins(12, 10, 12, 10)
        rc.setSpacing(3)

        self.lbl_sigma_big = QLabel("—")
        self.lbl_sigma_big.setStyleSheet(
            "font-size:22pt; font-weight:700; color:#ffffff;")
        self.lbl_sigma_unc = QLabel("")
        self.lbl_sigma_unc.setStyleSheet("color:#aab4d0;")
        self.lbl_qc_badge = QLabel("")
        self.lbl_qc_badge.setStyleSheet("color:#9aa3b8; font-weight:600;")
        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("color:#c3c8dc;")
        self.lbl_meta.setWordWrap(True)
        self.lbl_dsde = QLabel("")
        self.lbl_dsde.setStyleSheet("color:#9aa3b8; font-style:italic;")
        self.lbl_unc_breakdown = QLabel("")
        self.lbl_unc_breakdown.setStyleSheet(
            "color:#8e95ae; font-family:'Consolas','Menlo',monospace; "
            "font-size:9pt;")

        rc.addWidget(self.lbl_sigma_big)
        rc.addWidget(self.lbl_sigma_unc)
        rc.addWidget(self.lbl_qc_badge)
        rc.addWidget(self.lbl_meta)
        rc.addWidget(self.lbl_dsde)
        rc.addWidget(self.lbl_unc_breakdown)

        l_analyze.addWidget(self.results_card)
        self.box_analyze.setLayout(l_analyze)
        self.box_analyze.setEnabled(False)
        right_panel.addWidget(self.box_analyze)

        # Step 4 · Domain
        self.box_domain = QGroupBox("Step 4 · Domain")
        l_domain = QVBoxLayout()
        self.btn_build_domain = QPushButton("🔧  Build Domain")
        self.btn_build_domain.setStyleSheet(BTN_PRIMARY)
        self.lbl_domain_status = QLabel("Run IFT analysis first.")
        self.lbl_domain_status.setWordWrap(True)
        self.lbl_domain_status.setStyleSheet("color:#9aa3b8;")
        l_domain.addWidget(self.btn_build_domain)
        l_domain.addWidget(self.lbl_domain_status)
        self.box_domain.setLayout(l_domain)
        self.box_domain.setEnabled(False)
        right_panel.addWidget(self.box_domain)

        # Step 5 · Meshing
        self.box_mesh = QGroupBox("Step 5 · Mesh")
        l_mesh = QVBoxLayout()
        self.btn_generate_mesh = QPushButton("▦  Generate Mesh")
        self.btn_generate_mesh.setStyleSheet(BTN_PRIMARY)
        self.lbl_mesh_status = QLabel("Build domain first.")
        self.lbl_mesh_status.setWordWrap(True)
        self.lbl_mesh_status.setStyleSheet("color:#9aa3b8;")
        l_mesh.addWidget(self.btn_generate_mesh)
        l_mesh.addWidget(self.lbl_mesh_status)
        self.box_mesh.setLayout(l_mesh)
        self.box_mesh.setEnabled(False)
        right_panel.addWidget(self.box_mesh)

        # Step 6 · FEM Assembly
        self.box_fem = QGroupBox("Step 6 · FEM Assembly")
        l_fem = QVBoxLayout()
        self.btn_run_fem = QPushButton("∑  Assemble FEM")
        self.btn_run_fem.setStyleSheet(BTN_PRIMARY)
        self.lbl_fem_status = QLabel("Generate mesh first.")
        self.lbl_fem_status.setWordWrap(True)
        self.lbl_fem_status.setStyleSheet("color:#9aa3b8;")
        l_fem.addWidget(self.btn_run_fem)
        l_fem.addWidget(self.lbl_fem_status)
        self.box_fem.setLayout(l_fem)
        self.box_fem.setEnabled(False)
        right_panel.addWidget(self.box_fem)

        # Calibration curve (concentration → IFT).  Optional: without it
        # the solver uses a crude linear γ(Cs) placeholder.
        self.box_calib_curve = QGroupBox("Calibration curve (concentration → IFT)")
        l_cal = QVBoxLayout()

        cal_head = QHBoxLayout()
        self.btn_calib_download = QPushButton("⬇  Download template")
        self.btn_calib_upload   = QPushButton("⬆  Upload calibration")
        self.btn_calib_upload.setStyleSheet(BTN_PRIMARY)
        self.btn_calib_clear = QPushButton("Clear")
        self.btn_calib_clear.setToolTip(
            "Forget the loaded calibration — reverts to the placeholder model")
        self.btn_calib_help = QPushButton("?")
        self.btn_calib_help.setCheckable(True)
        self.btn_calib_help.setFixedWidth(30)
        self.btn_calib_help.setStyleSheet(BTN_ICON)
        self.btn_calib_help.setToolTip("Show / hide instructions")
        cal_head.addWidget(self.btn_calib_download)
        cal_head.addWidget(self.btn_calib_upload)
        cal_head.addWidget(self.btn_calib_clear)
        cal_head.addWidget(self.btn_calib_help)
        l_cal.addLayout(cal_head)

        # Click-to-toggle instructions (no hover), hidden until '?' is on.
        self.lbl_calib_help = QLabel(
            "<b>How to supply a calibration curve</b><br>"
            "1. <b>Download template</b> and open it in any spreadsheet.<br>"
            "2. Under the <tt>concentration,ift</tt> header, enter your "
            "equilibrium points — one per row, concentration increasing. "
            "First row = clean surface (highest IFT), last row = saturated "
            "(lowest IFT).<br>"
            "3. <b>concentration_unit</b>: any unit, as long as "
            "<tt>c_sat</tt> uses the same one. IFT is mN/m.<br>"
            "4. <b>c_sat</b>: leave blank to use your last row as the "
            "saturation concentration, or type a value to override it.<br>"
            "5. Save as CSV and press <b>Upload calibration</b>.")
        self.lbl_calib_help.setWordWrap(True)
        self.lbl_calib_help.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_calib_help.setStyleSheet(
            "background:#171a26; border:1px solid #343852; border-radius:6px;"
            "padding:8px; color:#c3c8dc;")
        self.lbl_calib_help.setVisible(False)
        l_cal.addWidget(self.lbl_calib_help)

        self.lbl_calib_status = QLabel(
            "No curve loaded — using placeholder linear γ(Cs) model.")
        self.lbl_calib_status.setWordWrap(True)
        self.lbl_calib_status.setStyleSheet("color:#9aa3b8;")
        l_cal.addWidget(self.lbl_calib_status)

        self.box_calib_curve.setLayout(l_cal)
        right_panel.addWidget(self.box_calib_curve)

        # Step 7 · Simulation
        self.box_sim = QGroupBox("Step 7 · Simulation")
        l_sim = QVBoxLayout()
        fv = QDoubleValidator(0.0001, 1e6, 6)
        form_sim = QFormLayout()
        self.sim_duration_input = QLineEdit("10")
        self.sim_duration_input.setValidator(fv)
        self.sim_dtau_input = QLineEdit("0.01")
        self.sim_dtau_input.setValidator(fv)
        form_sim.addRow("Simulated duration (min):", self.sim_duration_input)
        form_sim.addRow("Time step Δt (s):", self.sim_dtau_input)
        l_sim.addLayout(form_sim)
        self.btn_run_sim = QPushButton("▶  Run Simulation")
        self.btn_run_sim.setStyleSheet(BTN_SUCCESS)
        self.sim_progress = QProgressBar()
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setVisible(False)
        self.lbl_sim_status = QLabel("Assemble FEM first.")
        self.lbl_sim_status.setWordWrap(True)
        self.lbl_sim_status.setStyleSheet("color:#9aa3b8;")
        l_sim.addWidget(self.btn_run_sim)
        l_sim.addWidget(self.sim_progress)
        l_sim.addWidget(self.lbl_sim_status)
        self.box_sim.setLayout(l_sim)
        self.box_sim.setEnabled(False)
        right_panel.addWidget(self.box_sim)

        # Step 8 · Parameter estimation (inverse problem)
        self.box_est = QGroupBox("Step 8 · Estimate D, k")
        l_est = QVBoxLayout()

        # Search-range mode: Auto (wide general defaults, auto-widened) vs
        # Manual (user pins the D and kD ranges for a system they know).
        range_row = QHBoxLayout()
        self.radio_range_auto = QRadioButton("Auto range")
        self.radio_range_manual = QRadioButton("Manual range")
        self.radio_range_auto.setChecked(True)
        self.radio_range_auto.setToolTip(
            "Wide, system-agnostic search that auto-widens if the minimum "
            "hits an edge. Recommended when you don't know D/kD magnitudes.")
        self.radio_range_manual.setToolTip(
            "Pin the D and kD search ranges yourself (no auto-widening).")
        range_row.addWidget(self.radio_range_auto)
        range_row.addWidget(self.radio_range_manual)
        range_row.addStretch(1)
        l_est.addLayout(range_row)

        # Manual range inputs — hidden unless Manual is selected.
        self.manual_range_box = QWidget()
        mr = QFormLayout(self.manual_range_box)
        mr.setContentsMargins(0, 4, 0, 4)
        fv_pos = QDoubleValidator(0.0, 1e12, 15)
        self.in_D_min  = QLineEdit("1e-11")
        self.in_D_min.setValidator(fv_pos)
        self.in_D_max  = QLineEdit("1e-8")
        self.in_D_max.setValidator(fv_pos)
        self.in_kD_min = QLineEdit("0.01")
        self.in_kD_min.setValidator(fv_pos)
        self.in_kD_max = QLineEdit("100")
        self.in_kD_max.setValidator(fv_pos)
        mr.addRow("D min (m²/s):", self.in_D_min)
        mr.addRow("D max (m²/s):", self.in_D_max)
        mr.addRow("kD min:", self.in_kD_min)
        mr.addRow("kD max:", self.in_kD_max)
        self.manual_range_box.setVisible(False)
        l_est.addWidget(self.manual_range_box)

        self.btn_estimate = QPushButton("⇆  Fit D, k to measured IFT")
        self.btn_estimate.setStyleSheet(BTN_PRIMARY)
        self.lbl_est_status = QLabel(
            "Needs FEM (Step 6) + a time-series IFT curve (Step 3).")
        self.lbl_est_status.setWordWrap(True)
        self.lbl_est_status.setStyleSheet("color:#9aa3b8;")
        l_est.addWidget(self.btn_estimate)
        l_est.addWidget(self.lbl_est_status)
        self.box_est.setLayout(l_est)
        self.box_est.setEnabled(False)
        right_panel.addWidget(self.box_est)

        # Plot
        pg.setConfigOption('background', '#0f1018')
        pg.setConfigOption('foreground', '#c3c8dc')
        self.graph_widget = pg.PlotWidget(title="IFT vs time")
        self.graph_widget.setLabel('left',   'IFT (mN/m)')
        self.graph_widget.setLabel('bottom', 'Time (s)')
        self.graph_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_line = self.graph_widget.plot(
            [], [], pen=pg.mkPen(color='#8ab4ff', width=2))
        self.plot_fit_line = self.graph_widget.plot(
            [], [], pen=pg.mkPen(color='#ffd166', width=2,
                                 style=Qt.PenStyle.DashLine))
        self.plot_pts_pass = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush('#4fd07b'))
        self.plot_pts_fail = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush('#ff6a4d'))
        self.graph_widget.addItem(self.plot_pts_pass)
        self.graph_widget.addItem(self.plot_pts_fail)
        self.err_item = pg.ErrorBarItem(
            x=np.array([]), y=np.array([]),
            top=np.array([]), bottom=np.array([]),
            pen=pg.mkPen('#5a85e2', width=1.2))
        self.graph_widget.addItem(self.err_item)
        self.graph_widget.setMinimumHeight(220)
        right_panel.addWidget(self.graph_widget)
        right_panel.addStretch(1)

        # wiring
        self.radio_static.toggled.connect(self.switch_mode)
        self.radio_ts.toggled.connect(self.switch_mode)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_upload_img.clicked.connect(self.upload_picture)
        self.btn_upload_vid.clicked.connect(self.upload_video)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_crop.clicked.connect(self.start_crop)
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_thresh.clicked.connect(self.open_threshold_dialog)
        self.btn_calib_auto.clicked.connect(self.start_auto_calib)
        self.btn_calib_manual.clicked.connect(self.start_manual_calib)
        self.combo_calib.currentTextChanged.connect(self.update_calib_buttons)
        self.btn_run_base.clicked.connect(self.run_baseline)
        self.btn_analyze_static.clicked.connect(self.run_static_analysis)
        self.btn_analyze_time.clicked.connect(self.toggle_time_series)
        self.btn_build_domain.clicked.connect(self.build_domain)
        self.btn_generate_mesh.clicked.connect(self.run_meshing)
        self.btn_run_fem.clicked.connect(self.run_fem)
        self.btn_run_sim.clicked.connect(self.run_simulation)
        self.btn_estimate.clicked.connect(self.run_estimation)
        self.btn_calib_download.clicked.connect(self.download_calib_template)
        self.btn_calib_upload.clicked.connect(self.upload_calibration)
        self.btn_calib_clear.clicked.connect(self.reset_calibration)
        self.btn_calib_help.toggled.connect(self.lbl_calib_help.setVisible)
        self.btn_restart.clicked.connect(self.restart_application)
        self.radio_range_manual.toggled.connect(self.manual_range_box.setVisible)

        self.switch_mode()

    # =========================================================================
    # Overlay management
    # =========================================================================
    def _clear_result_overlays(self):
        for item in self._result_overlays:
            if item.scene() is self.image_label.scene:
                self.image_label.scene.removeItem(item)
        self._result_overlays.clear()

    def _clear_baseline_overlay(self):
        if self._baseline_overlay is not None and \
                self._baseline_overlay.scene() is self.image_label.scene:
            self.image_label.scene.removeItem(self._baseline_overlay)
        self._baseline_overlay = None

    def _add_overlay_polyline(self, xs, ys, color, dashed=False):
        path = QPainterPath()
        path.moveTo(float(xs[0]) + 0.5, float(ys[0]) + 0.5)
        for i in range(1, len(xs)):
            path.lineTo(float(xs[i]) + 0.5, float(ys[i]) + 0.5)
        item = QGraphicsPathItem(path)
        item.setPen(make_cosmetic_pen(color, dashed=dashed))
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.image_label.scene.addItem(item)
        self._result_overlays.append(item)

    def _paint_baseline(self, y):
        self._clear_baseline_overlay()
        if self.current_raw_image is None:
            return
        w = self.current_raw_image.shape[1]
        line = QGraphicsLineItem(0.0, float(y) + 0.5, float(w), float(y) + 0.5)
        line.setPen(make_cosmetic_pen(_COL_BASELINE))
        self.image_label.scene.addItem(line)
        self._baseline_overlay = line

    def _draw_pendant_overlays(self, results):
        """Build all Qt-side overlays for the current fit: contour, YL fit,
        axis of symmetry, apex R₀ osculating circle, apex crosshair."""
        self._clear_result_overlays()
        scene = self.image_label.scene

        ys  = results['ys_use']
        lxs = results['lxs']
        rxs = results['rxs']

        # Smoothed detected contour (both sides, cyan)
        self._add_overlay_polyline(lxs, ys, _COL_CONTOUR)
        self._add_overlay_polyline(rxs, ys, _COL_CONTOUR)

        # Fitted YL profile
        _, xn, zn, _ = yl_profile(results['beta'], n=700)
        xr = results['x_axis_fit'] + xn * results['R0_px']
        xl = results['x_axis_fit'] - xn * results['R0_px']
        yt = results['y_apex_fit'] - zn * results['R0_px']
        mask = (yt >= self.baseline_y - 2) & (yt <= results['y_apex_fit'] + 2)
        if mask.sum() > 2:
            self._add_overlay_polyline(xr[mask], yt[mask], _COL_FITCURVE)
            self._add_overlay_polyline(xl[mask], yt[mask], _COL_FITCURVE)

        # Apex osculating (R₀) circle — centered R₀ above the fitted apex
        R = results['R0_px']
        cx_circle = results['x_axis_fit']
        cy_circle = results['y_apex_fit'] - R
        ellipse = QGraphicsEllipseItem(cx_circle - R + 0.5,
                                       cy_circle - R + 0.5,
                                       2 * R, 2 * R)
        ellipse.setPen(make_cosmetic_pen(_COL_R0CIRC, dashed=True))
        ellipse.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        scene.addItem(ellipse)
        self._result_overlays.append(ellipse)

        # Axis of symmetry (dashed, from baseline to just below apex)
        ax = results['x_axis_fit']
        axis = QGraphicsLineItem(ax + 0.5, float(self.baseline_y),
                                 ax + 0.5, results['y_apex_fit'] + 10)
        axis.setPen(make_cosmetic_pen(_COL_AXIS, dashed=True))
        scene.addItem(axis)
        self._result_overlays.append(axis)

        # Apex crosshair
        ay = results['y_apex_fit']
        ch_len = 7
        h = QGraphicsLineItem(ax - ch_len + 0.5, ay + 0.5,
                              ax + ch_len + 0.5, ay + 0.5)
        h.setPen(make_cosmetic_pen(_COL_APEX))
        scene.addItem(h)
        self._result_overlays.append(h)
        v = QGraphicsLineItem(ax + 0.5, ay - ch_len + 0.5,
                              ax + 0.5, ay + ch_len + 0.5)
        v.setPen(make_cosmetic_pen(_COL_APEX))
        scene.addItem(v)
        self._result_overlays.append(v)

    # =========================================================================
    # UI helpers
    # =========================================================================
    def switch_mode(self):
        self.stop_camera()
        self.clear_guide()
        if self.radio_static.isChecked():
            self.btn_upload_img.show()
            self.btn_upload_vid.hide()
            self.btn_analyze_static.show()
            self.btn_analyze_time.hide()
            self.graph_widget.hide()
        else:
            self.btn_upload_img.hide()
            self.btn_upload_vid.show()
            self.btn_analyze_static.hide()
            self.btn_analyze_time.show()
            self.graph_widget.show()

    def open_settings(self):
        dialog = SettingsDialog(
            self.needle_mm, self.density_inner, self.density_outer,
            self.ts_interval,
            self.needle_tol_rel, self.rho_in_tol_rel, self.rho_out_tol_rel,
            needle_id_mm=self.needle_id_mm,
            parent=self)
        dialog.setStyleSheet(DARK_STYLESHEET)
        if dialog.exec():
            (self.needle_mm, self.density_inner, self.density_outer,
             self.ts_interval, self.needle_tol_rel,
             self.rho_in_tol_rel, self.rho_out_tol_rel,
             self.needle_id_mm) = dialog.get_values()
            if self.width_px is not None:
                self.pixel_to_m = (self.needle_mm / 1000.0) / self.width_px

    def start_crop(self):
        self.image_label.set_mode('crop')
        self.btn_apply_crop.show()
        self.btn_crop.hide()

    def apply_crop(self):
        rect = self.image_label.get_crop_rect()
        if rect:
            x, y, w, h = rect
            if w > 10 and h > 10 and self.current_raw_image is not None:
                H, W = self.current_raw_image.shape[:2]
                y1, y2 = max(0, y), min(H, y + h)
                x1, x2 = max(0, x), min(W, x + w)
                cropped = self.current_raw_image[y1:y2, x1:x2]
                self.setup_image_for_analysis(cropped)
        self.btn_apply_crop.hide()
        self.btn_crop.show()
        self.image_label.set_mode('idle')

    def open_threshold_dialog(self):
        if self.current_raw_image is not None:
            dialog = ThresholdDialog(self.current_raw_image, self)
            dialog.setStyleSheet(DARK_STYLESHEET)
            if dialog.exec():
                self.setup_image_for_analysis(dialog.get_processed_image())

    # --- calibration ---------------------------------------------------
    def update_calib_buttons(self, text):
        self.btn_calib_auto.setEnabled(text not in ("Misc", "Direct Input"))
        self.btn_calib_manual.setText(
            "Enter value" if text == "Direct Input" else "Manual")

    def start_manual_calib(self):
        if self.current_raw_image is None:
            return
        mode = self.combo_calib.currentText()
        if mode == "Direct Input":
            val, ok = QInputDialog.getDouble(
                self, "Pixel scale", "Enter scale (mm/px):", 0.01, decimals=6)
            if ok and val > 0:
                self.pixel_to_m = val / 1000.0
                self._set_calib_label(f"Scale: {val:.5f} mm/px", success=True)
                self.box_base.setEnabled(True)
        else:
            self.click_mode = 'calib'
            self.calib_pts = []
            self.image_label.waiting_for_click = True
            self.image_label.set_image(self.current_raw_image, auto_fit=False)
            self.lbl_calib_res.setText("Click TWO points.")

    def start_auto_calib(self):
        if self.current_raw_image is None:
            return
        self.image_label.set_image(self.current_raw_image, auto_fit=False)
        self.image_label.set_mode('calib_box')
        self.lbl_calib_res.setText("Draw a tight box around the reference object.")

    def handle_calib_box(self, x, y, w, h):
        if w < 5 or h < 5:
            self._set_calib_label("Box too small. Try again.", success=False)
            return
        mode = self.combo_calib.currentText()
        H, W = self.current_raw_image.shape[:2]
        y1, y2 = max(0, y), min(H, y + h)
        x1, x2 = max(0, x), min(W, x + w)
        roi = self.current_raw_image[y1:y2, x1:x2]
        pixel_dist, draw_info = self.auto_measure_roi(roi, mode)
        if pixel_dist is None or pixel_dist == 0:
            self._set_calib_label("Auto failed. Shape not found.", success=False)
            self.image_label.set_mode('idle')
            return

        preview = self.current_raw_image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 1)
        if draw_info[0] == 'line':
            pt1, pt2 = draw_info[1]
            cv2.line(preview,
                     (pt1[0] + x1, pt1[1] + y1),
                     (pt2[0] + x1, pt2[1] + y1),
                     (0, 255, 255), 1)
        elif draw_info[0] == 'circle':
            c_global = (draw_info[1][0] + x1, draw_info[1][1] + y1)
            cv2.circle(preview, c_global, draw_info[2], (0, 255, 255), 1)
        self.image_label.set_image(preview, auto_fit=False)
        self.image_label.set_mode('idle')

        val, ok = QInputDialog.getDouble(
            self, "Verify auto calibration",
            f"Auto detected: {pixel_dist:.1f} px.\n"
            "Verify the cyan marker on the image.\n"
            "Enter real physical size (mm):",
            self.needle_mm, decimals=4)
        if ok and val > 0:
            self.pixel_to_m = (val / 1000.0) / pixel_dist
            self.width_px = pixel_dist
            self.needle_mm = val
            self._set_calib_label(f"Auto: {pixel_dist:.1f} px ≡ {val} mm",
                                  success=True)
            self.box_base.setEnabled(True)
        else:
            self.image_label.set_image(self.current_raw_image, auto_fit=False)

    def auto_measure_roi(self, img_roi, mode):
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        corners = (float(thresh[0, 0]) + thresh[-1, -1]
                   + thresh[0, -1] + thresh[-1, 0]) / 4
        if corners > 127:
            thresh = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)

        if mode == "Vertical Needle":
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            widths, best_line = [], None
            mid_y = mask.shape[0] // 2
            for row in range(mask.shape[0]):
                cols = np.where(mask[row] > 0)[0]
                if len(cols):
                    widths.append(cols[-1] - cols[0])
                    if row == mid_y:
                        best_line = ((cols[0], row), (cols[-1], row))
            if not best_line and widths:
                r = np.where(mask > 0)[0][0]
                c = np.where(mask[r] > 0)[0]
                best_line = ((c[0], r), (c[-1], r))
            return ((np.median(widths), ('line', best_line))
                    if widths else (None, None))
        elif mode == "Sphere":
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            return (radius * 2, ('circle', (int(cx), int(cy)), int(radius)))
        return None, None

    # --- source loading ------------------------------------------------
    def upload_picture(self):
        self.stop_camera()
        self.is_video_file = False
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if fp:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is not None:
                self.setup_image_for_analysis(img)

    def upload_video(self):
        self.stop_camera()
        self.is_video_file = True
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if fp:
            self.capture = cv2.VideoCapture(fp)
            ret, frame = self.capture.read()
            if ret:
                self.setup_image_for_analysis(frame)
                QMessageBox.information(
                    self, "Video loaded",
                    "Video paused on first frame.\n"
                    "Complete calibration and baseline, then press "
                    "'Start time series' to play.")

    def start_camera(self):
        self.is_video_file = False
        try:
            self.live_camera.start()
            if self.radio_static.isChecked():
                self.btn_capture.show()
            self.btn_camera.setText("Stop camera")
            self.btn_camera.clicked.disconnect()
            self.btn_camera.clicked.connect(self.stop_camera)
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Camera Error",
                                 f"Failed to connect:\n{str(e)}")

    def stop_camera(self):
        self.timer.stop()
        if self.worker is not None and self.worker.running:
            self.toggle_time_series()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.live_camera.stop()
        self.btn_capture.hide()
        self.btn_camera.setText("Start live camera")
        try:
            self.btn_camera.clicked.disconnect()
        except TypeError:
            pass
        self.btn_camera.clicked.connect(self.start_camera)

    def capture_frame(self):
        if self.current_raw_image is not None and not self.is_video_file:
            self.stop_camera()
            self.setup_image_for_analysis(self.current_raw_image.copy())

    def update_frame(self):
        frame = None
        if self.is_video_file and self.capture is not None:
            ret, f = self.capture.read()
            if ret:
                frame = f
            else:
                if self.worker is not None:
                    self.toggle_time_series()
                    QMessageBox.information(self, "Done", "Video playback complete.")
                return
        elif not self.is_video_file:
            ret, f = self.live_camera.get_frame()
            if ret:
                frame = f
        if frame is not None:
            self.current_raw_image = frame
            self.display_image(frame)
            if self.worker is not None and self.worker.running:
                if self.is_video_file:
                    current_time = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                else:
                    current_time = time.time() - self.ts_start_wall_time
                if current_time - self.last_grab_time >= self.ts_interval:
                    if len(self.worker.frame_queue) < 5:
                        self.worker.frame_queue.append((current_time, frame.copy()))
                    self.last_grab_time = current_time

    # --- analysis hand-off --------------------------------------------
    def setup_image_for_analysis(self, img):
        self.clear_guide()
        self._clear_result_overlays()
        self._clear_baseline_overlay()
        self.current_raw_image = img
        self.image_label.set_image(img, auto_fit=True)

        self.btn_crop.setEnabled(True)
        self.btn_thresh.setEnabled(True)
        self.edges_df = extract_pendant_edges(img)
        if self.edges_df is None:
            return
        self.box_calib.setEnabled(True)
        self.lbl_calib_res.setText("Ready to calibrate.")
        self.box_base.setEnabled(False)
        self.box_analyze.setEnabled(False)
        self._clear_results_card()

    def clear_guide(self):
        if self.guide_item is not None:
            if self.guide_item.scene() == self.image_label.scene:
                self.image_label.scene.removeItem(self.guide_item)
            self.guide_item = None

    def run_baseline(self):
        self.clear_guide()
        mode = self.combo_base.currentText()
        if mode == "Auto":
            bly = attempt_auto_baseline(self.edges_df, self.width_px)
            if bly:
                self.baseline_y = bly
                self._set_base_label(f"Auto ✓   y = {bly}", success=True)
                self._paint_baseline(bly)
                self.box_analyze.setEnabled(True)
            else:
                self._set_base_label("Auto failed. Try manual.", success=False)
        else:
            self.click_mode = 'baseline'
            self.image_label.waiting_for_click = True
            self.lbl_base_res.setText("Click ONCE on the shoulder.")

    # --- mouse dispatch ------------------------------------------------
    def handle_mouse_move(self, x, y):
        if self.click_mode == 'calib' and len(self.calib_pts) == 1:
            if self.guide_item is None:
                self.guide_item = QGraphicsLineItem()
                self.guide_item.setPen(make_pixel_pen(Qt.GlobalColor.cyan))
                self.image_label.scene.addItem(self.guide_item)
            x1, y1 = self.calib_pts[0]
            self.guide_item.setLine(x1 + 0.5, y1 + 0.5, x + 0.5, y + 0.5)
        elif self.click_mode == 'baseline':
            if self.guide_item is None:
                self.guide_item = QGraphicsLineItem()
                self.guide_item.setPen(make_pixel_pen(Qt.GlobalColor.green))
                self.image_label.scene.addItem(self.guide_item)
            w = self.current_raw_image.shape[1]
            self.guide_item.setLine(0, y + 0.5, w, y + 0.5)

    def handle_image_click(self, x, y):
        if self.click_mode == 'calib':
            self.calib_pts.append((x, y))
            dot = QGraphicsRectItem(x, y, 1, 1)
            dot.setBrush(QBrush(Qt.GlobalColor.red))
            dot.setPen(QPen(Qt.GlobalColor.transparent))
            self.image_label.scene.addItem(dot)
            self.image_label.dynamic_items.append(dot)

            if len(self.calib_pts) == 2:
                self.image_label.clear_dynamic_drawings()
                self.image_label.waiting_for_click = False
                self.click_mode = None
                pixel_dist = np.hypot(self.calib_pts[0][0] - self.calib_pts[1][0],
                                      self.calib_pts[0][1] - self.calib_pts[1][1])
                if pixel_dist < 5:
                    self._set_calib_label("Line too short. Try again.",
                                          success=False)
                    return
                val, ok = QInputDialog.getDouble(
                    self, "Reference size",
                    f"Distance: {pixel_dist:.1f} px.\nEnter physical size (mm):",
                    self.needle_mm, decimals=4)
                if ok and val > 0:
                    self.pixel_to_m = (val / 1000.0) / pixel_dist
                    self.width_px = pixel_dist
                    self.needle_mm = val
                    self._set_calib_label(
                        f"Manual: {pixel_dist:.1f} px ≡ {val} mm",
                        success=True)
                    self.box_base.setEnabled(True)
                self.display_image(self.current_raw_image)

        elif self.click_mode == 'baseline':
            self.image_label.clear_dynamic_drawings()
            self.image_label.waiting_for_click = False
            self.click_mode = None
            self.baseline_y = y
            self._set_base_label(f"Manual ✓   y = {y}", success=True)
            self._paint_baseline(y)
            self.box_analyze.setEnabled(True)

    # =========================================================================
    # Analysis
    # =========================================================================
    def run_static_analysis(self):
        self.lbl_sigma_big.setText("…")
        self.lbl_sigma_unc.setText("Calculating…")
        self.lbl_qc_badge.setText("")
        self.lbl_meta.setText("")
        self.lbl_dsde.setText("")
        self.lbl_unc_breakdown.setText("")
        try:
            results = calculate_physics(
                self.edges_df, self.baseline_y, self.pixel_to_m,
                density_inner=self.density_inner,
                density_outer=self.density_outer,
                needle_tol_rel=self.needle_tol_rel,
                density_inner_tol_rel=self.rho_in_tol_rel,
                density_outer_tol_rel=self.rho_out_tol_rel)
            if not results.get('fit_success'):
                self.lbl_sigma_big.setText("—")
                self.lbl_sigma_unc.setText(
                    f"Analysis failed: {results.get('error', 'unknown')}")
                self.lbl_sigma_unc.setStyleSheet("color:#ff6a4d;")
                return
            results = compute_worthington(results, self.needle_mm / 1000.0)
            self.last_results = results

            # Clean pixmap, overlays painted as Qt items (cosmetic pens)
            self.display_image(self.current_raw_image)
            self._paint_baseline(self.baseline_y)
            self._draw_pendant_overlays(results)
            self._populate_results_card(results)

            # Enable domain building
            self.box_domain.setEnabled(True)
            self.lbl_domain_status.setText(
                "IFT analysis complete. Ready to build domain.")
            self.lbl_domain_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def toggle_time_series(self):
        if self.worker is None or not self.worker.running:
            if self.capture is None and self.live_camera.camera_type == "Unknown":
                QMessageBox.warning(self, "Warning",
                                    "Please start the camera or load a video first.")
                return
            self.time_data, self.ift_data, self.ift_err, self.qc_flags = [], [], [], []
            self.plot_line.setData([], [])
            self.plot_pts_pass.setData([], [])
            self.plot_pts_fail.setData([], [])
            self.err_item.setData(x=np.array([]), y=np.array([]),
                                  top=np.array([]), bottom=np.array([]))
            self.worker = AnalysisWorker(
                self.pixel_to_m, self.baseline_y,
                self.density_inner, self.density_outer,
                self.needle_tol_rel, self.rho_in_tol_rel, self.rho_out_tol_rel,
                self.needle_mm / 1000.0)
            self.worker.result_ready.connect(self.update_graph)
            self.worker.start()
            if self.is_video_file and self.capture is not None:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.timer.start(30)
            self.last_grab_time = -999.0
            self.ts_start_wall_time = time.time()
            self.btn_analyze_time.setText("■  Stop time series")
            self.btn_analyze_time.setStyleSheet(BTN_DANGER)
            self.box_calib.setEnabled(False)
            self.box_base.setEnabled(False)
        else:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            if self.is_video_file:
                self.timer.stop()
            self.btn_analyze_time.setText("▶  Start time series")
            self.btn_analyze_time.setStyleSheet(BTN_SUCCESS)
            self.box_calib.setEnabled(True)
            self.box_base.setEnabled(True)

    def update_graph(self, timestamp, results):
        ift  = results['ift_mNm']
        dift = results.get('ift_uncertainty_mNm', 0.0)
        qc   = results.get('qc_pass', True)
        self.time_data.append(timestamp)
        self.ift_data.append(ift)
        self.ift_err.append(dift)
        self.qc_flags.append(qc)

        x = np.array(self.time_data)
        y = np.array(self.ift_data)
        e = np.array(self.ift_err)
        flags = np.array(self.qc_flags, dtype=bool)
        self.plot_line.setData(x, y)
        self.plot_pts_pass.setData(x[flags], y[flags])
        self.plot_pts_fail.setData(x[~flags], y[~flags])
        self.err_item.setData(x=x, y=y, top=e, bottom=e)

        self._populate_results_card(results)
        self._draw_pendant_overlays(results)

    # =========================================================================
    # Results card plumbing
    # =========================================================================
    def _populate_results_card(self, r):
        sigma = r['ift_mNm']
        dsig  = r.get('ift_uncertainty_mNm')
        self.lbl_sigma_big.setText(f"σ  =  {sigma:0.2f}  mN/m")
        if dsig is not None:
            rel = r.get('ift_rel_uncertainty', 0) * 100
            self.lbl_sigma_unc.setText(f"±  {dsig:0.2f}  mN/m   ({rel:+.2f} %)")
            self.lbl_sigma_unc.setStyleSheet("color:#aab4d0;")

        qc_pass = r.get('qc_pass', True)
        if qc_pass:
            self.lbl_qc_badge.setText("✓  QC  PASS")
            self.lbl_qc_badge.setStyleSheet(
                "color:#4fd07b; font-weight:700;")
        else:
            fails = [name for name, ok, _ in r.get('qc_checks', []) if not ok]
            self.lbl_qc_badge.setText("⚠  QC  ISSUES — "
                                      + ", ".join(fails[:3])
                                      + ("…" if len(fails) > 3 else ""))
            self.lbl_qc_badge.setStyleSheet(
                "color:#ff9f5a; font-weight:700;")

        Wo = r.get('Wo')
        meta = (f"β   =  {r['beta']:.4f}\n"
                f"R₀  =  {r['R0_mm']:.4f} mm\n"
                f"V    =  {r['volume_ul']:.2f} µL\n"
                f"RMSE =  {r.get('rmse_px', 0):.2f} px")
        if Wo is not None:
            meta += f"\nWo   =  {Wo:.3f}"
        self.lbl_meta.setText(meta)

        sigma_dsde = r.get('ift_dsde_mNm')
        if sigma_dsde is not None:
            delta = sigma_dsde - sigma
            self.lbl_dsde.setText(
                f"DS/DE cross-check:  "
                f"{sigma_dsde:0.2f} mN/m  (Δ = {delta:+.2f})")

        u = r.get('uncertainty')
        if u is not None:
            self.lbl_unc_breakdown.setText(
                "Uncertainty budget  (mN/m)\n"
                f"  R₀ fit statistic      ±{u['c_R0_fit']:7.3f}\n"
                f"  Needle OD tolerance   ±{u['c_R0_calib']:7.3f}\n"
                f"  β  fit statistic      ±{u['c_beta']:7.3f}\n"
                f"  Density tolerance     ±{u['c_rho']:7.3f}")

    def _clear_results_card(self):
        self.lbl_sigma_big.setText("—")
        self.lbl_sigma_unc.setText("")
        self.lbl_qc_badge.setText("")
        self.lbl_meta.setText("")
        self.lbl_dsde.setText("")
        self.lbl_unc_breakdown.setText("")

    def _set_calib_label(self, text, success=True):
        self.lbl_calib_res.setText(text)
        self.lbl_calib_res.setStyleSheet(
            "color:#4fd07b; font-weight:600;" if success
            else "color:#ff6a4d; font-weight:600;")

    def _set_base_label(self, text, success=True):
        self.lbl_base_res.setText(text)
        self.lbl_base_res.setStyleSheet(
            "color:#4fd07b; font-weight:600;" if success
            else "color:#ff6a4d; font-weight:600;")

    def display_image(self, img):
        self.image_label.set_image(img, auto_fit=False)

    # =========================================================================
    # Domain builder
    # =========================================================================
    def build_domain(self):
        """Build the FEM domain polygon from the current contour data."""
        if self.edges_df is None or self.baseline_y is None or self.pixel_to_m is None:
            QMessageBox.warning(self, "Missing data",
                                "Please complete calibration, baseline, and "
                                "IFT analysis before building the domain.")
            return
        try:
            polygon, metadata = contour_to_domain(
                self.edges_df,
                self.baseline_y,
                self.pixel_to_m,
                needle_od_mm=self.needle_mm,
                needle_id_mm=self.needle_id_mm,
            )
            n = metadata['n_points']
            h = metadata['drop_height_m'] * 1e3
            self.lbl_domain_status.setText(
                f"✓ Domain built — {n} points, "
                f"drop height {h:.3f} mm")
            self.lbl_domain_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")

            # Store on self for downstream use (meshing)
            self.domain_polygon = polygon
            self.domain_metadata = metadata

            # Enable meshing
            self.box_mesh.setEnabled(True)
            self.lbl_mesh_status.setText(
                "Domain ready. Click to generate mesh.")
            self.lbl_mesh_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")

            # Open visualisation window
            self._domain_window = DomainWindow(polygon, metadata, parent=None)
            self._domain_window.show()

        except Exception as e:
            self.lbl_domain_status.setText(f"Domain build failed: {e}")
            self.lbl_domain_status.setStyleSheet(
                "color:#ff6a4d; font-weight:600;")
            QMessageBox.critical(self, "Domain Error", str(e))

    # =========================================================================
    # Mesh generation
    # =========================================================================
    def run_meshing(self):
        """Generate a triangular FEM mesh from the domain polygon."""
        if not hasattr(self, 'domain_polygon') or self.domain_polygon is None:
            QMessageBox.warning(self, "No domain",
                                "Build the domain first (Step 4).")
            return
        self.lbl_mesh_status.setText("⏳ Generating mesh…")
        self.lbl_mesh_status.setStyleSheet("color:#8ab4ff; font-weight:600;")
        self.btn_generate_mesh.setEnabled(False)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            points, triangles, boundary_edges, quality = generate_mesh(
                self.domain_polygon,
                domain_metadata=self.domain_metadata,
                verbose=True)

            self.mesh_points = points
            self.mesh_triangles = triangles
            self.mesh_boundary_edges = boundary_edges
            self.mesh_quality = quality

            n_nodes = quality['n_nodes']
            n_elem = quality['n_elements']
            min_ang = quality['min_angle']
            ar = quality['area_ratio']
            self.lbl_mesh_status.setText(
                f"✓ {n_nodes:,} nodes, {n_elem:,} elements\n"
                f"  min∠ {min_ang:.1f}°  area ratio {ar:.0f}×")
            self.lbl_mesh_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")

            # Enable FEM
            self.box_fem.setEnabled(True)
            self.lbl_fem_status.setText(
                "Mesh ready. Click to assemble FEM matrices.")
            self.lbl_fem_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")

            self._mesh_window = MeshWindow(
                points, triangles, boundary_edges,
                self.domain_polygon, self.domain_metadata,
                parent=None)
            self._mesh_window.show()

        except Exception as e:
            self.lbl_mesh_status.setText(f"Mesh failed: {e}")
            self.lbl_mesh_status.setStyleSheet(
                "color:#ff6a4d; font-weight:600;")
            QMessageBox.critical(self, "Mesh Error", str(e))
        finally:
            self.btn_generate_mesh.setEnabled(True)

    # =========================================================================
    # FEM Assembly
    # =========================================================================
    def run_fem(self):
        """Assemble the FEM system matrices."""
        if not hasattr(self, 'mesh_points') or self.mesh_points is None:
            QMessageBox.warning(self, "No mesh",
                                "Generate the mesh first (Step 5).")
            return
        self.lbl_fem_status.setText("⏳ Assembling FEM matrices…")
        self.lbl_fem_status.setStyleSheet("color:#8ab4ff; font-weight:600;")
        self.btn_run_fem.setEnabled(False)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            fem = assemble_fem_system(
                self.mesh_points,
                self.mesh_triangles,
                self.mesh_boundary_edges,
                self.domain_metadata,
            )

            self.fem_results = fem
            s = fem['sanity']
            status = "✓" if s['all_pass'] else "⚠"
            n = fem['H'].shape[0]
            self.lbl_fem_status.setText(
                f"{status} FEM assembled: {n}×{n}\n"
                f"  Interface: {s['n_interface_nodes']} nodes\n"
                f"  Wall: {s['n_wall_nodes']} nodes")
            self.lbl_fem_status.setStyleSheet(
                f"color:{'#4fd07b' if s['all_pass'] else '#ff9f5a'};"
                f" font-weight:600;")

            self._fem_window = FEMWindow(
                fem, self.domain_polygon, self.domain_metadata,
                parent=None)
            self._fem_window.show()

            # Enable simulation + parameter estimation
            self.box_sim.setEnabled(True)
            self.lbl_sim_status.setText(
                "FEM ready. Click to run simulation.")
            self.lbl_sim_status.setStyleSheet(
                "color:#4fd07b; font-weight:600;")
            self.box_est.setEnabled(True)
            self.lbl_est_status.setText(
                "Ready. Fits the forward model to the measured IFT curve.")
            self.lbl_est_status.setStyleSheet("color:#4fd07b;")

        except Exception as e:
            self.lbl_fem_status.setText(f"FEM failed: {e}")
            self.lbl_fem_status.setStyleSheet(
                "color:#ff6a4d; font-weight:600;")
            QMessageBox.critical(self, "FEM Error", str(e))
        finally:
            self.btn_run_fem.setEnabled(True)

    # =========================================================================
    # Forward Simulation
    # =========================================================================
    # =========================================================================
    # Calibration curve
    # =========================================================================
    def download_calib_template(self):
        """Save a blank template for the user to fill in."""
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save calibration template",
            "calibration_template.csv", "CSV files (*.csv)")
        if not dest:
            return
        try:
            shutil.copyfile(CALIB_TEMPLATE_PATH, dest)
        except OSError as e:
            QMessageBox.critical(self, "Could not save template", str(e))
            return
        self.lbl_calib_status.setText(
            f"Template saved to {dest} — fill it in, then Upload.")
        self.lbl_calib_status.setStyleSheet("color:#8ab4ff;")

    def upload_calibration(self):
        """Validate a filled template and, only if it fully passes, make
        it the active calibration.  A rejected or failed file leaves the
        previously-loaded calibration untouched (all-or-nothing)."""
        src, _ = QFileDialog.getOpenFileName(
            self, "Upload calibration curve", "",
            "CSV files (*.csv);;All files (*)")
        if not src:
            return
        try:
            # Commits to the single canonical file only on full success.
            calibration, warnings = install_calibration(src)
        except CalibrationError as e:
            # Prior self.calibration is deliberately left as-is.
            QMessageBox.warning(self, "Calibration not loaded", str(e))
            return
        except Exception as e:                       # unexpected — stay safe
            QMessageBox.critical(self, "Calibration error", str(e))
            return

        self.calibration = calibration
        n = len(calibration.c_points)
        self.lbl_calib_status.setText(
            f"✓ Calibration loaded: {n} points, "
            f"c_sat = {calibration.c_sat:g}, "
            f"γ {calibration.gamma0:g} → {calibration.gamma_inf:g} mN/m. "
            f"Steps 7 & 8 will use it.")
        self.lbl_calib_status.setStyleSheet("color:#4fd07b; font-weight:600;")
        if warnings:
            QMessageBox.information(
                self, "Calibration loaded (with notes)",
                "The curve was accepted. Please check:\n\n• "
                + "\n• ".join(warnings))

    def reset_calibration(self):
        """Forget the loaded calibration — in memory and on disk — so no
        stale curve can leak into a later run.  Reverts Steps 7 & 8 to the
        placeholder linear γ(Cs) model.  Confirms first, since the curve
        must then be re-uploaded."""
        if self.calibration is None and not ACTIVE_CALIBRATION_PATH.exists():
            return                          # nothing to clear
        if QMessageBox.question(
                self, "Clear calibration?",
                "This deletes the loaded calibration curve (both in memory "
                "and the saved copy). Steps 7 & 8 will fall back to the "
                "placeholder linear model, and you will have to upload the "
                "CSV again to use a calibration.\n\nClear it now?"
                ) != QMessageBox.StandardButton.Yes:
            return
        self.calibration = None
        clear_active_calibration()          # remove the canonical file too
        self.lbl_calib_status.setText(
            "No curve loaded — using placeholder linear γ(Cs) model.")
        self.lbl_calib_status.setStyleSheet("color:#9aa3b8;")

    def restart_application(self):
        """Relaunch the whole application with a clean state.

        Spawns a fresh process from the current interpreter/argv, then
        quits this one.  A running fit/simulation would be abandoned, so
        confirm first."""
        if QMessageBox.question(
                self, "Restart application?",
                "This closes the current window and starts a fresh one. "
                "Any unsaved work and in-progress runs will be lost.\n\n"
                "Continue?") != QMessageBox.StandardButton.Yes:
            return
        # startDetached launches the new process independently of this one;
        # it is up and running before we tell the old app to quit.
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

    def run_simulation(self):
        """Start (or cancel) the forward diffusion simulation."""
        # A second click while running acts as Cancel.
        if self.sim_worker is not None and self.sim_worker.isRunning():
            self.sim_worker.cancel()
            self.btn_run_sim.setEnabled(False)
            self.lbl_sim_status.setText("⏳ Cancelling…")
            self.lbl_sim_status.setStyleSheet("color:#8ab4ff; font-weight:600;")
            return

        if not hasattr(self, 'fem_results') or self.fem_results is None:
            QMessageBox.warning(self, "No FEM",
                                "Assemble FEM matrices first (Step 6).")
            return

        try:
            duration_min = float(self.sim_duration_input.text())
            dtau = float(self.sim_dtau_input.text())
            if duration_min <= 0 or dtau <= 0:
                raise ValueError("Duration and Δt must be positive.")
        except ValueError:
            QMessageBox.warning(self, "Invalid input",
                                "Enter positive numbers for duration and Δt.")
            return

        self.sim_progress.setValue(0)
        self.sim_progress.setVisible(True)
        self.lbl_sim_status.setText("⏳ Running simulation…")
        self.lbl_sim_status.setStyleSheet("color:#8ab4ff; font-weight:600;")
        self.btn_run_sim.setText("■  Cancel")
        self.btn_run_sim.setStyleSheet(BTN_DANGER)

        self.sim_worker = SimulationWorker(
            self.fem_results, self.domain_metadata,
            dtau=dtau, total_time=duration_min * 60.0,
            calibration=self.calibration)
        self.sim_worker.progress.connect(self._on_sim_progress)
        self.sim_worker.finished_ok.connect(self._on_sim_finished)
        self.sim_worker.failed.connect(self._on_sim_failed)
        self.sim_worker.cancelled.connect(self._on_sim_cancelled)
        self.sim_worker.start()

    def _reset_sim_ui(self):
        self.btn_run_sim.setText("▶  Run Simulation")
        self.btn_run_sim.setStyleSheet(BTN_SUCCESS)
        self.btn_run_sim.setEnabled(True)
        self.sim_progress.setVisible(False)
        self.sim_worker = None

    def _on_sim_progress(self, step, n_steps):
        self.sim_progress.setValue(int(100 * step / n_steps))

    def _on_sim_finished(self, sim):
        self.sim_results = sim
        Cs_final = sim['Cs_history'][-1]
        gamma_final = sim['gamma_history'][-1]
        self.lbl_sim_status.setText(
            f"✓ Simulation complete ({sim['n_steps']} steps, "
            f"{sim['actual_time']/60.0:.2f} min simulated)\n"
            f"  Cs = {Cs_final:.4f}  γ = {gamma_final:.1f} mN/m")
        self.lbl_sim_status.setStyleSheet("color:#4fd07b; font-weight:600;")
        self._reset_sim_ui()
        self._solver_window = SolverWindow(
            sim, self.domain_metadata, parent=None)
        self._solver_window.show()

    def _on_sim_failed(self, message):
        self.lbl_sim_status.setText(f"Simulation failed: {message}")
        self.lbl_sim_status.setStyleSheet("color:#ff6a4d; font-weight:600;")
        QMessageBox.critical(self, "Simulation Error", message)
        self._reset_sim_ui()

    def _on_sim_cancelled(self):
        self.lbl_sim_status.setText("Simulation cancelled.")
        self.lbl_sim_status.setStyleSheet("color:#9aa3b8;")
        self._reset_sim_ui()

    # =========================================================================
    # Parameter estimation (inverse problem)
    # =========================================================================
    # Minimum number of usable (QC-passed) points the fit needs.
    MIN_FIT_POINTS = 10

    def _fit_input_data(self):
        """Measured curve restricted to points the fit can actually use.

        Drops QC-failed measurements and any non-positive / non-finite IFT:
        a failed drop-shape fit can report γ ≤ 0, and the default objective
        (Yang eq 14) divides by γ, so a single such point poisons every
        residual.  Returns ``(t, gamma, n_dropped)``.
        """
        t = np.asarray(self.time_data, dtype=float)
        g = np.asarray(self.ift_data, dtype=float)
        qc = np.asarray(self.qc_flags, dtype=bool)
        keep = qc & np.isfinite(g) & (g > 0.0) & np.isfinite(t)
        return t[keep], g[keep], int(np.sum(~keep))

    def run_estimation(self):
        """Start (or cancel) the (D, k) fit against the measured IFT curve."""
        # A second click while running acts as Cancel.
        if self.est_worker is not None and self.est_worker.isRunning():
            self.est_worker.cancel()
            self.btn_estimate.setEnabled(False)
            self.lbl_est_status.setText("⏳ Cancelling…")
            self.lbl_est_status.setStyleSheet("color:#8ab4ff; font-weight:600;")
            return

        if not hasattr(self, 'fem_results') or self.fem_results is None:
            QMessageBox.warning(self, "No FEM",
                                "Assemble FEM matrices first (Step 6).")
            return

        # Fit only QC-passed points — see _fit_input_data.
        t_exp, gamma_exp, n_dropped = self._fit_input_data()
        if len(t_exp) < self.MIN_FIT_POINTS:
            QMessageBox.warning(
                self, "Not enough usable data",
                f"The fit needs at least {self.MIN_FIT_POINTS} QC-passed "
                f"points from the time-series analysis (Step 3).\n\n"
                f"Measured: {len(self.time_data)}   "
                f"usable: {len(t_exp)}   discarded: {n_dropped}.")
            return

        # Search-range mode: Auto (general defaults + auto-widen) or Manual.
        if self.radio_range_manual.isChecked():
            try:
                D_lo = float(self.in_D_min.text())
                D_hi = float(self.in_D_max.text())
                kD_lo = float(self.in_kD_min.text())
                kD_hi = float(self.in_kD_max.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid range",
                                    "Enter numbers for all four range fields.")
                return
            if not (0 < D_lo < D_hi) or not (0 < kD_lo < kD_hi):
                QMessageBox.warning(
                    self, "Invalid range",
                    "Each range needs 0 < min < max (D and kD).")
                return
            kD_range, D_bounds, auto_widen = (kD_lo, kD_hi), (D_lo, D_hi), False
        else:
            kD_range, D_bounds, auto_widen = None, None, True

        # γ_fit comes back on this grid, so keep it for the overlay plot.
        self._est_t_exp = t_exp

        dropped = (f", {n_dropped} QC-failed point(s) discarded"
                   if n_dropped else "")
        self.btn_estimate.setText("■  Cancel")
        self.btn_estimate.setStyleSheet(BTN_DANGER)
        self.lbl_est_status.setText(
            f"⏳ Fitting {len(t_exp)} points{dropped}… "
            f"(runs many forward sims)")
        self.lbl_est_status.setStyleSheet("color:#8ab4ff; font-weight:600;")

        self.est_worker = EstimationWorker(
            self.fem_results, self.domain_metadata, t_exp, gamma_exp,
            calibration=self.calibration,
            kD_range=kD_range, D_bounds=D_bounds, auto_widen=auto_widen)
        self.est_worker.finished_ok.connect(self._on_est_finished)
        self.est_worker.failed.connect(self._on_est_failed)
        self.est_worker.cancelled.connect(self._on_est_cancelled)
        self.est_worker.start()

    def _reset_est_ui(self):
        self.btn_estimate.setText("⇆  Fit D, k to measured IFT")
        self.btn_estimate.setStyleSheet(BTN_PRIMARY)
        self.btn_estimate.setEnabled(True)
        self.est_worker = None

    def _on_est_finished(self, result):
        status = "✓" if result['success'] else "⚠"
        n_sims = result['n_evaluations'] + result.get('n_scan_evaluations', 0)
        self.lbl_est_status.setText(
            f"{status} D = {result['D']:.3e} m²/s\n"
            f"   k = {result['k']:.3e} m/s\n"
            f"   kD = {result['kD']:.3f}  (Biot number)\n"
            f"   E = {result['E_percent']:.2f} %  (Yang eq 14 minimum)\n"
            f"   RMS misfit {result['residual_rms']:.3f} mN/m "
            f"({n_sims} simulations)\n"
            f"   {result['message']}")
        self.lbl_est_status.setStyleSheet(
            f"color:{'#4fd07b' if result['success'] else '#ff9f5a'};"
            f" font-weight:600;")
        # Overlay the fitted curve on the measured IFT plot.  Plot against
        # the grid the fit actually used, not the full measured series —
        # QC-failed points were dropped, so the lengths differ.
        self.plot_fit_line.setData(self._est_t_exp, result['gamma_fit'])
        self._reset_est_ui()

    def _on_est_failed(self, message):
        self.lbl_est_status.setText(f"Estimation failed: {message}")
        self.lbl_est_status.setStyleSheet("color:#ff6a4d; font-weight:600;")
        QMessageBox.critical(self, "Estimation Error", message)
        self._reset_est_ui()

    def _on_est_cancelled(self):
        self.lbl_est_status.setText("Estimation cancelled.")
        self.lbl_est_status.setStyleSheet("color:#9aa3b8;")
        self._reset_est_ui()

    def closeEvent(self, event):
        self.stop_camera()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        # Cancel before waiting: both workers poll a flag between forward
        # simulations, so they unwind in ~one sim.  Waiting without
        # cancelling would block the close for the whole run (minutes).
        if self.sim_worker is not None and self.sim_worker.isRunning():
            self.sim_worker.cancel()
            self.sim_worker.wait()
        if self.est_worker is not None and self.est_worker.isRunning():
            self.est_worker.cancel()
            self.est_worker.wait()
        event.accept()
