"""
gui.sessile_window
==================

Sessile-drop main window.

Updates in this revision
------------------------
    · Both CONTOUR-tangent angles AND CIRCLE-tangent angles are shown.
      Contour is the primary/default (physically meaningful); the
      weighted fitted circle is the reference overlay.
    · New toggle (Contour / Both / Circle) in the analyze card lets the
      user choose what gets drawn and what's highlighted in the results.
    · Detected contact points come from the actual contour — they never
      drift outside the ROI.
    · The LSQ circle is anchored at apex + two contacts, so it also
      stays inside a sensibly drawn ROI.

Visual layer
------------
All overlays are Qt graphics items with cosmetic pens so lines stay
visible at every zoom level; all numeric readout lives in the right-
panel results card.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsLineItem, QGraphicsEllipseItem,
                             QGraphicsRectItem, QGraphicsPathItem, QGroupBox,
                             QComboBox, QInputDialog, QFrame, QScrollArea,
                             QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
                         QPen, QBrush, QColor, QPainterPath)

from core.sessile_calcs import analyze_sessile_drop
from core.camera_handler import CameraHandler
from gui.threshold_dialog import ThresholdDialog
from gui.settings_dialog import SettingsDialog
from gui.styles import (DARK_STYLESHEET, BTN_PRIMARY, BTN_SUCCESS, BTN_ICON,
                        make_cosmetic_pen, make_pixel_pen)


# Overlay palette
_COL_CONTOUR_PTS = QColor(100, 200, 255)       # detected contour cloud
_COL_BASELINE    = QColor(  0, 220,  80)
_COL_ROI         = QColor(200, 120, 255)

# Fitted circle (secondary reference)
_COL_CIRCLE      = QColor(255, 180,  60)

# Tangents — warmer for contour (primary), cooler for circle (reference)
_COL_TANG_C_L    = QColor(255, 200,  60)       # contour tangent, left  — amber
_COL_TANG_C_R    = QColor(255, 130, 110)       # contour tangent, right — coral
_COL_TANG_S_L    = QColor(120, 190, 255)       # circle  tangent, left  — sky
_COL_TANG_S_R    = QColor(190, 130, 255)       # circle  tangent, right — violet

_COL_CONTACT     = QColor(255, 235,  80)
_COL_APEX        = QColor(180, 255, 180)


# =============================================================================
# Graphics view
# =============================================================================
class ZoomableGraphicsView(QGraphicsView):
    calib_ready        = pyqtSignal(int, int, int, int)
    calib_box_ready    = pyqtSignal(int, int, int, int)
    baseline_1pt_ready = pyqtSignal(int)
    baseline_2pt_ready = pyqtSignal(int, int, int, int)
    roi_poly_ready     = pyqtSignal(list)

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.interaction_mode = 'idle'
        self.baseline_data = None
        self.click_points = []
        self.dynamic_items = []
        self.crop_start = None
        self.crop_rect_item = None
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

    def draw_pixel_dot(self, x, y, color):
        dot = QGraphicsRectItem(x, y, 1, 1)
        dot.setBrush(QBrush(color))
        dot.setPen(QPen(Qt.GlobalColor.transparent))
        self.scene.addItem(dot)
        self.dynamic_items.append(dot)

    def set_mode(self, mode, baseline_data=None):
        self.interaction_mode = mode
        self.baseline_data = baseline_data
        self.click_points.clear()
        self.clear_dynamic_drawings()
        if mode != 'idle':
            self.setFocus()

    def get_crop_rect(self):
        if self.crop_rect_item:
            r = self.crop_rect_item.rect()
            return int(r.x()), int(r.y()), int(r.width()), int(r.height())
        return None

    def get_baseline_y(self, x):
        if not self.baseline_data:
            return 0
        if self.baseline_data[0] == '1pt':
            return self.baseline_data[1]
        x1, y1, x2, y2 = self.baseline_data[1]
        if x1 == x2: x2 += 1e-5
        m = (y2 - y1) / (x2 - x1)
        return m * x + (y1 - m * x1)

    def get_baseline_intersection(self, px, py, qx, qy):
        if not self.baseline_data:
            return qx, qy
        if self.baseline_data[0] == '1pt':
            m1, c1 = 0.0, self.baseline_data[1]
        else:
            bx1, by1, bx2, by2 = self.baseline_data[1]
            if bx1 == bx2: bx2 += 1e-5
            m1 = (by2 - by1) / (bx2 - bx1)
            c1 = by1 - m1 * bx1
        A1, B1, C1 = -m1, 1.0, c1
        A2 = qy - py; B2 = px - qx
        C2 = A2 * px + B2 * py
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-5:
            return qx, qy
        ix = (B2 * C1 - B1 * C2) / det
        iy = (A1 * C2 - A2 * C1) / det
        return ix, iy

    def finish_roi(self):
        if self.interaction_mode == 'roi' and len(self.click_points) >= 3:
            self.roi_poly_ready.emit(self.click_points.copy())
            self.set_mode('idle')

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish_roi()
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.image_item.pixmap().isNull():
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.interaction_mode != 'idle' and not self.image_item.pixmap().isNull():
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

                elif self.interaction_mode == 'base1':
                    self.baseline_1pt_ready.emit(y)
                    self.set_mode('idle')

                elif self.interaction_mode in ('base2', 'calib'):
                    self.click_points.append((x, y))
                    if len(self.click_points) == 2:
                        p1, p2 = self.click_points
                        if self.interaction_mode == 'calib':
                            self.calib_ready.emit(p1[0], p1[1], p2[0], p2[1])
                        else:
                            self.baseline_2pt_ready.emit(p1[0], p1[1], p2[0], p2[1])
                        self.set_mode('idle')

                elif self.interaction_mode == 'roi':
                    if len(self.click_points) == 0:
                        y_base = self.get_baseline_y(x)
                        self.click_points.append((x, int(y_base)))
                        self.draw_pixel_dot(x, int(y_base), Qt.GlobalColor.magenta)
                    else:
                        y_base = self.get_baseline_y(x)
                        if y >= y_base:
                            last_x, last_y = self.click_points[-1]
                            ix, iy = self.get_baseline_intersection(last_x, last_y, x, y)
                            self.click_points.append((int(ix), int(iy)))
                            self.roi_poly_ready.emit(self.click_points.copy())
                            self.set_mode('idle')
                        else:
                            self.click_points.append((x, y))
                            self.draw_pixel_dot(x, y, Qt.GlobalColor.magenta)
            super().mousePressEvent(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_panning:
            dx = event.pos().x() - self._pan_start.x()
            dy = event.pos().y() - self._pan_start.y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
            self._pan_start = event.pos()
            event.accept()
        else:
            if self.interaction_mode != 'idle' and not self.image_item.pixmap().isNull():
                pos = self.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())

                if self.interaction_mode in ('crop', 'calib_box') and self.crop_start is not None:
                    sx, sy = self.crop_start
                    rect = QRectF(min(sx, x) + 0.5, min(sy, y) + 0.5,
                                  abs(x - sx), abs(y - sy))
                    self.crop_rect_item.setRect(rect)
                else:
                    guides = [item for item in self.dynamic_items
                              if isinstance(item, QGraphicsLineItem)]
                    for g in guides:
                        if g.scene() == self.scene:
                            self.scene.removeItem(g)
                            self.dynamic_items.remove(g)

                    if self.interaction_mode == 'base1':
                        w = self.image_item.pixmap().width()
                        line = QGraphicsLineItem(0, y + 0.5, w, y + 0.5)
                        line.setPen(make_pixel_pen(Qt.GlobalColor.green))
                        self.scene.addItem(line)
                        self.dynamic_items.append(line)
                    elif self.interaction_mode in ('base2', 'calib'):
                        if len(self.click_points) == 1:
                            p1 = self.click_points[0]
                            line = QGraphicsLineItem(p1[0] + 0.5, p1[1] + 0.5,
                                                     x + 0.5, y + 0.5)
                            color = (Qt.GlobalColor.cyan if self.interaction_mode == 'calib'
                                     else Qt.GlobalColor.green)
                            line.setPen(make_pixel_pen(color))
                            self.scene.addItem(line)
                            self.dynamic_items.append(line)
                    elif self.interaction_mode == 'roi':
                        for i in range(1, len(self.click_points)):
                            p1, p2 = self.click_points[i - 1], self.click_points[i]
                            line = QGraphicsLineItem(p1[0] + 0.5, p1[1] + 0.5,
                                                     p2[0] + 0.5, p2[1] + 0.5)
                            line.setPen(make_pixel_pen(Qt.GlobalColor.magenta))
                            self.scene.addItem(line)
                            self.dynamic_items.append(line)
                        if self.click_points:
                            p_last = self.click_points[-1]
                            disp_x, disp_y = x, y
                            y_base = self.get_baseline_y(x)
                            if y >= y_base:
                                ix, iy = self.get_baseline_intersection(
                                    p_last[0], p_last[1], x, y)
                                disp_x, disp_y = int(ix), int(iy)
                            line = QGraphicsLineItem(p_last[0] + 0.5, p_last[1] + 0.5,
                                                     disp_x + 0.5, disp_y + 0.5)
                            line.setPen(make_pixel_pen(Qt.GlobalColor.magenta))
                            self.scene.addItem(line)
                            self.dynamic_items.append(line)
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
        if event.button() == Qt.MouseButton.RightButton and not self.interaction_mode == 'roi':
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# =============================================================================
# Sessile window
# =============================================================================
class SessileWindow(QWidget):
    def __init__(self, camera_type="Basler"):
        super().__init__()
        self.setWindowTitle("Sessile Drop Analysis")
        self.resize(1300, 880)
        self.setStyleSheet(DARK_STYLESHEET)

        # state
        self.camera = CameraHandler(camera_type)
        self.timer = QTimer(); self.timer.timeout.connect(self.update_frame)
        self.current_raw_image = None
        self.pixel_to_m = None
        self.needle_mm = 0.7176
        self.density_inner = 1000.0
        self.density_outer = 1.204
        self.ts_interval = 1.0
        self.needle_tol_rel = 0.010
        self.rho_in_tol_rel = 0.0005
        self.rho_out_tol_rel = 0.020

        self.baseline_data = None
        self.roi_points = None
        self.last_results = None

        # display toggle — one of 'contour', 'both', 'circle'
        self.display_mode = 'contour'

        self._result_overlays = []
        self._baseline_overlay_items = []
        self._roi_overlay_items = []

        # --- root layout ----------------------------------------------
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.image_label = ZoomableGraphicsView()
        self.image_label.setStyleSheet(
            "QGraphicsView{background-color:#0f1018;border:1px solid #343852;"
            "border-radius:6px;}")
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label, stretch=3)

        self.image_label.calib_ready.connect(self.handle_calib_pts)
        self.image_label.calib_box_ready.connect(self.handle_calib_box)
        self.image_label.baseline_1pt_ready.connect(self.handle_baseline_1pt)
        self.image_label.baseline_2pt_ready.connect(self.handle_baseline_2pt)
        self.image_label.roi_poly_ready.connect(self.handle_roi_poly)

        # scrollable right panel
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

        # Header + ⚙
        hdr_row = QHBoxLayout()
        title = QLabel("Contact Angle Analysis")
        title.setStyleSheet("font-size:11pt; font-weight:600; color:#9ec4ff;")
        hdr_row.addWidget(title)
        hdr_row.addStretch(1)
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedWidth(40)
        self.btn_settings.setToolTip("Physical parameters and tolerances")
        self.btn_settings.setStyleSheet(BTN_ICON)
        hdr_row.addWidget(self.btn_settings)
        hdr_box = QGroupBox()
        hdr_box.setLayout(hdr_row)
        right_panel.addWidget(hdr_box)

        # Source & prep
        box_source = QGroupBox("Source input and preparation")
        l_source = QVBoxLayout()
        h_source = QHBoxLayout()
        self.btn_upload = QPushButton("Upload picture")
        self.btn_live   = QPushButton("Start live camera")
        h_source.addWidget(self.btn_upload)
        h_source.addWidget(self.btn_live)
        self.btn_capture = QPushButton("📸  Capture frame")
        self.btn_capture.hide()
        h_prep = QHBoxLayout()
        self.btn_crop       = QPushButton("✂  Crop")
        self.btn_apply_crop = QPushButton("✓  Apply")
        self.btn_apply_crop.setStyleSheet(BTN_SUCCESS)
        self.btn_thresh     = QPushButton("🎛  Threshold…")
        self.btn_crop.setEnabled(False)
        self.btn_thresh.setEnabled(False)
        self.btn_apply_crop.hide()
        h_prep.addWidget(self.btn_crop)
        h_prep.addWidget(self.btn_apply_crop)
        h_prep.addWidget(self.btn_thresh)
        l_source.addLayout(h_source)
        l_source.addWidget(self.btn_capture)
        l_source.addLayout(h_prep)
        box_source.setLayout(l_source)
        right_panel.addWidget(box_source)

        # Step 1 | Step 2 side by side
        calib_base_row = QHBoxLayout()
        calib_base_row.setSpacing(8)
        box_calib = QGroupBox("Step 1 · Calibration")
        l_calib = QVBoxLayout()
        self.combo_calib = QComboBox()
        self.combo_calib.addItems(
            ["Vertical Needle", "Horizontal Rectangle", "Sphere",
             "Misc", "Direct Input"])
        h_calib_btns = QHBoxLayout()
        self.btn_calib_auto   = QPushButton("Auto")
        self.btn_calib_manual = QPushButton("Manual")
        h_calib_btns.addWidget(self.btn_calib_auto)
        h_calib_btns.addWidget(self.btn_calib_manual)
        self.lbl_calib_status = QLabel("No scale set.")
        self.lbl_calib_status.setWordWrap(True)
        l_calib.addWidget(self.combo_calib)
        l_calib.addLayout(h_calib_btns)
        l_calib.addWidget(self.lbl_calib_status)
        box_calib.setLayout(l_calib)

        self.box_base = QGroupBox("Step 2 · Baseline")
        l_base = QVBoxLayout()
        self.btn_base1 = QPushButton("1-point (flat)")
        self.btn_base2 = QPushButton("2-point (tilted)")
        self.lbl_base_status = QLabel("Not set.")
        self.lbl_base_status.setWordWrap(True)
        l_base.addWidget(self.btn_base1)
        l_base.addWidget(self.btn_base2)
        l_base.addWidget(self.lbl_base_status)
        self.box_base.setLayout(l_base)

        calib_base_row.addWidget(box_calib, 1)
        calib_base_row.addWidget(self.box_base, 1)
        right_panel.addLayout(calib_base_row)

        # Step 3 ROI
        box_roi = QGroupBox("Step 3 · ROI mask  (optional)")
        l_roi = QHBoxLayout()
        self.btn_roi_draw  = QPushButton("Draw polygon")
        self.btn_roi_clear = QPushButton("Clear")
        l_roi.addWidget(self.btn_roi_draw)
        l_roi.addWidget(self.btn_roi_clear)
        box_roi_outer = QVBoxLayout()
        box_roi_outer.addLayout(l_roi)
        self.lbl_roi_status = QLabel("No ROI set.")
        self.lbl_roi_status.setWordWrap(True)
        box_roi_outer.addWidget(self.lbl_roi_status)
        box_roi.setLayout(box_roi_outer)
        right_panel.addWidget(box_roi)

        # Step 4 Analyze + toggle + results card
        box_analyze = QGroupBox("Step 4 · Analyze")
        l_analyze = QVBoxLayout(); l_analyze.setSpacing(6)
        self.btn_analyze = QPushButton("Calculate contact angles")
        self.btn_analyze.setStyleSheet(BTN_PRIMARY)
        l_analyze.addWidget(self.btn_analyze)

        # --- Display toggle  (Contour / Both / Circle) ---
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(10)
        lbl_show = QLabel("Show:")
        lbl_show.setStyleSheet("color:#9ec4ff;")
        self.radio_contour = QRadioButton("Contour")
        self.radio_both    = QRadioButton("Both")
        self.radio_circle  = QRadioButton("Fitted circle")
        self.radio_contour.setChecked(True)
        self._display_group = QButtonGroup(self)
        self._display_group.addButton(self.radio_contour)
        self._display_group.addButton(self.radio_both)
        self._display_group.addButton(self.radio_circle)
        toggle_row.addWidget(lbl_show)
        toggle_row.addWidget(self.radio_contour)
        toggle_row.addWidget(self.radio_both)
        toggle_row.addWidget(self.radio_circle)
        toggle_row.addStretch(1)
        l_analyze.addLayout(toggle_row)

        # Results card
        self.results_card = QFrame(objectName="card")
        rc = QVBoxLayout(self.results_card)
        rc.setContentsMargins(12, 10, 12, 10); rc.setSpacing(3)

        self.lbl_theta_big = QLabel("—")
        self.lbl_theta_big.setStyleSheet(
            "font-size:22pt; font-weight:700; color:#ffffff;")
        self.lbl_theta_method = QLabel("")
        self.lbl_theta_method.setStyleSheet("color:#8ab4ff; font-weight:600;")
        self.lbl_theta_sides = QLabel("")
        self.lbl_theta_sides.setStyleSheet("color:#aab4d0;")

        # Monospace comparison block for both methods
        self.lbl_both_methods = QLabel("")
        self.lbl_both_methods.setStyleSheet(
            "color:#8e95ae; font-family:'Consolas','Menlo',monospace; "
            "font-size:9.5pt;")

        self.lbl_qc_badge = QLabel("")
        self.lbl_qc_badge.setStyleSheet("color:#9aa3b8; font-weight:600;")

        self.lbl_geom = QLabel("")
        self.lbl_geom.setStyleSheet("color:#c3c8dc;")
        self.lbl_geom.setWordWrap(True)

        self.lbl_fit_quality = QLabel("")
        self.lbl_fit_quality.setStyleSheet(
            "color:#8e95ae; font-family:'Consolas','Menlo',monospace; "
            "font-size:9pt;")

        rc.addWidget(self.lbl_theta_big)
        rc.addWidget(self.lbl_theta_method)
        rc.addWidget(self.lbl_theta_sides)
        rc.addWidget(self.lbl_both_methods)
        rc.addWidget(self.lbl_qc_badge)
        rc.addWidget(self.lbl_geom)
        rc.addWidget(self.lbl_fit_quality)

        l_analyze.addWidget(self.results_card)
        box_analyze.setLayout(l_analyze)
        right_panel.addWidget(box_analyze)
        right_panel.addStretch(1)

        # wiring
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_upload.clicked.connect(self.upload_picture)
        self.btn_live.clicked.connect(self.start_camera)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_crop.clicked.connect(self.start_crop)
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_thresh.clicked.connect(self.open_threshold_dialog)

        self.btn_calib_auto.clicked.connect(self.start_auto_calib)
        self.btn_calib_manual.clicked.connect(self.start_manual_calib)
        self.combo_calib.currentTextChanged.connect(self.update_calib_buttons)

        self.btn_base1.clicked.connect(lambda: self.trigger_interaction('base1'))
        self.btn_base2.clicked.connect(lambda: self.trigger_interaction('base2'))
        self.btn_roi_draw.clicked.connect(lambda: self.trigger_interaction('roi'))
        self.btn_roi_clear.clicked.connect(self.clear_roi)
        self.btn_analyze.clicked.connect(self.run_analysis)

        self.radio_contour.toggled.connect(self._display_toggle_changed)
        self.radio_both.toggled.connect(self._display_toggle_changed)
        self.radio_circle.toggled.connect(self._display_toggle_changed)

    # =========================================================================
    # Overlay helpers
    # =========================================================================
    def _clear_list(self, items_list):
        for item in items_list:
            if item.scene() is self.image_label.scene:
                self.image_label.scene.removeItem(item)
        items_list.clear()

    def _clear_result_overlays(self):
        self._clear_list(self._result_overlays)

    def _clear_baseline_overlay(self):
        self._clear_list(self._baseline_overlay_items)

    def _clear_roi_overlay(self):
        self._clear_list(self._roi_overlay_items)

    def _paint_baseline(self):
        self._clear_baseline_overlay()
        if self.current_raw_image is None or self.baseline_data is None:
            return
        w = self.current_raw_image.shape[1]
        if self.baseline_data[0] == '1pt':
            y = float(self.baseline_data[1])
            y1 = y2 = y
            x1, x2 = 0.0, float(w)
        else:
            bx1, by1, bx2, by2 = self.baseline_data[1]
            if bx1 == bx2: bx2 += 1e-5
            m = (by2 - by1) / (bx2 - bx1)
            c = by1 - m * bx1
            x1, x2 = 0.0, float(w)
            y1, y2 = c, m * w + c
        ln = QGraphicsLineItem(x1, y1 + 0.5, x2, y2 + 0.5)
        ln.setPen(make_cosmetic_pen(_COL_BASELINE))
        self.image_label.scene.addItem(ln)
        self._baseline_overlay_items.append(ln)

    def _paint_roi(self):
        self._clear_roi_overlay()
        if self.roi_points is None:
            return
        pts = np.asarray(self.roi_points, dtype=float)
        path = QPainterPath()
        path.moveTo(pts[0, 0] + 0.5, pts[0, 1] + 0.5)
        for p in pts[1:]:
            path.lineTo(p[0] + 0.5, p[1] + 0.5)
        path.closeSubpath()
        item = QGraphicsPathItem(path)
        item.setPen(make_cosmetic_pen(_COL_ROI, dashed=True))
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.image_label.scene.addItem(item)
        self._roi_overlay_items.append(item)

    # ---- shared overlays (always drawn once results exist) -----------
    def _paint_common_overlays(self, r):
        """Detected contour cloud, contact markers, apex crosshair.
        Drawn regardless of which display mode is active."""
        scene = self.image_label.scene

        # detected contour points as small dots
        cp = r['contour_pts']
        path = QPainterPath()
        for x, y in cp:
            path.addEllipse(float(x), float(y), 0.5, 0.5)
        contour_item = QGraphicsPathItem(path)
        contour_item.setPen(make_cosmetic_pen(_COL_CONTOUR_PTS, width=1))
        scene.addItem(contour_item)
        self._result_overlays.append(contour_item)

        # contact point markers
        for pt in (r['contact_left_px'], r['contact_right_px']):
            px, py = float(pt[0]), float(pt[1])
            rad = 4
            dot = QGraphicsEllipseItem(px - rad + 0.5, py - rad + 0.5,
                                       2 * rad, 2 * rad)
            dot.setPen(make_cosmetic_pen(_COL_CONTACT, width=2))
            dot.setBrush(QBrush(QColor(_COL_CONTACT.red(),
                                       _COL_CONTACT.green(),
                                       _COL_CONTACT.blue(), 120)))
            scene.addItem(dot)
            self._result_overlays.append(dot)

        # apex crosshair
        apx, apy = float(r['apex_px'][0]), float(r['apex_px'][1])
        ch = 6
        for dx, dy in ((ch, 0), (0, ch)):
            ln = QGraphicsLineItem(apx - dx + 0.5, apy - dy + 0.5,
                                   apx + dx + 0.5, apy + dy + 0.5)
            ln.setPen(make_cosmetic_pen(_COL_APEX))
            scene.addItem(ln)
            self._result_overlays.append(ln)

    def _paint_contour_tangents(self, r):
        """Tangent segments from the contour polynomial fit (the actual
        contact-angle-making directions)."""
        if not r.get('contour_fit_valid'):
            return
        scene = self.image_label.scene
        R_ref = r['circle_radius_px']
        tangent_len = max(35.0, 0.55 * R_ref)
        for pt, tangent, color in (
                (r['contact_left_px'],  r['contour_tangent_left'],  _COL_TANG_C_L),
                (r['contact_right_px'], r['contour_tangent_right'], _COL_TANG_C_R)):
            if tangent is None or np.all(np.asarray(tangent) == 0):
                continue
            px, py = float(pt[0]), float(pt[1])
            tx, ty = float(tangent[0]), float(tangent[1])
            seg = QGraphicsLineItem(
                px + 0.5, py + 0.5,
                px + tx * tangent_len + 0.5,
                py + ty * tangent_len + 0.5)
            seg.setPen(make_cosmetic_pen(color, width=3))
            scene.addItem(seg)
            self._result_overlays.append(seg)

    def _paint_circle_overlays(self, r):
        """Fitted circle + its tangent directions at the contacts."""
        scene = self.image_label.scene
        cx, cy = r['circle_center']
        R = r['circle_radius_px']
        circle = QGraphicsEllipseItem(cx - R + 0.5, cy - R + 0.5,
                                      2 * R, 2 * R)
        circle.setPen(make_cosmetic_pen(_COL_CIRCLE, width=2, dashed=True))
        circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        scene.addItem(circle)
        self._result_overlays.append(circle)

        tangent_len = max(35.0, 0.55 * R)
        for pt, tangent, color in (
                (r['contact_left_px'],  r['circle_tangent_left'],  _COL_TANG_S_L),
                (r['contact_right_px'], r['circle_tangent_right'], _COL_TANG_S_R)):
            px, py = float(pt[0]), float(pt[1])
            tx, ty = float(tangent[0]), float(tangent[1])
            seg = QGraphicsLineItem(
                px + 0.5, py + 0.5,
                px + tx * tangent_len + 0.5,
                py + ty * tangent_len + 0.5)
            seg.setPen(make_cosmetic_pen(color, width=2, dashed=True))
            scene.addItem(seg)
            self._result_overlays.append(seg)

    def _render_overlays_for_mode(self):
        """Repaint overlays according to the current display_mode toggle."""
        self._clear_result_overlays()
        if self.last_results is None:
            return
        r = self.last_results
        self._paint_common_overlays(r)
        if self.display_mode in ('contour', 'both'):
            self._paint_contour_tangents(r)
        if self.display_mode in ('circle', 'both'):
            self._paint_circle_overlays(r)

    def _display_toggle_changed(self, _checked):
        if self.radio_contour.isChecked():
            self.display_mode = 'contour'
        elif self.radio_circle.isChecked():
            self.display_mode = 'circle'
        else:
            self.display_mode = 'both'
        # repaint and refresh the results card's "primary" number
        self._render_overlays_for_mode()
        if self.last_results is not None:
            self._populate_results_card(self.last_results)

    # =========================================================================
    # Settings / misc
    # =========================================================================
    def open_settings(self):
        dialog = SettingsDialog(
            self.needle_mm, self.density_inner, self.density_outer,
            self.ts_interval,
            self.needle_tol_rel, self.rho_in_tol_rel, self.rho_out_tol_rel,
            self)
        dialog.setStyleSheet(DARK_STYLESHEET)
        if dialog.exec():
            (self.needle_mm, self.density_inner, self.density_outer,
             self.ts_interval, self.needle_tol_rel,
             self.rho_in_tol_rel, self.rho_out_tol_rel) = dialog.get_values()

    def start_crop(self):
        self.image_label.set_mode('crop')
        self.btn_apply_crop.show(); self.btn_crop.hide()

    def apply_crop(self):
        rect = self.image_label.get_crop_rect()
        if rect:
            x, y, w, h = rect
            if w > 10 and h > 10 and self.current_raw_image is not None:
                H, W = self.current_raw_image.shape[:2]
                y1, y2 = max(0, y), min(H, y + h)
                x1, x2 = max(0, x), min(W, x + w)
                cropped = self.current_raw_image[y1:y2, x1:x2]
                self.setup_image(cropped)
        self.btn_apply_crop.hide(); self.btn_crop.show()
        self.image_label.set_mode('idle')

    def open_threshold_dialog(self):
        if self.current_raw_image is not None:
            dialog = ThresholdDialog(self.current_raw_image, self)
            dialog.setStyleSheet(DARK_STYLESHEET)
            if dialog.exec():
                self.setup_image(dialog.get_processed_image())

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
                self, "Pixel Scale", "Enter scale in mm/px:", 0.01, decimals=6)
            if ok and val > 0:
                self.pixel_to_m = val / 1000.0
                self._set_status(self.lbl_calib_status,
                                 f"Scale: {val:.5f} mm/px", True)
        else:
            self.image_label.set_image(self.current_raw_image, auto_fit=False)
            self.image_label.set_mode('calib')
            self.lbl_calib_status.setText("Draw line across reference object.")

    def handle_calib_pts(self, x1, y1, x2, y2):
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist < 5:
            self._set_status(self.lbl_calib_status,
                             "Line too short. Try again.", False)
            return
        val, ok = QInputDialog.getDouble(
            self, "Reference Size",
            f"Distance: {dist:.1f}px.\nEnter physical size (mm):",
            self.needle_mm, decimals=4)
        if ok and val > 0:
            self.pixel_to_m = (val / 1000.0) / dist
            self.needle_mm = val
            self._set_status(self.lbl_calib_status,
                             f"Manual: {dist:.1f} px ≡ {val} mm", True)
        self.image_label.set_image(self.current_raw_image, auto_fit=False)
        self._paint_baseline()
        self._paint_roi()

    def start_auto_calib(self):
        if self.current_raw_image is None:
            return
        self.image_label.set_image(self.current_raw_image, auto_fit=False)
        self.image_label.set_mode('calib_box')
        self.lbl_calib_status.setText("Draw a tight box around the object.")

    def handle_calib_box(self, x, y, w, h):
        if w < 5 or h < 5:
            self._set_status(self.lbl_calib_status,
                             "Box too small. Try again.", False)
            return
        mode = self.combo_calib.currentText()
        H, W = self.current_raw_image.shape[:2]
        y1, y2 = max(0, y), min(H, y + h)
        x1, x2 = max(0, x), min(W, x + w)
        roi = self.current_raw_image[y1:y2, x1:x2]
        pixel_dist, draw_info = self.auto_measure_roi(roi, mode)
        if pixel_dist is None or pixel_dist == 0:
            self._set_status(self.lbl_calib_status,
                             "Auto failed. Shape not found.", False)
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
            self, "Verify Auto Calibration",
            f"Auto detected: {pixel_dist:.1f} pixels.\n"
            "Verify the cyan marker on the image.\n"
            "Enter real physical size (mm):",
            self.needle_mm, decimals=4)
        if ok and val > 0:
            self.pixel_to_m = (val / 1000.0) / pixel_dist
            self.needle_mm = val
            self._set_status(self.lbl_calib_status,
                             f"Auto: {pixel_dist:.1f} px ≡ {val} mm", True)
        else:
            self.image_label.set_image(self.current_raw_image, auto_fit=False)
        self._paint_baseline()
        self._paint_roi()

    def auto_measure_roi(self, img_roi, mode):
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        corners_mean = (float(thresh[0, 0]) + thresh[-1, -1]
                        + thresh[0, -1] + thresh[-1, 0]) / 4
        if corners_mean > 127:
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

        elif mode == "Horizontal Rectangle":
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            heights, best_line = [], None
            mid_x = mask.shape[1] // 2
            for col in range(mask.shape[1]):
                rows = np.where(mask[:, col] > 0)[0]
                if len(rows):
                    heights.append(rows[-1] - rows[0])
                    if col == mid_x:
                        best_line = ((col, rows[0]), (col, rows[-1]))
            if not best_line and heights:
                c = np.where(mask > 0)[1][0]
                r = np.where(mask[:, c] > 0)[0]
                best_line = ((c, r[0]), (c, r[-1]))
            return ((np.median(heights), ('line', best_line))
                    if heights else (None, None))

        elif mode == "Sphere":
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            return (radius * 2, ('circle', (int(cx), int(cy)), int(radius)))

        return None, None

    # --- interaction triggers ------------------------------------------
    def trigger_interaction(self, mode):
        if self.current_raw_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        if mode == 'roi':
            if self.baseline_data is None:
                QMessageBox.warning(
                    self, "Warning",
                    "Please set the baseline before drawing the ROI.")
                return
            self.roi_points = None
            self._clear_roi_overlay()
            self.image_label.set_mode(mode, baseline_data=self.baseline_data)
        else:
            self.image_label.set_image(self.current_raw_image, auto_fit=False)
            self.image_label.set_mode(mode)
            if mode == 'base1':
                self.lbl_base_status.setText("Click once on the shoulder.")
            elif mode == 'base2':
                self.lbl_base_status.setText("Click two points to define line.")
            self.lbl_base_status.setStyleSheet("color:#c3c8dc;")
            self._paint_baseline(); self._paint_roi()

    def handle_baseline_1pt(self, y):
        self.baseline_data = ('1pt', y)
        self._set_status(self.lbl_base_status, f"Set: flat (y = {y})", True)
        self._paint_baseline()
        self._paint_roi()

    def handle_baseline_2pt(self, x1, y1, x2, y2):
        self.baseline_data = ('2pt', (x1, y1, x2, y2))
        self._set_status(self.lbl_base_status, "Set: slanted line", True)
        self._paint_baseline()
        self._paint_roi()

    def handle_roi_poly(self, pts_list):
        self.roi_points = np.array(pts_list, dtype=np.int32)
        self._set_status(self.lbl_roi_status, "ROI polygon saved.", True)
        self._paint_baseline()
        self._paint_roi()

    def clear_roi(self):
        self.roi_points = None
        self.image_label.set_mode('idle')
        self._clear_roi_overlay()
        self._set_status(self.lbl_roi_status, "ROI cleared.", False)

    # --- source loading -----------------------------------------------
    def upload_picture(self):
        self.stop_camera()
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fp:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is not None:
                self.setup_image(img)

    def start_camera(self):
        try:
            self.camera.start()
            self.btn_capture.show()
            self.btn_live.setText("Stop camera")
            self.btn_live.clicked.disconnect()
            self.btn_live.clicked.connect(self.stop_camera)
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Camera Error",
                                 f"Failed to connect:\n{str(e)}")

    def stop_camera(self):
        self.timer.stop()
        self.camera.stop()
        self.btn_capture.hide()
        self.btn_live.setText("Start live camera")
        try:
            self.btn_live.clicked.disconnect()
        except TypeError:
            pass
        self.btn_live.clicked.connect(self.start_camera)

    def update_frame(self):
        ret, frame = self.camera.get_frame()
        if ret and frame is not None:
            self.current_raw_image = frame
            self.image_label.set_image(frame, auto_fit=False)

    def capture_frame(self):
        if self.current_raw_image is not None:
            self.stop_camera()
            self.setup_image(self.current_raw_image.copy())

    def setup_image(self, img):
        self.current_raw_image = img
        self.image_label.set_image(img, auto_fit=True)
        self.baseline_data = None
        self.roi_points = None
        self.last_results = None
        self._clear_result_overlays()
        self._clear_baseline_overlay()
        self._clear_roi_overlay()

        self.btn_crop.setEnabled(True)
        self.btn_thresh.setEnabled(True)
        self.btn_apply_crop.hide()
        self.btn_crop.show()

        self._set_status(self.lbl_base_status, "Baseline not set.", False)
        self._set_status(self.lbl_roi_status,  "No ROI set.",       False)
        self._clear_results_card()

    # =========================================================================
    # Analysis
    # =========================================================================
    def run_analysis(self):
        if self.current_raw_image is None or self.baseline_data is None:
            QMessageBox.warning(self, "Warning",
                                "Please set the baseline first.")
            return
        self.lbl_theta_big.setText("…")
        self.lbl_theta_method.setText("Calculating…")

        try:
            results = analyze_sessile_drop(
                image_input=self.current_raw_image,
                baseline_data=self.baseline_data,
                roi_poly=self.roi_points,
                px_to_m=self.pixel_to_m,
                rho=self.density_inner)
            self.last_results = results

            self.image_label.set_image(self.current_raw_image, auto_fit=False)
            self._paint_baseline()
            self._paint_roi()
            self._render_overlays_for_mode()
            self._populate_results_card(results)
        except Exception as e:
            self.lbl_theta_big.setText("—")
            self.lbl_theta_method.setText("Analysis failed.")
            self.lbl_theta_method.setStyleSheet("color:#ff6a4d;")
            QMessageBox.critical(self, "Analysis Error", str(e))

    # =========================================================================
    # Results card
    # =========================================================================
    def _populate_results_card(self, r):
        # Primary = whichever method the toggle says
        contour_valid = r.get('contour_fit_valid', False)
        if self.display_mode == 'circle' or not contour_valid:
            primary_name = "Fitted circle"
            pL = r['circle_left_angle_deg']
            pR = r['circle_right_angle_deg']
            pAvg = r['circle_avg_angle_deg']
            pAsym = r['circle_angle_asymmetry_deg']
            if self.display_mode != 'circle' and not contour_valid:
                primary_name += "  (contour fit failed)"
        else:
            primary_name = "Contour tangent"
            pL = r['contour_left_angle_deg']
            pR = r['contour_right_angle_deg']
            pAvg = r['contour_avg_angle_deg']
            pAsym = r['contour_angle_asymmetry_deg']

        self.lbl_theta_big.setText(f"θ̄  =  {pAvg:0.2f}°")
        self.lbl_theta_method.setText(primary_name)
        self.lbl_theta_method.setStyleSheet("color:#8ab4ff; font-weight:600;")
        self.lbl_theta_sides.setText(
            f"left  {pL:6.2f}°     right  {pR:6.2f}°     |Δ| = {pAsym:.2f}°")

        # Side-by-side comparison block (both methods, always)
        cL  = r['contour_left_angle_deg']
        cR  = r['contour_right_angle_deg']
        cAv = r['contour_avg_angle_deg']
        sL  = r['circle_left_angle_deg']
        sR  = r['circle_right_angle_deg']
        sAv = r['circle_avg_angle_deg']
        rows = [
            "              left      right     avg",
            f"Contour     {self._fmt_deg(cL)}  {self._fmt_deg(cR)}  {self._fmt_deg(cAv)}",
            f"Circle      {self._fmt_deg(sL)}  {self._fmt_deg(sR)}  {self._fmt_deg(sAv)}",
        ]
        if np.isfinite(cAv) and np.isfinite(sAv):
            rows.append(f"Δ(C−S) avg   {cAv - sAv:+7.2f}°")
        self.lbl_both_methods.setText("\n".join(rows))

        # QC: small asymmetry + small RMS residual (on the ACTIVE method)
        active_asym = pAsym if np.isfinite(pAsym) else float('inf')
        rms = r['rms_residual_px']
        asym_ok = active_asym <= 3.0
        rms_ok  = rms <= 2.0
        if asym_ok and rms_ok:
            self.lbl_qc_badge.setText("✓  QC  PASS")
            self.lbl_qc_badge.setStyleSheet(
                "color:#4fd07b; font-weight:700;")
        else:
            flags = []
            if not asym_ok and np.isfinite(active_asym):
                flags.append(f"|Δ|={active_asym:.1f}° > 3°")
            if not rms_ok:
                flags.append(f"RMS={rms:.2f} px > 2")
            if not flags:
                flags.append("contour fit uncertain")
            self.lbl_qc_badge.setText("⚠  QC  ISSUES — " + ", ".join(flags))
            self.lbl_qc_badge.setStyleSheet(
                "color:#ff9f5a; font-weight:700;")

        Bo = r.get('bond_number')
        V  = r.get('volume_uL')
        R_m = r.get('radius_m')
        geom = f"Circle radius  : {r['circle_radius_px']:.2f} px"
        if R_m is not None:
            geom += f"   ({R_m*1000:.3f} mm)"
        geom += (f"\nCap height     : {r['cap_height_px']:.2f} px")
        geom += (f"\nFootprint span : {r['footprint_px']:.2f} px")
        if V is not None:
            geom += f"\nVolume (cap)   : {V:.3f} µL"
        if Bo is not None:
            geom += (f"\nBond number    : {Bo:.4f}"
                     f"   [{r.get('suitability','')}]")
        self.lbl_geom.setText(geom)

        fit_txt = (f"Circle fit RMS  : {rms:.3f} px\n"
                   f"Contour pts used (L/R): "
                   f"{r.get('contour_npts_left', 0)} / "
                   f"{r.get('contour_npts_right', 0)}")
        self.lbl_fit_quality.setText(fit_txt)

    @staticmethod
    def _fmt_deg(val):
        if val is None or not np.isfinite(val):
            return "  —   "
        return f"{val:7.2f}°"

    def _clear_results_card(self):
        self.lbl_theta_big.setText("—")
        self.lbl_theta_method.setText("")
        self.lbl_theta_sides.setText("")
        self.lbl_both_methods.setText("")
        self.lbl_qc_badge.setText("")
        self.lbl_geom.setText("")
        self.lbl_fit_quality.setText("")

    def _set_status(self, label, text, success):
        label.setText(text)
        label.setStyleSheet(
            "color:#4fd07b; font-weight:600;" if success
            else "color:#c3c8dc;")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
