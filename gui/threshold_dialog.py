"""
gui.threshold_dialog
====================

Live threshold-tuning preview.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QPushButton, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class ThresholdDialog(QDialog):
    def __init__(self, cv_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tune Image Threshold")
        self.resize(1040, 580)

        self.original_img = cv_img.copy()
        self.current_threshold = 40
        self.processed_bgr = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(14)

        def _build_pane(title):
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            v = QVBoxLayout(frame)
            v.setContentsMargins(8, 8, 8, 8)
            hdr = QLabel(title)
            hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr.setStyleSheet("font-weight: bold; color: #9ec4ff;")
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(hdr)
            v.addWidget(lbl, 1)
            return frame, lbl

        left_frame,  self.orig_label    = _build_pane("Original image")
        right_frame, self.thresh_label  = _build_pane("Binarized preview")
        self.set_pixmap(self.orig_label, self.original_img)
        images_layout.addWidget(left_frame, 1)
        images_layout.addWidget(right_frame, 1)
        layout.addLayout(images_layout, 1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("0"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(self.current_threshold)
        self.slider.valueChanged.connect(self.update_threshold)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(QLabel("255"))
        self.val_label = QLabel(f"Threshold: {self.current_threshold}")
        self.val_label.setMinimumWidth(120)
        self.val_label.setStyleSheet("font-weight: bold; color: #9ec4ff;")
        slider_row.addWidget(self.val_label)
        layout.addLayout(slider_row)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply  = QPushButton("Apply to analysis")
        self.btn_apply.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #3a7ae8,stop:1 #2355c2);color:white;font-weight:600;"
            "border:1px solid #2c5fcb;border-bottom:2px solid #12275c;"
            "border-radius:4px;padding:5px 12px;}"
            "QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #17368a,stop:1 #254db0);border-top:2px solid #12275c;"
            "border-bottom:1px solid #2c5fcb;padding-top:6px;padding-bottom:4px;}"
        )
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_apply)
        layout.addLayout(btn_row)

        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.update_threshold(self.current_threshold)

    def update_threshold(self, val):
        self.current_threshold = val
        self.val_label.setText(f"Threshold: {val}")
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blurred, val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                          kernel, iterations=3)
        self.processed_bgr = cv2.cvtColor(binary_cleaned, cv2.COLOR_GRAY2BGR)
        self.set_pixmap(self.thresh_label, self.processed_bgr)

    def set_pixmap(self, label, cv_img):
        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
        else:
            h, w = cv_img.shape
            ch = 1
            rgb_image = cv_img
            fmt = QImage.Format.Format_Grayscale8
        qt_img = QImage(rgb_image.data, w, h, ch * w, fmt)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            470, 420,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pixmap)

    def get_processed_image(self):
        return self.processed_bgr
