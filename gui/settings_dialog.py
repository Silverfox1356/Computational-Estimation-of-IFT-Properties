"""
gui.settings_dialog
===================

Physical parameters + measurement tolerances for uncertainty propagation.
Shared between pendant and sessile windows.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QDialogButtonBox, QGroupBox, QLabel)
from PyQt6.QtGui import QDoubleValidator


class SettingsDialog(QDialog):
    def __init__(self, needle_mm, rho_in, rho_out, ts_interval,
                 needle_tol_rel=0.01,
                 rho_in_tol_rel=0.0005,
                 rho_out_tol_rel=0.02,
                 needle_id_mm=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Physical Parameters & Tolerances")
        self.resize(460, 420)

        root = QVBoxLayout(self)
        fv = QDoubleValidator()

        grp_phys = QGroupBox("Physical parameters")
        form_phys = QFormLayout()
        self.needle_input  = QLineEdit(str(needle_mm));  self.needle_input.setValidator(fv)
        id_val = str(needle_id_mm) if needle_id_mm is not None else ""
        self.needle_id_input = QLineEdit(id_val);        self.needle_id_input.setValidator(fv)
        self.needle_id_input.setPlaceholderText("auto (70% of OD)")
        self.rho_in_input  = QLineEdit(str(rho_in));     self.rho_in_input.setValidator(fv)
        self.rho_out_input = QLineEdit(str(rho_out));    self.rho_out_input.setValidator(fv)
        form_phys.addRow("Needle OD (mm):", self.needle_input)
        form_phys.addRow("Needle ID (mm):", self.needle_id_input)
        form_phys.addRow("Inner fluid density (kg/m³):", self.rho_in_input)
        form_phys.addRow("Outer fluid density (kg/m³):", self.rho_out_input)
        grp_phys.setLayout(form_phys)
        root.addWidget(grp_phys)

        grp_tol = QGroupBox("Tolerances for uncertainty propagation  (relative, 1-σ)")
        form_tol = QFormLayout()
        self.needle_tol_input  = QLineEdit(f"{needle_tol_rel*100:.3f}"); self.needle_tol_input.setValidator(fv)
        self.rho_in_tol_input  = QLineEdit(f"{rho_in_tol_rel*100:.3f}"); self.rho_in_tol_input.setValidator(fv)
        self.rho_out_tol_input = QLineEdit(f"{rho_out_tol_rel*100:.3f}"); self.rho_out_tol_input.setValidator(fv)
        form_tol.addRow("Needle OD tolerance (%):", self.needle_tol_input)
        form_tol.addRow("Inner density tolerance (%):", self.rho_in_tol_input)
        form_tol.addRow("Outer density tolerance (%):", self.rho_out_tol_input)
        hint = QLabel(
            "<small><i>Typical values: syringe-grade needle 0.5–1 %, "
            "pure water at known T 0.05 %, "
            "dry air 1–3 %.</i></small>")
        hint.setWordWrap(True)
        form_tol.addRow(hint)
        grp_tol.setLayout(form_tol)
        root.addWidget(grp_tol)

        grp_ts = QGroupBox("Time series")
        form_ts = QFormLayout()
        self.ts_input = QLineEdit(str(ts_interval)); self.ts_input.setValidator(fv)
        form_ts.addRow("Sample interval (s):", self.ts_input)
        grp_ts.setLayout(form_ts)
        root.addWidget(grp_ts)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def get_values(self):
        id_text = self.needle_id_input.text().strip()
        needle_id = float(id_text) if id_text else None
        return (
            float(self.needle_input.text()),
            float(self.rho_in_input.text()),
            float(self.rho_out_input.text()),
            float(self.ts_input.text()),
            float(self.needle_tol_input.text()) / 100.0,
            float(self.rho_in_tol_input.text()) / 100.0,
            float(self.rho_out_tol_input.text()) / 100.0,
            needle_id,
        )
