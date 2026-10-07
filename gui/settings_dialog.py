"""
gui.settings_dialog
===================

Physical parameters + measurement tolerances for uncertainty propagation,
organised as two kinds of reusable presets (core.presets):

    Needle        – OD, ID, OD tolerance          (changes with hardware)
    Fluid system  – densities, tolerances, T, P   (changes per experiment)

Shared between pendant and sessile windows.  ``SettingsStateMixin`` gives
both windows the same startup-restore / apply / remember behaviour.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QDoubleValidator
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QFormLayout, QFrame, QGroupBox,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                             QMessageBox, QPlainTextEdit, QPushButton,
                             QScrollArea, QToolButton, QVBoxLayout, QWidget)

from core import presets as P

CUSTOM = "Custom (not saved as a preset)"
BUILTIN_SUFFIX = "  (built-in)"

# Dialog field → (preset key, scale from field to stored value)
_NEEDLE_FIELDS = [('od_mm', "Needle OD (mm):"),
                  ('id_mm', "Needle ID (mm):"),
                  ('od_tol_pct', "OD tolerance (%):")]
_FLUID_FIELDS = [('rho_in', "Inner fluid density (kg/m³):"),
                 ('rho_out', "Outer fluid density (kg/m³):"),
                 ('rho_in_tol_pct', "Inner density tolerance (%):"),
                 ('rho_out_tol_pct', "Outer density tolerance (%):"),
                 ('T_C', "Temperature (°C):"),
                 ('P_MPa', "Pressure (MPa):")]
_OPTIONAL = {'id_mm', 'T_C', 'P_MPa'}
_PLACEHOLDER = {'id_mm': "auto (70% of OD)", 'T_C': "optional",
                'P_MPa': "optional"}


def _fmt(v):
    return "" if v is None else f"{v:g}"


class SettingsDialog(QDialog):
    # Emitted by Apply (and by OK) with the same dict as get_result().
    applied = pyqtSignal(dict)

    def __init__(self, needle_name, needle_values, fluid_name, fluid_values,
                 ts_interval, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Physical Parameters & Tolerances")
        self._loading = False
        self._lists = {}                      # kind → [Preset]
        self._fields = {'needle': {}, 'fluid': {}}
        self._notes = {}
        self._combo, self._modlbl, self._notelbl = {}, {}, {}
        self._btn_save, self._btn_del = {}, {}
        self._applied = None                  # last result sent to the window

        # Fields scroll; the button bar stays pinned at the bottom, so it is
        # reachable even on short (e.g. 1280×800 at 150 %) screens.
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        fv = QDoubleValidator()

        self._problems = QLabel()
        self._problems.setWordWrap(True)
        self._problems.setStyleSheet("color:#ff9f5a;")
        self._problems.hide()

        for kind, title, specs in (('needle', "Needle", _NEEDLE_FIELDS),
                                   ('fluid', "Fluid system", _FLUID_FIELDS)):
            grp = QGroupBox(title)
            lay = QVBoxLayout(grp)

            row = QHBoxLayout()
            combo = QComboBox()
            combo.setMinimumWidth(260)
            combo.currentIndexChanged.connect(
                lambda _i, k=kind: self._on_select(k))
            row.addWidget(combo, 1)
            b_save = QPushButton("Save")
            b_save.setToolTip("Save the fields below into this preset")
            b_save.clicked.connect(lambda _c, k=kind: self._save(k))
            b_saveas = QPushButton("Save as…")
            b_saveas.clicked.connect(lambda _c, k=kind: self._save_as(k))
            b_del = QPushButton("Delete")
            b_del.clicked.connect(lambda _c, k=kind: self._delete(k))
            for b in (b_save, b_saveas, b_del):
                b.setAutoDefault(False)          # Enter in a field means OK
                row.addWidget(b)
            lay.addLayout(row)

            mod = QLabel()
            mod.setStyleSheet("color:#ffb347;")
            lay.addWidget(mod)

            form = QFormLayout()
            for key, label in specs:
                le = QLineEdit()
                le.setValidator(fv)
                if key in _PLACEHOLDER:
                    le.setPlaceholderText(_PLACEHOLDER[key])
                le.textChanged.connect(lambda _t, k=kind: self._on_edit(k))
                form.addRow(label, le)
                self._fields[kind][key] = le
            notes = QPlainTextEdit()
            notes.setFixedHeight(54)
            notes.setPlaceholderText("Notes (source of values, conditions…)")
            notes.textChanged.connect(lambda k=kind: self._on_edit(k))
            form.addRow("Notes:", notes)
            self._notes[kind] = notes
            lay.addLayout(form)

            if kind == 'fluid':
                self._lbl_drho = QLabel()
                self._lbl_drho.setStyleSheet("color:#8ab4ff; font-weight:600;")
                lay.addWidget(self._lbl_drho)
                hint = QLabel("<small><i>Typical tolerances: syringe-grade needle "
                              "0.5–1 %, pure water at known T 0.05 %, dry air "
                              "1–3 %, dense gas near saturation ≥ 2 %.</i></small>")
                hint.setWordWrap(True)
                lay.addWidget(hint)

            self._combo[kind], self._modlbl[kind] = combo, mod
            self._btn_save[kind], self._btn_del[kind] = b_save, b_del
            root.addWidget(grp)

        root.addWidget(self._problems)

        grp_ts = QGroupBox("Time series")
        form_ts = QFormLayout(grp_ts)
        self.ts_input = QLineEdit(_fmt(ts_interval))
        self.ts_input.setValidator(fv)
        self.ts_input.textChanged.connect(self._refresh_apply)
        form_ts.addRow("Sample interval (s):", self.ts_input)
        root.addWidget(grp_ts)
        root.addStretch(1)

        # Bottom bar: preset file actions on the left, dialog buttons
        # (OK / Cancel / Apply) in the bottom-right corner.
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#3a3f55;")
        outer.addWidget(line)
        btn_row = QHBoxLayout()
        b_imp = QPushButton("Import preset…")
        b_imp.setAutoDefault(False)
        b_imp.clicked.connect(self._import)
        b_exp = QToolButton()
        b_exp.setText("Export preset")
        b_exp.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(b_exp)
        for kind, label in (('needle', "Export needle…"), ('fluid', "Export fluid system…")):
            act = QAction(label, menu)
            act.triggered.connect(lambda _c, k=kind: self._export(k))
            menu.addAction(act)
        b_exp.setMenu(menu)
        btn_row.addWidget(b_imp)
        btn_row.addWidget(b_exp)
        btn_row.addStretch(1)
        self._lbl_applied = QLabel()
        self._lbl_applied.setStyleSheet("color:#7fd88f;")
        btn_row.addWidget(self._lbl_applied)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Apply |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        self._btn_apply = btns.button(QDialogButtonBox.StandardButton.Apply)
        self._btn_apply.setToolTip("Use these values now and keep the dialog open")
        self._btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(btns)
        outer.addLayout(btn_row)

        for kind, name, values in (('needle', needle_name, needle_values),
                                   ('fluid', fluid_name, fluid_values)):
            self._reload(kind, select=name)
            self._set_fields(kind, values)
            self._on_edit(kind)
        self._result = None
        self._applied = self._collect(quiet=True)   # what the window already uses
        self._refresh_apply()
        self._fit_to_screen(scroll, body, outer, line, btn_row)

    def _fit_to_screen(self, scroll, body, outer, line, btn_row):
        """Full height when it fits, else as tall as the screen allows."""
        m = outer.contentsMargins()
        want_w = (body.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                  + m.left() + m.right())
        want_h = (body.sizeHint().height() + line.sizeHint().height()
                  + btn_row.sizeHint().height() + 2 * outer.spacing()
                  + m.top() + m.bottom())
        scr = (self.parentWidget() or self).screen() or QApplication.primaryScreen()
        # Leave room for the title bar and a little breathing space.
        self.resize(max(560, want_w), min(want_h, scr.availableGeometry().height() - 60))
    # ------------------------------------------------------------------ data
    def _reload(self, kind, select=None):
        """Re-read presets of ``kind`` and repopulate its combo."""
        items, problems = P.list_presets(kind)
        self._lists[kind] = items
        if problems:
            self._problems.setText("Some preset files were skipped:\n• " +
                                   "\n• ".join(problems))
            self._problems.show()
        combo = self._combo[kind]
        self._loading = True
        combo.clear()
        combo.addItem(CUSTOM)
        idx = 0
        for i, p in enumerate(items, start=1):
            combo.addItem(p.name + (BUILTIN_SUFFIX if p.builtin else ""))
            if select and p.name.casefold() == select.casefold():
                idx = i
        combo.setCurrentIndex(idx)
        self._loading = False

    def _selected(self, kind):
        i = self._combo[kind].currentIndex()
        return self._lists[kind][i - 1] if i >= 1 else None

    def _set_fields(self, kind, values):
        self._loading = True
        for key, le in self._fields[kind].items():
            le.setText(_fmt(values.get(key)))
        self._notes[kind].setPlainText(values.get('notes') or '')
        self._loading = False

    def _read_fields(self, kind):
        """Field values → validated dict.  Raises PresetError."""
        raw = {}
        for key, le in self._fields[kind].items():
            t = le.text().strip().replace(',', '.')
            if not t:
                raw[key] = None
                continue
            try:
                raw[key] = float(t)
            except ValueError:
                raise P.PresetError(f"'{t}' is not a number.")
        raw['notes'] = self._notes[kind].toPlainText().strip()
        return P.validate_values(kind, raw)

    # ------------------------------------------------------------ reactions
    def _on_select(self, kind):
        if self._loading:
            return
        p = self._selected(kind)
        if p is not None:
            self._set_fields(kind, p.values)
        self._on_edit(kind)

    def _on_edit(self, kind):
        if self._loading:
            return
        p = self._selected(kind)
        try:
            vals = self._read_fields(kind)
            err = None
        except P.PresetError as e:
            vals, err = None, str(e)
        if err:
            self._modlbl[kind].setText(f"Invalid: {err}")
            self._modlbl[kind].setStyleSheet("color:#ff6a4d;")
        elif p is None:
            self._modlbl[kind].setText("Custom values — use “Save as…” to keep them "
                                       "as a preset.")
            self._modlbl[kind].setStyleSheet("color:#aab4d0;")
        elif not P.values_equal(kind, vals, p.values):
            self._modlbl[kind].setText("Modified — not yet saved to this preset.")
            self._modlbl[kind].setStyleSheet("color:#ffb347;")
        else:
            self._modlbl[kind].setText("Built-in preset (read-only)" if p.builtin
                                       else "")
            self._modlbl[kind].setStyleSheet("color:#aab4d0;")
        self._btn_save[kind].setEnabled(p is not None and not p.builtin)
        self._btn_del[kind].setEnabled(p is not None and not p.builtin)
        if kind == 'fluid':
            if vals:
                drho = abs(vals['rho_in'] - vals['rho_out'])
                self._lbl_drho.setText(f"Δρ = {drho:.2f} kg/m³")
            else:
                self._lbl_drho.setText("Δρ = —")
        self._refresh_apply()

    def _refresh_apply(self, *_):
        """Enable Apply only when the fields differ from what the window uses."""
        if self._loading or not hasattr(self, '_btn_apply'):
            return
        cur = self._collect(quiet=True)
        same = (cur is not None and self._applied is not None
                and self._same_result(cur, self._applied))
        self._btn_apply.setEnabled(not same)
        if not same:
            self._lbl_applied.clear()

    @staticmethod
    def _same_result(a, b):
        return (a['needle_name'] == b['needle_name']
                and a['fluid_name'] == b['fluid_name']
                and P.values_equal('needle', a['needle'], b['needle'])
                and P.values_equal('fluid', a['fluid'], b['fluid'])
                and abs(a['ts_interval'] - b['ts_interval']) <= 1e-12)

    # --------------------------------------------------------------- actions
    def _warn(self, title, msg):
        QMessageBox.warning(self, title, msg)

    def _save(self, kind):
        p = self._selected(kind)
        if p is None or p.builtin:
            return self._save_as(kind)
        try:
            vals = self._read_fields(kind)
            P.save_user_preset(kind, p.name, vals, overwrite=True)
        except (P.PresetError, OSError) as e:
            return self._warn("Cannot save preset", str(e))
        self._reload(kind, select=p.name)
        self._on_edit(kind)

    def _save_as(self, kind):
        try:
            vals = self._read_fields(kind)
        except P.PresetError as e:
            return self._warn("Cannot save preset", str(e))
        p = self._selected(kind)
        suggestion = "" if p is None else (p.name + " (copy)" if p.builtin else p.name)
        name, ok = QInputDialog.getText(
            self, "Save preset as",
            f"Name for this {'needle' if kind == 'needle' else 'fluid system'} preset:",
            text=suggestion)
        if not ok:
            return
        try:
            P.save_user_preset(kind, name, vals)
        except P.PresetError as e:
            existing = P.find_preset(kind, name.strip())
            if existing is not None and not existing.builtin:
                if QMessageBox.question(
                        self, "Replace preset?",
                        f"A preset named '{existing.name}' already exists. "
                        f"Replace it?") != QMessageBox.StandardButton.Yes:
                    return
                try:
                    P.save_user_preset(kind, name, vals, overwrite=True)
                except (P.PresetError, OSError) as e2:
                    return self._warn("Cannot save preset", str(e2))
            else:
                return self._warn("Cannot save preset", str(e))
        except OSError as e:
            return self._warn("Cannot save preset", str(e))
        self._reload(kind, select=name.strip())
        self._on_edit(kind)

    def _delete(self, kind):
        p = self._selected(kind)
        if p is None or p.builtin:
            return
        if QMessageBox.question(
                self, "Delete preset?",
                f"Delete the preset '{p.name}'? The values in the fields are "
                f"kept.") != QMessageBox.StandardButton.Yes:
            return
        try:
            P.delete_user_preset(kind, p.name)
        except (P.PresetError, OSError) as e:
            return self._warn("Cannot delete preset", str(e))
        self._reload(kind)
        self._on_edit(kind)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "",
                                              "Preset files (*.json)")
        if not path:
            return
        try:
            p = P.import_preset(path)
        except P.PresetError as e:
            if "already exists" not in str(e):
                return self._warn("Cannot import preset", str(e))
            if QMessageBox.question(self, "Replace preset?",
                                    f"{e}\nReplace it with the imported one?") \
                    != QMessageBox.StandardButton.Yes:
                return
            try:
                p = P.import_preset(path, overwrite=True)
            except (P.PresetError, OSError) as e2:
                return self._warn("Cannot import preset", str(e2))
        except OSError as e:
            return self._warn("Cannot import preset", str(e))
        self._reload(p.kind, select=p.name)
        self._set_fields(p.kind, p.values)
        self._on_edit(p.kind)

    def _export(self, kind):
        try:
            vals = self._read_fields(kind)
        except P.PresetError as e:
            return self._warn("Cannot export preset", str(e))
        p = self._selected(kind)
        name = p.name if p is not None else ""
        if not name:
            name, ok = QInputDialog.getText(self, "Export preset",
                                            "Name to store in the file:")
            if not ok:
                return
        try:
            name = P.validate_name(name)
        except P.PresetError as e:
            return self._warn("Cannot export preset", str(e))
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export preset", f"{P._slug(name)}.json", "Preset files (*.json)")
        if not dest:
            return
        try:
            P.export_preset(P.Preset(kind, name, vals, False, None), dest)
        except OSError as e:
            self._warn("Cannot export preset", str(e))

    def _collect(self, quiet=False):
        """Validated result dict, or None (with a warning unless ``quiet``)."""
        try:
            needle = self._read_fields('needle')
            fluid = self._read_fields('fluid')
        except P.PresetError as e:
            return None if quiet else self._warn("Invalid value", str(e))
        try:
            ts = float(self.ts_input.text().replace(',', '.'))
            if not ts > 0:
                raise ValueError
        except ValueError:
            return None if quiet else self._warn(
                "Invalid value", "Sample interval must be a positive number.")
        pn, pf = self._selected('needle'), self._selected('fluid')
        return dict(needle_name=pn.name if pn else None, needle=needle,
                    fluid_name=pf.name if pf else None, fluid=fluid,
                    ts_interval=ts)

    def _on_apply(self):
        res = self._collect()
        if res is None:
            return False
        self._result = self._applied = res
        self.applied.emit(res)
        self._refresh_apply()
        self._lbl_applied.setText("Applied.")
        return True

    def _on_ok(self):
        if self._on_apply():
            self.accept()

    def _ask_apply_on_close(self):
        """'apply', 'discard' or 'stay' for closing with unapplied changes."""
        box = QMessageBox(QMessageBox.Icon.Question, "Apply changes?",
                          "The selected presets / values have not been applied "
                          "to the window yet.\n\nSaving a preset only stores it; "
                          "Apply or OK makes the window use it.", parent=self)
        b_apply = box.addButton("Apply and close", QMessageBox.ButtonRole.AcceptRole)
        b_disc = box.addButton("Discard changes", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Keep editing", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_apply)
        box.exec()
        clicked = box.clickedButton()
        return 'apply' if clicked is b_apply else 'discard' if clicked is b_disc else 'stay'

    def reject(self):
        # Cancel, Esc and the window's close button all land here.
        if self._btn_apply.isEnabled():
            choice = self._ask_apply_on_close()
            if choice == 'stay':
                return
            if choice == 'apply':
                return self._on_ok()
        super().reject()

    def get_result(self):
        """Last applied settings: dict(needle_name, needle, fluid_name, fluid,
        ts_interval), or None if nothing was applied.  Names are None for
        custom (unsaved) values."""
        return self._result


class SettingsStateMixin:
    """Startup restore / apply / remember for windows with physical settings.

    The window keeps its historical attributes (needle_mm, density_inner, …)
    so analysis code is unchanged; this mixin maps presets onto them.
    """

    def init_settings_state(self):
        self._apply_settings(P.load_state())

    def _apply_settings(self, st):
        n, f = st['needle'], st['fluid']
        self.needle_preset_name = st.get('needle_name')
        self.fluid_preset_name = st.get('fluid_name')
        self.needle_mm = n['od_mm']
        self.needle_id_mm = n.get('id_mm')
        self.needle_tol_rel = n['od_tol_pct'] / 100.0
        self.needle_notes = n.get('notes', '')
        self.density_inner = f['rho_in']
        self.density_outer = f['rho_out']
        self.rho_in_tol_rel = f['rho_in_tol_pct'] / 100.0
        self.rho_out_tol_rel = f['rho_out_tol_pct'] / 100.0
        self.fluid_T_C = f.get('T_C')
        self.fluid_P_MPa = f.get('P_MPa')
        self.fluid_notes = f.get('notes', '')
        self.ts_interval = st['ts_interval']
        self._update_settings_tooltip()

    def _current_settings(self):
        return dict(
            needle_name=self.needle_preset_name,
            needle=dict(od_mm=self.needle_mm, id_mm=self.needle_id_mm,
                        od_tol_pct=self.needle_tol_rel * 100.0,
                        notes=getattr(self, 'needle_notes', '')),
            fluid_name=self.fluid_preset_name,
            fluid=dict(rho_in=self.density_inner, rho_out=self.density_outer,
                       rho_in_tol_pct=self.rho_in_tol_rel * 100.0,
                       rho_out_tol_pct=self.rho_out_tol_rel * 100.0,
                       T_C=getattr(self, 'fluid_T_C', None),
                       P_MPa=getattr(self, 'fluid_P_MPa', None),
                       notes=getattr(self, 'fluid_notes', '')),
            ts_interval=self.ts_interval)

    def settings_summary(self):
        """One line naming the active presets, for tooltips / result cards."""
        n = self.needle_preset_name or "custom needle"
        f = self.fluid_preset_name or "custom fluids"
        return f"{n}  ·  {f}"

    def _update_settings_tooltip(self):
        btn = getattr(self, 'btn_settings', None)
        if btn is not None:
            drho = abs(self.density_inner - self.density_outer)
            btn.setToolTip("Physical parameters and tolerances\n"
                           f"Needle: {self.needle_preset_name or 'custom'}"
                           f"  —  OD {self.needle_mm:g} mm\n"
                           f"Fluids: {self.fluid_preset_name or 'custom'}"
                           f"  —  Δρ = {drho:.1f} kg/m³")

    def on_settings_applied(self):
        """Hook for windows to refresh anything derived from the settings."""

    def open_settings_dialog(self, stylesheet=None):
        """Show the dialog.  Apply and OK both apply + remember immediately;
        Cancel after Apply keeps what was applied.  Returns True if anything
        was applied."""
        cur = self._current_settings()
        dlg = SettingsDialog(cur['needle_name'], cur['needle'],
                             cur['fluid_name'], cur['fluid'],
                             cur['ts_interval'], parent=self)
        if stylesheet:
            dlg.setStyleSheet(stylesheet)
        dlg.applied.connect(self._apply_and_remember)
        dlg.exec()
        return dlg.get_result() is not None

    def _apply_and_remember(self, st):
        self._apply_settings(st)
        self.on_settings_applied()
        try:
            P.save_state(st['needle_name'], st['needle'], st['fluid_name'],
                         st['fluid'], st['ts_interval'])
        except (P.PresetError, OSError) as e:
            QMessageBox.warning(self, "Settings not remembered",
                                f"The settings are applied but could not be saved "
                                f"for next time:\n{e}")
        return True
