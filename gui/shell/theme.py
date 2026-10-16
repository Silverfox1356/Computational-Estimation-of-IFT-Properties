"""
gui.shell.theme
===============

Light theme for Drop Analysis Studio, in the style of engineering
simulation software (light panels, blue accents).  Applied to the studio
window only, so the older dark windows keep their own look.

:class:`StudioLook` covers what a window stylesheet alone does not: the
Studio's dialogs and popups get the light palette (with Windows in dark mode
they would otherwise be dark behind the stylesheet's dark text), its spin
boxes and dropdowns sit on Qt's own Fusion style rather than the Windows 11
one, and dropdown lists use a delegate that follows the hover rule.  The
spin-box arrow buttons are drawn by the stylesheet, so they click where
they are drawn.
"""
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QStyle, QStyleFactory,
                             QStyledItemDelegate, QWidget)

ACCENT = '#2b6cb0'
TEXT = '#1f2328'
MUTED = '#5c6470'
LEVEL_COLOURS = dict(info='#1f2328', ok='#2f7a44', warn='#a35f00', error='#c0392b')
PLOT_PASS = '#2f9e44'
PLOT_FAIL = '#d64545'
PLOT_LINE = '#2b6cb0'

STYLESHEET = """
QMainWindow, QWidget#StudioRoot { background: #e4e6e9; }
QWidget { color: #1f2328; font-family: 'Segoe UI', Arial, sans-serif; font-size: 9pt; }

/* ribbon */
QTabWidget#Ribbon { background: #f7f7f7; }
QTabWidget#Ribbon > QTabBar { background: #f7f7f7; }
QTabWidget#Ribbon::pane { border: none; border-top: 1px solid #d0d0d0; background: #ffffff; }
QTabWidget#Ribbon > QTabBar::tab { background: #f7f7f7; border: none; padding: 5px 14px;
    margin-right: 1px; color: #333; }
QTabWidget#Ribbon > QTabBar::tab:selected { background: #ffffff; color: #1f5a96; font-weight: 600;
    border: 1px solid #d0d0d0; border-bottom: 1px solid #ffffff; }
QTabWidget#Ribbon > QTabBar::tab:hover:!selected { background: #e9f1fa; }
QToolButton#FileButton { background: #2b6cb0; color: #ffffff; border: none; padding: 5px 18px;
    margin: 0 4px 0 0; }
QToolButton#FileButton:hover { background: #3b82c9; }
QToolButton#FileButton::menu-indicator { image: none; width: 0; }
QTabWidget#Ribbon QScrollArea, QWidget#RibbonPage { background: #f7f7f7; border: none; }
QFrame#RibbonGroup { border: none; border-right: 1px solid #e3e3e3; background: #ffffff; }
QLabel#RibbonGroupLabel { color: #7a7a7a; font-size: 8pt; background: transparent; }
QFrame#RibbonGroup QToolButton { border: 1px solid transparent; border-radius: 3px; padding: 2px 3px;
    background: transparent; }
QFrame#RibbonGroup QToolButton:hover { background: #e5f1fb; border-color: #a9d1f3; }
QFrame#RibbonGroup QToolButton:pressed { background: #cce4f7; }
QFrame#RibbonGroup QToolButton:disabled { color: #a0a4aa; }

/* side panels and the activity bar */
QWidget#ActivityBar { background: #e9ecf0; border-right: 1px solid #c9d2dd; }
QWidget#ActivityBar QToolButton { border: none; border-left: 3px solid transparent; padding: 7px 6px;
    background: transparent; }
QWidget#ActivityBar QToolButton:hover { background: #dbe3ec; }
QWidget#ActivityBar QToolButton:checked { border-left-color: #2b6cb0; background: #ffffff; }
QWidget#PanelHeader { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f7f9fc, stop:1 #e7ecf2);
    border: 1px solid #c9d2dd; }
QLabel#PanelTitle { font-weight: 600; color: #26364a; padding: 4px 8px; background: transparent; }
QToolButton#PanelHide { border: none; background: transparent; padding: 0 6px; font-weight: 600;
    color: #4a5868; }
QToolButton#PanelHide:hover { background: #d3dde8; }
QWidget#Panel { background: #ffffff; }
QSplitter::handle { background: #d5d9de; }
QSplitter::handle:hover { background: #8fb8e0; }

/* tree */
QTreeWidget { background: #ffffff; border: none; outline: 0; }
QTreeWidget::item { height: 21px; padding-right: 4px; }
QTreeWidget::item:selected { background: #cce4f7; color: #1f2328; border: 1px solid #8fc2ec; }
QTreeWidget::item:hover:!selected { background: #eef5fc; }
QLabel#LegendLabel { color: #5c6470; font-size: 8pt; }

/* settings pages */
QWidget#PageBody { background: #ffffff; }
QLabel#PageTitle { font-size: 13pt; font-weight: 600; color: #1d3557; }
QLabel#PageCrumb { color: #6b7380; }
QLabel#WarningList { color: #7a4a00; }
QLabel#NoWarnings { color: #2f7a44; }
/* page buttons: one height, no icons, rows of equal widths (pages.FlowLayout) */
QWidget#PageBody QPushButton { min-height: 20px; min-width: 52px; padding: 3px 14px; }
QToolButton#SectionHeader { background: #eef1f5; border: 1px solid #d9dfe7; padding: 3px 6px;
    font-weight: 600; color: #2a3a4f; text-align: left; }
QLabel#FieldValue { background: #f3f5f7; border: 1px solid #d3d8de; padding: 2px 5px; }
QLabel#FieldValueResult { background: #f2f8f2; border: 1px solid #b9d7bd; padding: 2px 5px; font-weight: 600; }
QLabel#FieldUnit { color: #5c6470; }
QFrame#Callout { background: #fff8e6; border: 1px solid #e0b252; border-radius: 3px; }
QFrame#CalloutGood { background: #f0faf2; border: 1px solid #9fd0a9; border-radius: 3px; }
QFrame#CalloutInfo { background: #eef5fc; border: 1px solid #a9c9ec; border-radius: 3px; }
QLabel#Note { color: #5c6470; }
QFrame#CurvePicker { background: #ffffff; border: 1px solid #b9c2cc; }
QWidget#LogPlot { background: #ffffff; }
QToolButton#PickerButton { background: #ffffff; border: 1px solid #b9c2cc; border-radius: 3px;
    padding: 2px 6px; }
QToolButton#PickerButton:hover { border-color: #5a85e2; }
QScrollArea#ResultScroll, QWidget#ResultBody { background: #ffffff; border: none; }
QLabel#PickHint { background: #fff4c2; border-bottom: 1px solid #e0c060; padding: 5px 10px;
    color: #5a4500; font-weight: 600; }

/* buttons */
QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #eceff2);
    border: 1px solid #adb5bd; border-radius: 3px; padding: 4px 12px; min-height: 18px; }
QPushButton:hover { border-color: #5a85e2; }
QPushButton:pressed { background: #dfe6ee; }
QPushButton:disabled { color: #9aa0a6; background: #f3f3f3; border-color: #d5d5d5; }
QPushButton#Primary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3b82c9, stop:1 #2b6cb0);
    color: #ffffff; border-color: #1f5a96; }
QPushButton#Primary:disabled { background: #a9c3df; border-color: #a9c3df; color: #f4f7fb; }

/* graphics and bottom tabs */
QTabWidget#GraphicsTabs::pane, QTabWidget#BottomTabs::pane { border: 1px solid #c3c7cc; background: #ffffff; top: -1px; }
QTabWidget#GraphicsTabs > QTabBar::tab, QTabWidget#BottomTabs > QTabBar::tab {
    background: #f1f2f4; border: 1px solid #d0d0d0; padding: 4px 12px; margin-right: 2px; color: #444; }
QTabWidget#GraphicsTabs > QTabBar::tab:selected, QTabWidget#BottomTabs > QTabBar::tab:selected {
    background: #ffffff; border-bottom-color: #ffffff; font-weight: 600; color: #1f3550; }
QWidget#Canvas { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #fdfdfe, stop:1 #dfe6ef); }
QGraphicsView { border: none; background: transparent; }
QToolBar { background: #ffffff; border: none; border-bottom: 1px solid #e3e3e3; spacing: 3px; padding: 2px; }

QTextEdit#Messages { font-family: Consolas, 'Cascadia Mono', monospace; font-size: 9pt; border: none; }
QScrollArea#ModelInfo { background: #ffffff; border: none; }
QLabel#ModelInfoText { background: #ffffff; color: #1f2328; padding: 4px; }
QPlainTextEdit#StepLog { background: #f8f9fa; color: #343a40; border: 1px solid #dee2e6;
    font-family: Consolas, 'Cascadia Mono', monospace; font-size: 8pt; }
QTableView { gridline-color: #e3e6ea; border: none; selection-background-color: #cce4f7;
    selection-color: #1f2328; }
QHeaderView::section { background: #eef1f5; border: 1px solid #d6dce4; padding: 3px 6px; font-weight: 600; }

/* inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #b9c2cc;
    padding: 2px 5px; min-height: 18px; }
QComboBox QAbstractItemView { background: #ffffff; color: #1f2328; border: 1px solid #b9c2cc;
    outline: 0; selection-background-color: #cce4f7; selection-color: #1f2328; }
QComboBox QAbstractItemView::item { min-height: 22px; padding: 0 6px; }
QComboBox QAbstractItemView::item:hover { background: #e5f1fb; color: #1f2328; }
QComboBox QAbstractItemView::item:selected { background: #cce4f7; color: #1f2328; }

/* popups and dialogs: light whatever the Windows colour mode */
QMenu { background: #ffffff; border: 1px solid #b9c2cc; padding: 3px 0; }
QMenu::item { padding: 5px 22px 5px 28px; background: transparent; color: #1f2328; }
QMenu::item:selected { background: #e5f1fb; }
QMenu::item:disabled { color: #a0a4aa; }
QMenu::icon { padding-left: 6px; }
QMenu::separator { height: 1px; background: #e0e3e7; margin: 3px 8px; }
QDialog, QMessageBox, QInputDialog, QFileDialog { background: #f3f4f6; }
QToolTip { background: #ffffdc; color: #1f2328; border: 1px solid #b9b9a0; padding: 2px; }
QListWidget { border: 1px solid #d3d8de; background: #ffffff; }
QTextBrowser { border: 1px solid #d3d8de; background: #ffffff; color: #1f2328; }
QListWidget::item:selected { background: #cce4f7; color: #1f2328; }

/* scroll bars: always drawn (the Windows 11 style hides them until hovered) */
QScrollBar:vertical { background: #f0f2f5; width: 12px; margin: 0; border: none;
    border-left: 1px solid #dde2e8; }
QScrollBar:horizontal { background: #f0f2f5; height: 12px; margin: 0; border: none;
    border-top: 1px solid #dde2e8; }
QScrollBar::handle:vertical { background: #b3bcc7; min-height: 28px; border-radius: 4px;
    margin: 2px 2px 2px 3px; }
QScrollBar::handle:horizontal { background: #b3bcc7; min-width: 28px; border-radius: 4px;
    margin: 3px 2px 2px 2px; }
QScrollBar::handle:hover { background: #8a9aae; }
QScrollBar::handle:pressed { background: #2b6cb0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* status bar */
QStatusBar { background: #f3f3f3; border-top: 1px solid #cfcfcf; }
QStatusBar QLabel { padding: 0 8px; color: #333; }
QStatusBar::item { border: none; border-right: 1px solid #d6d6d6; }
"""

# Spin-box arrow buttons drawn by the stylesheet itself, so where they are
# drawn and where they click are the same rectangle under any Qt style.
_ASSETS = (Path(__file__).resolve().parent.parent / 'assets').as_posix()
STYLESHEET += f"""
QAbstractSpinBox {{ padding-right: 20px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border;
    width: 18px; background: #f3f5f8; border-left: 1px solid #c9d0d8; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-bottom: 1px solid #dde2e8; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: #e5f1fb; }}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{ background: #cce4f7; }}
QAbstractSpinBox::up-arrow {{ image: url({_ASSETS}/spin_up.svg); width: 9px; height: 6px; }}
QAbstractSpinBox::down-arrow {{ image: url({_ASSETS}/spin_down.svg); width: 9px; height: 6px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({_ASSETS}/spin_up_off.svg); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({_ASSETS}/spin_down_off.svg); }}
QListView::indicator {{ width: 13px; height: 13px; border: 1px solid #8a96a3; border-radius: 2px;
    background: #ffffff; }}
QListView::indicator:checked {{ background: #2b6cb0; border-color: #1f5a96;
    image: url({_ASSETS}/check.svg); }}
QListView::indicator:disabled {{ background: #e3e6ea; border-color: #b9c2cc; }}
QListView::indicator:checked:disabled {{ background: #9fb3c8; border-color: #9fb3c8; }}
"""


def light_palette():
    """A light palette, so panels the stylesheet doesn't cover stay light
    when Windows is in dark mode."""
    from PyQt6.QtGui import QColor, QPalette
    pal = QPalette()
    R = QPalette.ColorRole
    for role, colour in ((R.Window, '#f3f4f6'), (R.WindowText, TEXT), (R.Base, '#ffffff'),
                         (R.AlternateBase, '#f6f8fa'), (R.Text, TEXT), (R.Button, '#f3f4f6'),
                         (R.ButtonText, TEXT), (R.ToolTipBase, '#ffffdc'), (R.ToolTipText, TEXT),
                         (R.Highlight, '#cce4f7'), (R.HighlightedText, TEXT),
                         (R.PlaceholderText, '#8a9099'), (R.Link, ACCENT)):
        pal.setColor(role, QColor(colour))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor('#a0a4aa'))
    return pal


def fusion_style():
    """One Fusion style for every Studio window, owned by the application:
    a widget does not own its style, and the style must outlive every
    widget that uses it."""
    app = QApplication.instance()
    style = app.findChild(QStyle, 'StudioFusion')
    if style is None:
        style = QStyleFactory.create('Fusion')
        style.setObjectName('StudioFusion')
        style.setParent(app)
    return style


class StudioLook(QObject):
    """Application event filter, scoped to one Studio window and what it
    opens: dialogs and popups get the light palette, spin boxes and
    dropdowns Qt's Fusion style (the Windows 11 style lays their arrows out
    differently from the stylesheet, so a click can miss the arrow it is
    on), and dropdowns may shrink below their longest entry."""

    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.palette = light_palette()
        self.fusion = fusion_style()
        self._on = False

    def attach(self):
        """Watch the application's widgets.  Only while the window is open:
        a Python filter called for widgets being destroyed can crash."""
        app = QApplication.instance()
        if not self._on and app is not None:
            app.installEventFilter(self)
            app.aboutToQuit.connect(self.detach)
            self._on = True

    def detach(self):
        app = QApplication.instance()
        if self._on and app is not None:
            app.removeEventFilter(self)
            app.aboutToQuit.disconnect(self.detach)
            self._on = False

    def ours(self, w):
        while w is not None:
            if w is self.root:
                return True
            w = w.parentWidget()
        return False

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Polish and isinstance(obj, QWidget) and self.ours(obj):
            if (isinstance(obj, (QAbstractSpinBox, QComboBox))
                    and not obj.testAttribute(Qt.WidgetAttribute.WA_SetStyle)):
                obj.setStyle(self.fusion)
                if isinstance(obj, QComboBox):
                    obj.setSizeAdjustPolicy(
                        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                    obj.setMinimumContentsLength(8)
                    # The default delegate ignores the stylesheet's item rules
                    # (no hover highlight); a styled one follows them.
                    obj.setItemDelegate(QStyledItemDelegate(obj))
                    obj.view().setMouseTracking(True)
                    obj.view().viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            if obj.isWindow() and obj is not self.root and not obj.property('studio_light'):
                obj.setProperty('studio_light', True)
                obj.setPalette(self.palette)
        return False
