"""
gui.shell.theme
===============

Light theme for Drop Analysis Studio, in the style of engineering
simulation software (light panels, blue accents).  Applied to the studio
window only, so the older dark windows keep their own look.
"""

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
QWidget#RibbonPage { background: #ffffff; }
QFrame#RibbonGroup { border: none; border-right: 1px solid #e3e3e3; background: #ffffff; }
QLabel#RibbonGroupLabel { color: #7a7a7a; font-size: 8pt; background: transparent; }
QFrame#RibbonGroup QToolButton { border: 1px solid transparent; border-radius: 3px; padding: 2px 3px;
    background: transparent; }
QFrame#RibbonGroup QToolButton:hover { background: #e5f1fb; border-color: #a9d1f3; }
QFrame#RibbonGroup QToolButton:pressed { background: #cce4f7; }
QFrame#RibbonGroup QToolButton:disabled { color: #a0a4aa; }

/* docks */
QDockWidget { font-weight: 600; color: #26364a; }
QDockWidget::title { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f7f9fc, stop:1 #e7ecf2);
    border: 1px solid #c9d2dd; padding: 4px 8px; text-align: left; }
QDockWidget > QWidget { background: #ffffff; border: 1px solid #c3c7cc; border-top: none; }

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
QToolButton#SectionHeader { background: #eef1f5; border: 1px solid #d9dfe7; padding: 3px 6px;
    font-weight: 600; color: #2a3a4f; text-align: left; }
QLabel#FieldValue { background: #f3f5f7; border: 1px solid #d3d8de; padding: 2px 5px; }
QLabel#FieldValueResult { background: #f2f8f2; border: 1px solid #b9d7bd; padding: 2px 5px; font-weight: 600; }
QLabel#FieldUnit { color: #5c6470; }
QFrame#Callout { background: #fff8e6; border: 1px solid #e0b252; border-radius: 3px; }
QFrame#CalloutGood { background: #f0faf2; border: 1px solid #9fd0a9; border-radius: 3px; }
QFrame#CalloutInfo { background: #eef5fc; border: 1px solid #a9c9ec; border-radius: 3px; }
QLabel#Note { color: #5c6470; }

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
QTableView { gridline-color: #e3e6ea; border: none; selection-background-color: #cce4f7;
    selection-color: #1f2328; }
QHeaderView::section { background: #eef1f5; border: 1px solid #d6dce4; padding: 3px 6px; font-weight: 600; }

/* inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #b9c2cc;
    padding: 2px 5px; min-height: 18px; }
QListWidget { border: 1px solid #d3d8de; background: #ffffff; }

/* status bar */
QStatusBar { background: #f3f3f3; border-top: 1px solid #cfcfcf; }
QStatusBar QLabel { padding: 0 8px; color: #333; }
QStatusBar::item { border: none; border-right: 1px solid #d6d6d6; }
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
