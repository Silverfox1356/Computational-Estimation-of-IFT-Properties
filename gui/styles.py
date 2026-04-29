"""
gui.styles
==========

Single source of truth for the app's visual style.
    DARK_STYLESHEET          — Qt stylesheet applied to every window/dialog
    BTN_PRIMARY / SUCCESS / DANGER / ICON — inline button accents
    make_cosmetic_pen        — pen that stays 1 screen-px thick at every zoom
                               so overlay lines never disappear
    make_pixel_pen           — legacy 1-IMAGE-px pen used for user-drawn
                               calibration guides where pixel-spanning
                               matters more than zoom visibility
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
QWidget {
    background-color: #1b1d2a;
    color: #e6e6ee;
    font-family: 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
}
QDialog { background-color: #1b1d2a; }

QGroupBox {
    border: 1px solid #343852;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 8px 6px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #8ab4ff;
}

/* Buttons — gradient with clear pressed inversion */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #363a52, stop:1 #292c44);
    color: #e6e6ee;
    border: 1px solid #343852;
    border-bottom: 2px solid #15161f;
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #40456a, stop:1 #2f3350);
    border-color: #5a85e2;
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #1d2035, stop:1 #2a2d44);
    border-top: 2px solid #15161f;
    border-bottom: 1px solid #343852;
    padding-top: 6px;
    padding-bottom: 4px;
}
QPushButton:disabled {
    background: #222432;
    color: #565977;
    border: 1px solid #2a2d40;
}

/* ComboBoxes — visibly a dropdown, pressable drop-down area */
QComboBox {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2d3048, stop:1 #22243a);
    color: #e6e6ee;
    border: 1px solid #343852;
    border-bottom: 2px solid #15161f;
    border-radius: 3px;
    padding: 3px 8px;
    padding-right: 28px;
    min-height: 22px;
}
QComboBox:hover { border-color: #5a85e2; }
QComboBox:on {
    background: #242739;
    border-top: 2px solid #15161f;
    border-bottom: 1px solid #343852;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #343852;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3a3e58, stop:1 #282b42);
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}
QComboBox::drop-down:hover { background: #454a6a; }
QComboBox::drop-down:pressed { background: #1e2033; }
QComboBox::down-arrow {
    image: none;
    width: 0; height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-top:   6px solid #9ec4ff;
    margin-right: 7px;
    margin-top: 3px;
}
QComboBox::down-arrow:on {
    border-top-color: #ffffff;
    margin-top: 4px;
}
QComboBox QAbstractItemView {
    background-color: #22243a;
    color: #e6e6ee;
    selection-background-color: #3b63b4;
    border: 1px solid #343852;
    outline: 0;
    padding: 2px;
}

/* Line edits */
QLineEdit {
    background-color: #22243a;
    color: #e6e6ee;
    border: 1px solid #343852;
    border-radius: 3px;
    padding: 4px 7px;
    selection-background-color: #3b63b4;
}
QLineEdit:focus { border-color: #5a85e2; }

/* Radios, labels */
QRadioButton, QCheckBox { background: transparent; }
QLabel { background: transparent; }

/* Scroll area */
QScrollArea { background: transparent; border: 0; }
QScrollArea > QWidget > QWidget { background: transparent; }

QScrollBar:vertical, QScrollBar:horizontal {
    background: #1b1d2a;
    border: 0;
    border-radius: 5px;
    width: 10px;
    height: 10px;
}
QScrollBar::handle {
    background: #3a3e58;
    border-radius: 5px;
    min-height: 30px;
    min-width: 30px;
}
QScrollBar::handle:hover { background: #4a5278; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }

/* Slider */
QSlider::groove:horizontal {
    height: 6px; background: #343852; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5a85e2; width: 14px; margin: -4px 0; border-radius: 7px;
}
QSlider::handle:horizontal:pressed { background: #6fa0ff; }

/* "Card" frames used for result readouts */
QFrame#card {
    background-color: #20233a;
    border: 1px solid #343852;
    border-radius: 6px;
}
"""


# ---------------------------------------------------------------------------
# Inline button accents (all have pressed-state inversion)
# ---------------------------------------------------------------------------
BTN_PRIMARY = (
    "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #3a7ae8,stop:1 #2355c2);color:white;font-weight:600;"
    "border:1px solid #2c5fcb;border-bottom:2px solid #12275c;"
    "border-radius:4px;padding:5px 12px;min-height:22px;}"
    "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #4a8aef,stop:1 #2d66d4);}"
    "QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #17368a,stop:1 #254db0);border-top:2px solid #12275c;"
    "border-bottom:1px solid #2c5fcb;padding-top:6px;padding-bottom:4px;}"
    "QPushButton:disabled{background:#2a3650;color:#6a7a9a;"
    "border:1px solid #2a3650;}"
)

BTN_SUCCESS = (
    "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #3ba85a,stop:1 #27803e);color:white;font-weight:600;"
    "border:1px solid #2f9049;border-bottom:2px solid #114b22;"
    "border-radius:4px;padding:5px 12px;min-height:22px;}"
    "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #46b866,stop:1 #2d9148);}"
    "QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #1a5c2a,stop:1 #28833f);border-top:2px solid #114b22;"
    "border-bottom:1px solid #2f9049;padding-top:6px;padding-bottom:4px;}"
    "QPushButton:disabled{background:#2a3d30;color:#6a8a6e;"
    "border:1px solid #2a3d30;}"
)

BTN_DANGER = (
    "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #c24d4d,stop:1 #8f2a2a);color:white;font-weight:600;"
    "border:1px solid #a23636;border-bottom:2px solid #4a1212;"
    "border-radius:4px;padding:5px 12px;min-height:22px;}"
    "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #d05a5a,stop:1 #9b3232);}"
    "QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #6a1c1b,stop:1 #9a3233);border-top:2px solid #4a1212;"
    "border-bottom:1px solid #a23636;padding-top:6px;padding-bottom:4px;}"
)

BTN_ICON = (
    "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #363a52,stop:1 #292c44);color:#9ec4ff;"
    "border:1px solid #343852;border-bottom:2px solid #15161f;"
    "border-radius:4px;font-size:13pt;padding:2px 6px;min-height:22px;}"
    "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #40456a,stop:1 #2f3350);border-color:#5a85e2;}"
    "QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #1d2035,stop:1 #2a2d44);border-top:2px solid #15161f;"
    "border-bottom:1px solid #343852;padding-top:3px;padding-bottom:1px;}"
)


# ---------------------------------------------------------------------------
# Pen factories — "cosmetic" is the key to fixing the zoom-disappearing lines
# ---------------------------------------------------------------------------
def make_cosmetic_pen(color, width=2, dashed=False):
    """Pen that always renders at `width` SCREEN pixels regardless of zoom.

    Setting cosmetic=True makes QPen's width independent of the view
    transform, so overlay lines stay visible at every zoom level
    (the classic "lines disappear at certain zooms" problem is caused
    by non-cosmetic 1-px pens that sub-sample to nothing at fractional
    zoom levels).

    Position is still in scene (image-pixel) coordinates, so they remain
    pixel-accurate.
    """
    if isinstance(color, Qt.GlobalColor):
        color = QColor(color)
    pen = QPen(color, width)
    pen.setCosmetic(True)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    if dashed:
        pen.setDashPattern([6, 4])
    return pen


def make_pixel_pen(color):
    """Pen that bounds exactly 1 IMAGE pixel — used for user-drawn
    calibration guides where spanning a single image pixel matters more
    than visibility at zoom-out extremes."""
    if isinstance(color, Qt.GlobalColor):
        color = QColor(color)
    pen = QPen(color, 1)
    pen.setCosmetic(False)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    pen.setCapStyle(Qt.PenCapStyle.SquareCap)
    pen.setDashPattern([20, 5])
    return pen
