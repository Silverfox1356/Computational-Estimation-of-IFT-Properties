"""
gui.shell.ribbon
================

A simple ribbon: tabs of labelled button groups, each button bound to a
QAction (so the same action can also sit in a menu and share its enabled
state).  A blue File button with a menu sits left of the tabs.  Every page
scrolls sideways when the window is narrower than its buttons.
"""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea,
                             QTabWidget, QToolButton, QVBoxLayout, QWidget)

RIBBON_HEIGHT = 122


class RibbonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('RibbonPage')
        # Paint the stylesheet background: otherwise the scroll area's
        # viewport shows through, black when Windows is in dark mode.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(2, 2, 2, 0)
        self._row.setSpacing(0)
        self._row.addStretch(1)

    def add_group(self, title, actions):
        group = QFrame()
        group.setObjectName('RibbonGroup')
        col = QVBoxLayout(group)
        col.setContentsMargins(3, 1, 3, 1)
        col.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(1)
        for a in actions:
            b = QToolButton()
            b.setDefaultAction(a)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setIconSize(QSize(26, 26))
            b.setMinimumWidth(48)
            b.setAutoRaise(True)
            row.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(row, 1)
        label = QLabel(title)
        label.setObjectName('RibbonGroupLabel')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(label)
        self._row.insertWidget(self._row.count() - 1, group)
        return group


class Ribbon(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Ribbon')
        self.setDocumentMode(True)
        self.setFixedHeight(RIBBON_HEIGHT)
        self.file_menu = QMenu(self)
        self.file_button = QToolButton()
        self.file_button.setObjectName('FileButton')
        self.file_button.setText('File')
        self.file_button.setMenu(self.file_menu)
        self.file_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setCornerWidget(self.file_button, Qt.Corner.TopLeftCorner)
        self.pages = {}

    def add_page(self, title):
        page = RibbonPage()
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.addTab(scroll, title)
        self.pages[title] = page
        return page
