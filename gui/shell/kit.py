"""
gui.shell.kit
=============

The building blocks of a Settings page (see DESIGN.md): every page is made
of these, so pages look and behave alike.  Add to them rather than writing
one-off widgets on a page.

    Section            collapsible section with a form; .field() read-only values
    actions            the page's main buttons: equal widths, bottom-right
    button_row         a row of widgets that wraps when the panel is narrow
    flow_row           the same, with the FlowLayout options exposed
    page_button        a text-only page button (primary = blue)
    number_box         a number spin box
    sci_edit / parse_sci / parse_sci_list   numbers like 1e-9 typed as text
    warnings_section   a result's warnings as bullets, or "No warnings"
    callout / note     a highlighted box / a muted line of text
"""

import html
import math

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import (QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
                             QLayout, QLineEdit, QPushButton, QSizePolicy, QToolButton,
                             QVBoxLayout, QWidget)


class FlowLayout(QLayout):
    """Lays widgets out left to right and starts a new line when the next
    one does not fit (Qt's flow layout example), so a row of buttons wraps
    instead of holding the Settings panel wide.  ``uniform`` gives every
    item the width of the widest, ``right`` aligns each line right (the
    page's main actions sit bottom-right)."""

    def __init__(self, parent=None, spacing=6, uniform=False, right=False):
        super().__init__(parent)
        self._items = []
        self.uniform = uniform
        self.right = right
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        if self.uniform:
            size = size.expandedTo(QSize(self._widest(), 0))
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _widest(self):
        return max((it.sizeHint().width() for it in self._items if not it.isEmpty()), default=0)

    def _place(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        sp = self.spacing()
        same = self._widest() if self.uniform else None
        lines, line, x = [], [], r.x()
        for it in self._items:
            if it.isEmpty():
                continue
            hint = it.sizeHint()
            w = min(same or hint.width(), max(r.width(), it.minimumSize().width()))
            if line and x + w > r.right() + 1:
                lines.append(line)
                line, x = [], r.x()
            line.append((it, w, hint.height()))
            x += w + sp
        if line:
            lines.append(line)
        y = r.y()
        for line in lines:
            used = sum(w for _, w, _ in line) + sp * (len(line) - 1)
            x = r.x() + (max(0, r.width() - used) if self.right else 0)
            for it, w, h in line:
                if apply:
                    it.setGeometry(QRect(QPoint(x, y), QSize(w, h)))
                x += w + sp
            y += max(h for _, _, h in line) + sp
        return (y - sp if lines else y) - rect.y() + m.bottom()


def flow_row(*widgets, uniform=False, right=False):
    """A widget holding ``widgets`` in a FlowLayout."""
    w = QWidget()
    f = FlowLayout(w, uniform=uniform, right=right)
    for x in widgets:
        f.addWidget(x)
    return w


def actions(*buttons):
    """The page's main buttons (Run, Apply ...): equal widths, bottom-right."""
    row = flow_row(*buttons, uniform=True, right=True)
    row.setContentsMargins(0, 8, 0, 0)
    return row


class Section(QWidget):
    """A collapsible section with a form inside."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName('SectionHeader')
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setArrowType(Qt.ArrowType.DownArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Ignored: a long title is clipped instead of holding the panel wide.
        self.header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.header.setToolTip(title)
        self.header.toggled.connect(self._toggle)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(4, 6, 2, 2)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(5)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # When the panel is narrow a label goes above its box, not beside it.
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        lay.addWidget(self.header)
        lay.addWidget(self.body)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.header.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def field(self, label, value, unit='', result=False, buttons=()):
        """A read-only value with its unit; ``buttons`` sit on the same row
        (Load..., Go to ...).  ``value`` None: the buttons alone."""
        row = QHBoxLayout()
        row.setSpacing(6)
        if value is None:
            for b in buttons:
                row.addWidget(b)
            row.addStretch(1)
            self.form.addRow(label, row)
            return None
        v = QLabel(str(value))
        v.setObjectName('FieldValueResult' if result else 'FieldValue')
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.setWordWrap(True)
        v.setToolTip(str(value))             # long paths are clipped, not lost
        v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.setMinimumWidth(60)
        row.addWidget(v, 1)
        if unit or not buttons:
            u = QLabel(unit)
            u.setObjectName('FieldUnit')
            u.setMinimumWidth(30)
            row.addWidget(u)
        for b in buttons:
            row.addWidget(b)
        self.form.addRow(label, row)
        return v

    def widget(self, w):
        self.form.addRow(w)


def warnings_section(items, title='Warnings'):
    """The warnings of a result as a bulleted list, or 'No warnings'."""
    sec = Section(title)
    if items:
        lbl = QLabel('<ul style="margin:0 0 0 -24px">'
                     + ''.join(f'<li style="margin-bottom:3px">{html.escape(w)}</li>' for w in items)
                     + '</ul>')
        lbl.setObjectName('WarningList')
        lbl.setTextFormat(Qt.TextFormat.RichText)
    else:
        lbl = QLabel('No warnings')
        lbl.setObjectName('NoWarnings')
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    sec.widget(lbl)
    return sec


def callout(text, kind='warn'):
    box = QFrame()
    box.setObjectName({'warn': 'Callout', 'good': 'CalloutGood', 'info': 'CalloutInfo'}[kind])
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 6, 8, 6)
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lay.addWidget(lbl)
    return box


def note(text):
    lbl = QLabel(text)
    lbl.setObjectName('Note')
    lbl.setWordWrap(True)
    return lbl


def number_box(value, lo, hi, decimals=3, step=None, suffix=''):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(float(value))
    s.setSingleStep(step or 10 ** -decimals * 10)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    return s


def button_row(*widgets, uniform=True):
    """Widgets side by side, wrapping onto a new line when the panel is
    narrow; buttons get one width."""
    return flow_row(*widgets, uniform=uniform)


def page_button(text, slot, primary=False, tip=''):
    """A page button: text only, so every button on a page looks alike."""
    b = QPushButton(text)
    if primary:
        b.setObjectName('Primary')
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(lambda: slot())        # not slot(checked): see StudioWindow._act
    return b


def parse_sci(text, name):
    try:
        v = float(text.strip().replace(',', '.'))
    except ValueError:
        raise ValueError(f"{name}: '{text}' is not a number") from None
    if not (v > 0 and math.isfinite(v)):
        raise ValueError(f"{name} must be a positive number")
    return v


def sci_edit(value):
    e = QLineEdit(f"{value:.3g}")
    e.setToolTip('A number such as 1e-9')
    return e


def parse_sci_list(text, name):
    out = []
    for part in text.replace(';', ',').split(','):
        if part.strip():
            out.append(parse_sci(part, name))
    return out
