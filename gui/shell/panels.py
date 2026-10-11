"""
gui.shell.panels
================

The bottom panel's parts and the background job runner:

    MessagesPanel  timestamped history (info / ok / warn / error), newest last
    ProgressPanel  what is running, a progress bar and a Cancel button
    LogTable       the frame log as a table (core.frame_log columns)
    Job            runs one long task off the GUI thread with progress and cancel
"""

import datetime
import html
import math

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                             QTableView, QTextEdit, QVBoxLayout, QWidget)

from core import frame_log as fl
from gui.shell.theme import LEVEL_COLOURS


class MessagesPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Messages')
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    def add(self, text, level='info'):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        colour = LEVEL_COLOURS.get(level, LEVEL_COLOURS['info'])
        self.append(f'<span style="color:{colour}">[{stamp}] {html.escape(text)}</span>')


class ProgressPanel(QWidget):
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel('Nothing is running.')
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        row = QHBoxLayout()
        row.addWidget(self.bar, 1)
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_requested)
        row.addWidget(self.btn_cancel)
        lay.addWidget(self.label)
        lay.addLayout(row)
        lay.addStretch(1)

    def start(self, title):
        self.label.setText(title)
        self.bar.setRange(0, 0)          # busy until the first count arrives
        self.btn_cancel.setEnabled(True)

    def update_progress(self, i, n, text=''):
        self.bar.setRange(0, max(n, 1))
        self.bar.setValue(i)
        if text:
            self.label.setText(text)

    def finish(self, text):
        self.label.setText(text)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.btn_cancel.setEnabled(False)


class Job(QThread):
    """Run ``fn(progress, is_cancelled)`` in a thread.  ``progress(i, n,
    text='')`` may be called from the task; the result arrives in
    ``succeeded``, an exception's message in ``failed``."""
    progressed = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, title, fn, parent=None):
        super().__init__(parent)
        self.title = title
        self._fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from core.frame_series import FrameSeriesCancelled
        try:
            result = self._fn(lambda i, n, text='': self.progressed.emit(i, n, text),
                              lambda: self._cancel)
        except FrameSeriesCancelled:
            self.cancelled.emit()
        except Exception as e:                       # reported, never fatal
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.succeeded.emit(result)


class LogTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = None

    def set_log(self, log):
        self.beginResetModel()
        self._log = log
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if self._log is None or parent.isValid() else len(self._log.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._log is None or parent.isValid() else len(self._log.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self._log is None:
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        c = fl.COLUMNS[self._log.columns[section]]
        return f"{c.label} ({c.unit})" if c.unit else c.label

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if self._log is None or not index.isValid():
            return None
        key = self._log.columns[index.column()]
        v = self._log.rows[index.row()][key]
        if role == Qt.ItemDataRole.DisplayRole:
            if key in fl.BOOL_COLUMNS:
                return 'Yes' if v else 'No'
            return fl.format_cell(key, v)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in fl.TEXT_COLUMNS or key in fl.BOOL_COLUMNS:
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole and key in fl.BOOL_COLUMNS:
            return QColor('#2f7a44') if v else QColor('#c0392b')
        if role == Qt.ItemDataRole.ToolTipRole:
            return fl.COLUMNS[key].description
        return None


class LogTable(QTableView):
    row_selected = pyqtSignal(int)          # frame_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ = LogTableModel(self)
        self.setModel(self.model_)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setStretchLastSection(True)
        self.selectionModel().currentRowChanged.connect(self._row_changed)

    def set_log(self, log):
        self.model_.set_log(log)
        self.resizeColumnsToContents()

    def _row_changed(self, cur, _prev):
        log = self.model_._log
        if log is not None and cur.isValid():
            self.row_selected.emit(int(log.rows[cur.row()]['frame_index']))

    def select_frame(self, frame_index):
        log = self.model_._log
        if log is None:
            return
        for i, r in enumerate(log.rows):
            if r['frame_index'] == frame_index:
                if self.currentIndex().row() != i:
                    sm = self.selectionModel()
                    sm.blockSignals(True)       # no echo back to the viewer
                    self.selectRow(i)
                    sm.blockSignals(False)
                    self.viewport().update()
                self.scrollTo(self.model_.index(i, 0))
                return


def fmt(v, digits=4):
    """Short number for labels; blank for NaN."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    return f"{v:.{digits}g}"
