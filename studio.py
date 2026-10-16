"""Drop Analysis Studio: the new main window (UI overhaul preview).

    python studio.py [session folder]

The older windows stay available through main.py until every analysis has
moved here (see Academic/ui-foundation-plan.md).
"""
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

APP_ICON = Path(__file__).resolve().parent / "gui" / "assets" / "app_icon.ico"


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "IFTProperties.DropAnalysisStudio")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON)))
    from gui.shell.main_window import StudioWindow
    window = StudioWindow()
    window.show()
    if len(sys.argv) > 1:
        window.open_session(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
