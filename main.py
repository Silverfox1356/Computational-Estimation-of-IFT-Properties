import sys
from pathlib import Path
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

APP_ICON = Path(__file__).resolve().parent / "gui" / "assets" / "app_icon.ico"


def main():
    # On Windows, give the app its own taskbar identity; otherwise the
    # taskbar groups it under python.exe and shows Python's icon.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "IFTProperties.DropAnalysis")

    # 1. Create the application instance
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON)))

    # 2. Create an instance of our main window
    window = MainWindow()

    # 3. Tell the window to show itself on the screen
    window.show()

    # 4. Start the application's event loop (keeps it running until you close it)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()