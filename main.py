import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    # 1. Create the application instance
    app = QApplication(sys.argv)

    # 2. Create an instance of our main window
    window = MainWindow()

    # 3. Tell the window to show itself on the screen
    window.show()

    # 4. Start the application's event loop (keeps it running until you close it)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()