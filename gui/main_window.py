import sys
from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, QGroupBox
from PyQt6.QtCore import Qt
from gui.sessile_window import SessileWindow
from gui.pendant_window import PendantWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drop Analysis Software")
        self.resize(400, 350)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Select Drop Analysis Type")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        title_label.setFont(font)
        layout.addWidget(title_label)

        # --- CAMERA CONFIGURATION ---
        cam_box = QGroupBox("Hardware Configuration")
        cam_layout = QVBoxLayout()
        cam_layout.addWidget(QLabel("Select Active Camera:"))

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Basler Camera", "Laptop Webcam"])
        cam_layout.addWidget(self.camera_combo)

        cam_box.setLayout(cam_layout)
        layout.addWidget(cam_box)
        # ----------------------------

        self.sessile_btn = QPushButton("Sessile Drop")
        self.sessile_btn.setMinimumHeight(60)
        self.pendant_btn = QPushButton("Pendant Drop")
        self.pendant_btn.setMinimumHeight(60)

        layout.addWidget(self.sessile_btn)
        layout.addWidget(self.pendant_btn)

        self.sessile_btn.clicked.connect(self.open_sessile)
        self.pendant_btn.clicked.connect(self.open_pendant)

    def get_selected_camera(self):
        text = self.camera_combo.currentText()
        if text == "Basler Camera":
            return "Basler"
        return "Webcam"

    def open_sessile(self):
        # Pass the chosen camera type into the window
        self.sessile_window = SessileWindow(camera_type=self.get_selected_camera())
        self.sessile_window.show()

    def open_pendant(self):
        # Pass the chosen camera type into the window
        self.pendant_window = PendantWindow(camera_type=self.get_selected_camera())
        self.pendant_window.show()