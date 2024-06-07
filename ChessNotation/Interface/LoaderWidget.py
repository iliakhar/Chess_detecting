from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import *
from ChessNotation.Interface.UsefulFuncs import *

class LoaderWidget(QWidget):

    def __init__(self, file_types: str = '', is_load: bool = True, font: QFont = QFont('Arial', 13)):
        # "Images (*.png *.jpg)"
        # "Video Files (*.mp4)"
        super().__init__()
        self.font = font
        self.is_load = is_load
        self.file_types = file_types
        self.filename_edit = QLineEdit()
        self.initUI()

    def initUI(self):
        main_h_lay = QHBoxLayout()
        browse_btn = create_btn('Обзор', self.font, self.open_file_dialog, min_h=32, min_w=110)
        self.filename_edit.setMinimumHeight(30)
        self.filename_edit.setFont(self.font)

        main_h_lay.setContentsMargins(0, 0, 0, 0)
        main_h_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_h_lay.addWidget(self.filename_edit, 1)
        main_h_lay.addSpacing(5)
        main_h_lay.addWidget(browse_btn)
        self.setLayout(main_h_lay)

    def open_file_dialog(self):
        if self.is_load:
            if self.file_types != '':
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select a File",
                    filter=self.file_types)
            else:
                filename = QFileDialog.getExistingDirectory(self, "Select a Directory")
        else:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save File",
                filter=self.file_types)
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))