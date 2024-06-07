from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QLineEdit, QHBoxLayout


class ChessMoveWidget(QWidget):

    def __init__(self):
        # "Images (*.png *.jpg)"
        # "Video Files (*.mp4)"
        super().__init__()
        self.big_font = QFont('Arial', 13)
        self.filename_edit = QLineEdit()
        self.initUI()

    def initUI(self):
        main_h_lay = QHBoxLayout()
        self.setLayout(main_h_lay)