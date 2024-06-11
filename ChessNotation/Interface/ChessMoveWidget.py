from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QListWidget
from ChessNotation.Interface.UsefulFuncs import *

class ChessMoveWidget(QWidget):
    signal_save_alg_not = pyqtSignal()

    def __init__(self):
        # "Images (*.png *.jpg)"
        # "Video Files (*.mp4)"
        super().__init__()
        self.big_font = QFont('Arial', 13)
        self.moves_list = QListWidget()
        self.save_btn = create_btn('Сохранить', self.big_font, self.signal_save_alg_not.emit)
        self.initUI()

    def initUI(self):
        main_h_lay = QHBoxLayout()
        moves_v_lay = QVBoxLayout()
        moves_gbox = create_groupbox('Ходы', moves_v_lay, self.big_font)
        self.save_btn.setEnabled(False)
        self.moves_list.setSpacing(5)

        moves_v_lay.addWidget(self.moves_list)
        moves_v_lay.addWidget(self.save_btn)
        moves_gbox.setLayout(moves_v_lay)

        main_h_lay.addWidget(moves_gbox)
        self.setLayout(main_h_lay)

    def change_moves(self, move_describe: int, move_str: str):
        if self.moves_list.count() > 0:
            if self.moves_list.item(0).text() == '':
                self.moves_list.takeItem(0)
        if move_describe == 0:
            self.moves_list.addItem(move_str)
        elif move_describe == 1:
            self.moves_list.takeItem(self.moves_list.count()-1)
            self.moves_list.addItem(move_str)
