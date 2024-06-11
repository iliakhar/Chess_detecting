from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QTextEdit
from ChessNotation.Interface.UsefulFuncs import *

class FENInfoWidget(QWidget):

    def __init__(self):
        # "Images (*.png *.jpg)"
        # "Video Files (*.mp4)"
        super().__init__()
        self.save_path: str = ''
        self.filename_ind_fen: int = 0
        self.big_font = QFont('Arial', 13)
        self.fen_text = QTextEdit(self)
        self.save_btn = create_btn('Сохранить FEN', self.big_font, self.save_fen_not)
        self.initUI()

    def initUI(self):
        main_h_lay = QHBoxLayout()
        moves_v_lay = QVBoxLayout()
        fen_gbox = create_groupbox('FEN', moves_v_lay, self.big_font)

        self.save_btn.setEnabled(False)

        moves_v_lay.addWidget(self.fen_text)
        moves_v_lay.addWidget(self.save_btn)
        fen_gbox.setLayout(moves_v_lay)

        main_h_lay.addWidget(fen_gbox)
        self.setLayout(main_h_lay)

    def change_fen(self, fen_str: str):
        self.fen_text.setText(fen_str)

    def save_fen_not(self):
        path = str(self.save_path) + '\\' + str(self.filename_ind_fen) + '.txt'
        self.filename_ind_fen += 1
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.fen_text.toPlainText())
