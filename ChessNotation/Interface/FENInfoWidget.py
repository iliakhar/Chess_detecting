from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QTextEdit, QButtonGroup
from ChessNotation.Interface.UsefulFuncs import *


class FENInfoWidget(QWidget):

    def __init__(self):
        # "Images (*.png *.jpg)"
        # "Video Files (*.mp4)"
        super().__init__()
        self.save_path: str = ''
        self.media_type = 'video'
        self.filename_ind_fen: int = 0
        self.big_font = QFont('Arial', 13)
        self.fen_text = QTextEdit(self)
        self.save_btn = create_btn('Сохранить FEN', self.big_font, self.save_fen_not)

        self.wk_check = create_checkbox('(O-O)', True, self.big_font)
        self.wq_check = create_checkbox('(O-O-O)', True, self.big_font)
        self.bk_check = create_checkbox('(O-O)', True, self.big_font)
        self.bq_check = create_checkbox('(O-O-O)', True, self.big_font)

        self.wm_radio_btn = create_radio_btn("Ход белых", True, self.big_font)
        self.bm_radio_btn = create_radio_btn("Ход черных", False, self.big_font)
        self.moves_v_lay = QVBoxLayout()
        self.control_h_frame = QFrame(self)

        self.initUI()

    def initUI(self):
        main_h_lay = QHBoxLayout()
        w_control_v_lay = QVBoxLayout()
        b_control_v_lay = QVBoxLayout()
        control_h_lay = QHBoxLayout()

        self.control_h_frame.setLayout(control_h_lay)
        self.control_h_frame.hide()

        fen_gbox = create_groupbox('FEN', self.moves_v_lay, self.big_font)
        w_label = create_label('Белые', self.big_font)
        b_label = create_label('Черные', self.big_font)
        button_group = QButtonGroup()
        button_group.addButton(self.wm_radio_btn, 0)
        button_group.addButton(self.bm_radio_btn, 1)

        self.save_btn.setEnabled(False)

        w_control_v_lay.addWidget(w_label)
        w_control_v_lay.addWidget(self.wm_radio_btn)
        w_control_v_lay.addWidget(self.wk_check)
        w_control_v_lay.addWidget(self.wq_check)
        b_control_v_lay.addWidget(b_label)
        b_control_v_lay.addWidget(self.bm_radio_btn)
        b_control_v_lay.addWidget(self.bk_check)
        b_control_v_lay.addWidget(self.bq_check)
        control_h_lay.addLayout(w_control_v_lay)
        control_h_lay.addLayout(b_control_v_lay)
        self.moves_v_lay.addWidget(self.fen_text)
        self.moves_v_lay.addWidget(self.control_h_frame)
        self.moves_v_lay.addWidget(self.save_btn)
        fen_gbox.setLayout(self.moves_v_lay)

        main_h_lay.addWidget(fen_gbox)
        self.setLayout(main_h_lay)

        self.wm_radio_btn.clicked.connect(self.change_fen_describe)
        self.bm_radio_btn.clicked.connect(self.change_fen_describe)
        self.wk_check.clicked.connect(self.change_fen_describe)
        self.wq_check.clicked.connect(self.change_fen_describe)
        self.bk_check.clicked.connect(self.change_fen_describe)
        self.bq_check.clicked.connect(self.change_fen_describe)

    def change_fen_describe(self):
        fen_str = self.fen_text.toPlainText()
        fen_list: list[str] = fen_str.split()
        if len(fen_list) != 6:
            return
        castling = ''
        if self.wk_check.isChecked():
            castling += 'K'
        if self.wq_check.isChecked():
            castling += 'Q'
        if self.bk_check.isChecked():
            castling += 'k'
        if self.bq_check.isChecked():
            castling += 'q'
        if castling == '':
            castling = '-'
        fen_list[2] = castling
        if self.wm_radio_btn.isChecked():
            fen_list[1] = 'w'
        else:
            fen_list[1] = 'b'
        fen_str = ' '.join(fen_list)
        self.fen_text.setText(fen_str)

    def set_media_type(self, media_type: str):
        if media_type == 'img':
            self.control_h_frame.show()
        else:
            self.control_h_frame.hide()
        self.media_type = media_type

    def change_fen(self, fen_str: str):
        self.fen_text.setText(fen_str)
        if self.media_type == 'img':
            self.change_fen_describe()

    def save_fen_not(self):
        path = str(self.save_path) + '\\' + str(self.filename_ind_fen) + '.txt'
        self.filename_ind_fen += 1
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.fen_text.toPlainText())
