from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import (QFont, QPixmap, QIcon)

from ChessNotation.Interface.LoaderWidget import LoaderWidget
from ChessNotation.Interface.UsefulFuncs import *


class LoadDialog(QDialog):

    def __init__(self, media_type: str):
        super().__init__()
        self.media_type = media_type
        self.media_filter: str = "Video Files (*.mp4)"
        if media_type == 'img':
            self.media_filter: str = "Images (*.png *.jpg)"

        self.big_font = QFont('Arial', 13)
        self.small_font = QFont('Arial', 13)
        self.is_accept: bool = False

        self.loader_media = LoaderWidget(self.media_filter, font=self.small_font)
        self.loader_alg_not = LoaderWidget(font=self.big_font, default_path='D:\\stud\\diplom\\diplom\\notation_test\\alg')
        self.loader_fen_not = LoaderWidget(font=self.big_font, default_path='D:\\stud\\diplom\\diplom\\notation_test\\fen')

        self.board_spinbox = create_spinbox(1, 20, self.big_font, start_val=10)
        self.pieces_spinbox = create_spinbox(1, 10, self.big_font, start_val=3)
        self.auto_radio_btn = create_radio_btn("Автоматический", True, self.big_font)
        self.manual_radio_btn = create_radio_btn("По нажатию кнопки", False, self.big_font)

        self.initUI()

    def initUI(self):
        self.setFixedSize(600, 630)

        if self.media_type == 'img':
            source_lbl = create_label('Выберите изображение:', self.big_font)
        else:
            source_lbl = create_label('Выберите видео (путь, https или номер порта):', self.big_font)

        alg_not_lbl = create_label('Папка для сохранения алгебраичечской нотации:', self.big_font)
        fen_not_lbl = create_label('Папка для сохранения FEN нотации:', self.big_font)
        board_lbl = create_label('Колличество кадров для усреднения доски:', self.big_font)
        pieces_lbl = create_label('Колличество кадров для усреднения фигур:', self.big_font)
        board_detect_lbl = create_label('Поиск сетки доски:', self.big_font)

        ok_btn = create_btn('Загрузить', self.big_font, self.click_accept, 35, 120)

        button_group = QButtonGroup()
        button_group.addButton(self.auto_radio_btn)
        button_group.addButton(self.manual_radio_btn)

        board_frames_h_layout = QHBoxLayout()
        pieces_frames_h_layout = QHBoxLayout()

        board_frames_h_layout.addWidget(board_lbl)
        board_frames_h_layout.addSpacing(10)
        board_frames_h_layout.addWidget(self.board_spinbox)
        board_frames_h_layout.addStretch()

        pieces_frames_h_layout.addWidget(pieces_lbl)
        pieces_frames_h_layout.addSpacing(10)
        pieces_frames_h_layout.addWidget(self.pieces_spinbox)
        pieces_frames_h_layout.addStretch()

        location_v_layout = QVBoxLayout()
        accuracy_v_layout = QVBoxLayout()
        location_v_layout.setContentsMargins(15, 5, 15, 5)
        accuracy_v_layout.setContentsMargins(15, 5, 15, 5)

        location_gbox = create_groupbox('Расположение', location_v_layout, self.big_font)
        accuracy_gbox = create_groupbox('Точность', accuracy_v_layout, self.big_font)

        main_v_lay = QVBoxLayout()
        main_v_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)

        main_v_lay.setContentsMargins(20, 10, 20, 0)

        location_v_layout.addSpacing(5)
        location_v_layout.addWidget(source_lbl)
        location_v_layout.addWidget(self.loader_media)
        location_v_layout.addSpacing(20)
        location_v_layout.addWidget(alg_not_lbl)
        location_v_layout.addWidget(self.loader_alg_not)
        location_v_layout.addSpacing(20)
        location_v_layout.addWidget(fen_not_lbl)
        location_v_layout.addWidget(self.loader_fen_not)
        location_v_layout.addSpacing(5)

        accuracy_v_layout.addSpacing(5)
        accuracy_v_layout.addLayout(board_frames_h_layout)
        accuracy_v_layout.addSpacing(20)
        accuracy_v_layout.addLayout(pieces_frames_h_layout)
        accuracy_v_layout.addSpacing(20)
        accuracy_v_layout.addWidget(board_detect_lbl)
        accuracy_v_layout.addSpacing(5)
        accuracy_v_layout.addWidget(self.auto_radio_btn)
        accuracy_v_layout.addWidget(self.manual_radio_btn)

        accuracy_v_layout.addSpacing(5)

        btn_h_layout = QHBoxLayout()
        btn_h_layout.addStretch(1)
        btn_h_layout.addWidget(ok_btn)

        main_v_lay.addSpacing(10)
        main_v_lay.addWidget(location_gbox)
        main_v_lay.addSpacing(20)
        main_v_lay.addWidget(accuracy_gbox)
        main_v_lay.addStretch(1)
        main_v_lay.addLayout(btn_h_layout)
        main_v_lay.addSpacing(10)
        self.setLayout(main_v_lay)

    def click_accept(self):
        self.is_accept = True
        self.close()


