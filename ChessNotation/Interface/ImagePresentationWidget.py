import numpy as np
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QImage
from PyQt6.QtWidgets import QWidget, QLineEdit, QHBoxLayout, QTabWidget, QSlider, QVBoxLayout, QCheckBox, QGridLayout
import cv2

from ChessNotation.Interface.UsefulFuncs import *


class ImagePresentationWidget(QWidget):
    signal_check_board = pyqtSignal(bool)
    signal_check_pieces = pyqtSignal(bool)
    signal_stop_video = pyqtSignal(bool)
    signal_find_board_grid = pyqtSignal()
    signal_rotate_board = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.big_font = QFont('Arial', 13)
        self.label_detect = QLabel()
        self.label_chess_2d = QLabel()
        self.pixmap_detect = QImage()
        self.pixmap_chess_2d = QImage()
        self.size_label = create_label('Размер:', self.big_font)
        self.size_spinbox = create_spinbox(0, 200, self.big_font, start_val=100)
        self.size_slider = create_slider((0, 200), 5, 100, -1, False)
        self.check_board = create_checkbox('Отображать сетку доски', True, self.big_font)
        self.check_pieces = create_checkbox('Отображать рамки фигур', True, self.big_font)
        self.btns_h = 45
        self.stop_btn = create_btn('Стоп', self.big_font, self.stop_video, min_h=self.btns_h)
        self.find_board_btn = create_btn('Найти сетку', self.big_font, self.find_board_grid, min_h=self.btns_h)
        self.rotate_btn = create_btn('Повернуть доску на 90', self.big_font, self.rotate_board, min_h=self.btns_h)

        self.is_video_stop = False
        self.standard_width: int = 736
        self.tab_ind: int = 0
        self.detect_size_p, self.chess_2d_size_p = 100, 100
        self.initUI()

    def initUI(self):
        main_v_lay = QVBoxLayout()
        slider_h_lay = QHBoxLayout()
        grid_lay = QGridLayout()

        self.label_detect.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_chess_2d.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tab = QTabWidget(self)
        tab.setFont(self.big_font)
        tab.addTab(self.label_detect, "Поиск")
        tab.addTab(self.label_chess_2d, "Отображение")
        self.size_spinbox.setEnabled(False)

        self.stop_btn.setEnabled(False)
        self.rotate_btn.setEnabled(False)
        self.find_board_btn.setEnabled(False)

        slider_h_lay.addWidget(self.size_label)
        slider_h_lay.addWidget(self.size_slider, 1)
        slider_h_lay.addWidget(self.size_spinbox)
        # slider_h_lay.addStretch(1)

        left_control_v_lay = QVBoxLayout()
        right_control_v_lay = QVBoxLayout()
        radio_v_lay = QVBoxLayout()
        down_control_h_lay = QHBoxLayout()
        control_h_lay = QHBoxLayout()

        radio_v_lay.addWidget(self.check_board)
        radio_v_lay.addWidget(self.check_pieces)
        down_control_h_lay. addLayout(radio_v_lay)
        down_control_h_lay.addSpacing(5)
        down_control_h_lay.addWidget(self.stop_btn, 1)
        left_control_v_lay.addSpacing(11)
        left_control_v_lay.addLayout(slider_h_lay)
        left_control_v_lay.addSpacing(5)
        left_control_v_lay.addLayout(down_control_h_lay)
        right_control_v_lay.addWidget(self.find_board_btn)
        right_control_v_lay.addWidget(self.rotate_btn)
        control_h_lay.addLayout(left_control_v_lay, 1)
        control_h_lay.addSpacing(5)
        control_h_lay.addLayout(right_control_v_lay, 1)

        main_v_lay.addWidget(tab)
        main_v_lay.addSpacing(5)
        main_v_lay.addLayout(control_h_lay)
        # main_v_lay.addLayout(slider_h_lay)
        # main_v_lay.addWidget(self.check_board)
        # main_v_lay.addWidget(self.check_pieces)
        # main_v_lay.addWidget(self.stop_btn)
        # main_v_lay.addWidget(find_board_btn)
        # main_v_lay.addWidget(rotate_btn)

        slider_h_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_v_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(main_v_lay)

        self.size_spinbox.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spinbox.setValue)
        self.size_slider.valueChanged.connect(self.change_size)
        tab.currentChanged.connect(self.tab_changed)
        self.check_board.stateChanged.connect(self.check_board_changed)
        self.check_pieces.stateChanged.connect(self.check_pieces_changed)

    def rotate_board(self):
        self.signal_rotate_board.emit()

    def find_board_grid(self):
        self.signal_find_board_grid.emit()

    def stop_video(self):
        if self.stop_btn.text() == 'Стоп':
            self.stop_btn.setText('Старт')
        else:
            self.stop_btn.setText('Стоп')

        self.is_video_stop = not self.is_video_stop
        self.signal_stop_video.emit(self.is_video_stop)

    def check_board_changed(self):
        self.signal_check_board.emit(self.check_board.isChecked())

    def check_pieces_changed(self):
        self.signal_check_pieces.emit(self.check_pieces.isChecked())

    def tab_changed(self, tab_ind):
        self.tab_ind = tab_ind
        new_val = 0
        if tab_ind == 0:
            new_val = self.detect_size_p
        elif tab_ind == 1:
            new_val = self.chess_2d_size_p
        self.size_slider.setValue(new_val)

    def change_size(self, val):
        new_size = int(self.standard_width * val / 100)
        if self.tab_ind == 0:
            self.detect_size_p = val
            self.pixmap_detect = self.pixmap_detect.scaledToWidth(new_size)
            self.label_detect.setPixmap(self.pixmap_detect)
        elif self.tab_ind == 1:
            self.chess_2d_size_p = val
            self.pixmap_chess_2d = self.pixmap_chess_2d.scaledToWidth(new_size)
            self.label_chess_2d.setPixmap(self.pixmap_chess_2d)

    def set_images(self, detect_img: np.ndarray, chess_img: np.ndarray):
        self.size_slider.setEnabled(True)
        self.size_spinbox.setEnabled(True)
        pixmap_detect = self._prepare_img(detect_img)
        size = int(self.standard_width * self.detect_size_p / 100)
        self.pixmap_detect = pixmap_detect.scaledToWidth(size)
        pixmap_chess_2d = self._prepare_img(chess_img)
        size = int(self.standard_width * self.chess_2d_size_p / 100)
        self.pixmap_chess_2d = pixmap_chess_2d.scaledToWidth(size)

        self.label_detect.setPixmap(self.pixmap_detect)
        self.label_chess_2d.setPixmap(self.pixmap_chess_2d)

    def _prepare_img(self, img: np.ndarray):
        height, width, _ = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_image)
