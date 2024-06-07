import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QImage
from PyQt6.QtWidgets import QWidget, QLineEdit, QHBoxLayout, QTabWidget, QSlider, QVBoxLayout
import cv2

from ChessNotation.Interface.UsefulFuncs import *


def resizing(img, new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]

    if new_width is None and new_height is None:
        return img
    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)
    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))
    return cv2.resize(img, dimension, interpolation=interp)


class ImagePresentationWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.big_font = QFont('Arial', 13)
        self.label_detect = QLabel()
        self.label_chess_2d = QLabel()
        self.pixmap_detect = QImage()
        self.pixmap_chess_2d = QImage()
        self.size_label = create_label('Размер:', self.big_font)
        self.size_spinbox = create_spinbox(0, 200, self.big_font)
        self.size_slider = create_slider((0, 200), 5, 100, 300, False)
        self.standard_width: int = 736
        self.tab_ind: int = 0
        self.detect_size_p, self.chess_2d_size_p = 100, 100
        self.initUI()

    def initUI(self):
        main_v_lay = QVBoxLayout()
        slider_h_lay = QHBoxLayout()

        self.label_detect.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_chess_2d.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tab = QTabWidget(self)
        tab.setFont(self.big_font)
        tab.addTab(self.label_detect, "Поиск")
        tab.addTab(self.label_chess_2d, "Отображение")
        self.size_spinbox.setEnabled(False)

        slider_h_lay.addWidget(self.size_label)
        slider_h_lay.addWidget(self.size_slider)
        slider_h_lay.addWidget(self.size_spinbox)
        slider_h_lay.addStretch(1)
        main_v_lay.addWidget(tab)
        main_v_lay.addLayout(slider_h_lay)

        slider_h_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_v_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(main_v_lay)

        self.size_spinbox.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spinbox.setValue)
        self.size_slider.valueChanged.connect(self.change_size)
        tab.currentChanged.connect(self.tab_changed)

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
