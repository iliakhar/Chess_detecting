from PyQt6.QtCore import Qt
from PyQt6.QtGui import (QFont, QPixmap, QIcon)
from PyQt6.QtWidgets import QPushButton, QLabel, QSpinBox, QFrame, QSlider


def create_btn(title: str, font: QFont, func, min_h: int = 35, min_w: int = 100):
    btn = QPushButton(title)
    btn.setFont(font)
    btn.setMinimumHeight(min_h)
    btn.setMinimumWidth(min_w)
    btn.clicked.connect(func)
    return btn


def create_label(title: str, font: QFont):
    lbl = QLabel(title)
    lbl.setFont(font)
    return lbl


def create_spinbox(min_val: int, max_val: int, font: QFont, min_h: int = 32, min_w: int = 110, start_val: int = 1):
    spinbox = QSpinBox()
    spinbox.setMinimum(min_val)
    spinbox.setMaximum(max_val)
    spinbox.setFont(font)
    spinbox.setMinimumHeight(min_h)
    spinbox.setMinimumWidth(min_w)
    spinbox.setValue(start_val)
    return spinbox


def create_frame(layout):
    frame = QFrame()
    # QFrame.Shape.StyledPanel
    frame.setFrameShape(QFrame.Shape.Panel)
    frame.setLayout(layout)
    frame.setWindowIconText('afsdfasf')
    return frame


def create_slider(range_vals: tuple[int, int], step: int, val: int, width: int, enable: bool):
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setEnabled(enable)
    slider.setRange(range_vals[0], range_vals[1])
    slider.setValue(val)
    slider.setSingleStep(step)
    slider.setFixedWidth(width)
    return slider
