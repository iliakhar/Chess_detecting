import math
import random
import cv2
import numpy as np
from UsefulFunctions import *
from LinesGroups import *


class ChessBoardDetecting:

    def __init__(self):
        self.img: np.ndarray = None
        self.gray_img: np.ndarray = None
        self.blur_koef: int = 21
        self.density_koef: float = 0.08
        self.lines: LinesGroups = LinesGroups(self.blur_koef, self.density_koef)
        self.intersection_points: list[tuple[int, int]] = []

    def set_image(self, img: np.ndarray):
        # self.img = img
        if type(img) is not np.ndarray:
            return
        self.img = resizing(img=img, new_height=800)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_board(self) -> None:
        if type(self.img) is not np.ndarray:
            return
        self.lines.find_lines(self.img)
        self.intersection_points = find_intersection_points(self.lines.result_lines, self.img)
        # draw_lines(self.img, grouped_lines, colors_list)
        # draw_points(self.img, grouped_points, colors_list)
        # draw_lines(self.img, self.result_lines, colors_list)

    def show_lines(self, is_wait: bool = True, color: tuple[int, int, int] = (22, 173, 61)):
        draw_lines(self.img, [self.lines.result_lines], [color], is_wait)

    def show_points(self, is_wait: bool = True, color: tuple[int, int, int] = (22, 173, 61)):
        draw_points(self.img, [self.intersection_points], [color], is_wait)
