import math
import os
import random
import cv2
import numpy as np
from UsefulFunctions import *
from LinesGroups import *
from lattice_points_ml.ConvNet import ConvNet

class ChessBoardDetecting:

    def __init__(self):
        self.img: np.ndarray | None = None
        self.gray_img: np.ndarray | None = None
        # self.mono_image: np.ndarray | None = None
        self.blur_koef: int = 1
        self.density_koef: float = 0.08

        self.conv_model: ConvNet = ConvNet()
        self.conv_model = self.conv_model.to(self.conv_model.device)
        self.conv_model.load_model(os.getcwd() + '\\lattice_points_ml\\model\\model_1.pt')

        self.lines: LinesGroups = LinesGroups(self.blur_koef, self.density_koef)
        self.intersection_points: list[tuple[int, int]] = []
        self.lattice_points: list[tuple[int, int]] = []

        self.board_center: tuple[int, int] = (0, 0)

    def set_image(self, img: np.ndarray):
        # self.img = img
        if type(img) is not np.ndarray:
            return
        self.img = resizing(img=img, new_height=350)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # _, self.mono_image = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)

    def detect_board(self) -> None:
        if type(self.img) is not np.ndarray:
            return
        self.lines = LinesGroups(self.blur_koef, self.density_koef)
        self.lines.find_lines(self.img)
        # _, self.img = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_OTSU)
        self.intersection_points = find_intersection_points(self.lines.result_lines)
        self.lattice_points = self.get_lattice_points()
        self.board_center = self.get_board_center()
        # print(self.intersection_points)

    def get_lattice_points(self) -> list[tuple[int, int]]:
        lattice_points: list[tuple[int, int]] = []
        for point in self.intersection_points:
            x1, y1 = point[0] - 10, point[1] - 10
            sub_img = self.gray_img[y1:y1 + 21, x1:x1+21]
            ret, thresh1 = cv2.threshold(sub_img, 120, 255, cv2.THRESH_BINARY +
                                         cv2.THRESH_OTSU)
            edges = cv2.Canny(thresh1, 20, 30, apertureSize=3)
            if type(edges) == np.ndarray:
                if edges.shape == (21, 21):
                    if self.conv_model.predict_model(edges) == 1:
                        lattice_points.append(point)
                        # cv2.imshow('image', edges)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
        return lattice_points

    def get_board_center(self):
        if len(self.lattice_points) == 0:
            return 5, 5

        x, y = 0.0, 0.0
        for point in self.lattice_points:
            x+=point[0]
            y+=point[1]
        return int(x / len(self.lattice_points)), int(y / len(self.lattice_points))

    def show_lines(self, is_wait: bool = True, color: tuple[int, int, int] = (22, 173, 61)):
        # clr = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(self.lines.result_lines))]
        draw_lines(self.img, [self.lines.result_lines], [color], is_wait)
        # draw_lines(self.img, [self.lines.result_lines], clr, is_wait)

    def show_all_points(self, is_wait: bool = True, img_name: str = 'image1', color: tuple[int, int, int] = (22, 173, 61)):
        draw_points(self.img, [self.intersection_points], [color], img_name, is_wait)

    def show_lattice_points(self, is_wait: bool = True, img_name: str = 'image2', color: tuple[int, int, int] = (180, 130, 70)):
        draw_points(self.img, [self.lattice_points], [color], img_name, is_wait)

    def show_grupped_points(self, is_wait: bool = True, img_name: str = 'image3'):
        color1: tuple[int, int, int] = (22, 173, 61)
        color2: tuple[int, int, int] = (180, 130, 70)
        color3: tuple[int, int, int] = (0, 0, 139)
        draw_points(self.img, [self.intersection_points, self.lattice_points, [self.board_center]], [color1, color2, color3], img_name, is_wait)
