from bisect import bisect_right

import numpy as np

from lattice_points_ml.ConvNet import ConvNet
from ChessNotation.BoardDetecting.LatticeDetectFuncs import *


class LatticePoints:
    conv_model: ConvNet | None = None

    def __init__(self, img: np.ndarray, intersection_points: list[Point], lines: LinesGroups):

        self.lattice_points: list[Point] = []
        self.vert_lines: list[Line] = []
        self.horiz_lines: list[Line] = []
        self.border_points: list[Point] = []
        self.img = img
        self.color1: tuple[int, int, int] = (22, 173, 61)
        self.color2: tuple[int, int, int] = (180, 130, 70)

        self.get_lattice_points(intersection_points, lines)

    def get_lattice_points(self, intersection_points: list[Point], lines: LinesGroups):
        # draw_points(self.img, [intersection_points],[(22, 173, 61)], img_name='5_0', is_wait=False)
        tmp_lattice_points: list[Point] = []
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # draw_points(self.img, [intersection_points], [(0, 123, 255)], is_wait=False, img_name='zxcv')
        for point in intersection_points:
            x1, y1 = point.x - 10, point.y - 10
            edges = gray_img[y1:y1 + 21, x1:x1 + 21]
            # edges1 = get_point_neighborhood(gray_img, point)
            # edges2 = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # edges = get_point_neighborhood(gray_img, point)
            if type(edges) is np.ndarray:
                if edges.shape == (21, 21):
                    predicted_val = self.conv_model.predict_model(edges)
                    if predicted_val == 1:
                        tmp_lattice_points.append(point)
                    if predicted_val == 2:
                        self.border_points.append(point)
                    # cv2.imshow('imageq', edges)
                    # cv2.waitKey(0)
                    # cv2.imshow('imagew', edges1)
                    # cv2.imshow('imagee', edges2)
            # draw_points(self.img, [intersection_points, tmp_lattice_points, self.border_points],
            #             [(22, 173, 61), (180, 130, 70), (0, 123, 255)], img_name='5', is_wait=False)

        vert_lines, horiz_lines = get_all_lines(self.img, tmp_lattice_points, lines)
        self.vert_lines = exclude_the_wrong_lines(self.img, vert_lines, 0, 15, 10)
        self.horiz_lines = exclude_the_wrong_lines(self.img, horiz_lines, 1, 15, 10)
        self.lattice_points = tmp_lattice_points
        self.lattice_points = find_intersection_lattice_points(self.horiz_lines, self.vert_lines,
                                                               self.img.shape)
        # draw_lines(self.img, [vert_lines, horiz_lines], [(22, 173, 61), (180, 130, 70)], img_name='6', is_wait=False)
        # draw_points(self.img, [self.lattice_points], [(180, 130, 70)], img_name='7',
        #             is_wait=False)


    def shift_points_and_lines(self, x_shift: int, y_shift: int):
        self.lattice_points = self.shift_points(self.lattice_points, x_shift, y_shift)
        self.vert_lines = self.shift_lines(self.vert_lines, x_shift, y_shift)
        self.horiz_lines = self.shift_lines(self.horiz_lines, x_shift, y_shift)

    def shift_points(self, points: list[Point], x_shift: int, y_shift: int):
        for ind in range(len(points)):
            points[ind].x += x_shift
            points[ind].y += y_shift
        return points

    def shift_lines(self, lines: list[Line], x_shift: int, y_shift: int):
        for ind in range(len(lines)):
            x1, y1 = lines[ind].p1[0] + x_shift, lines[ind].p1[1] + y_shift
            x2, y2 = lines[ind].p2[0] + x_shift,lines[ind].p2[1] + y_shift
            lines[ind].set_by_raw_line(np.asarray([x1, y1, x2, y2]))
        return lines
