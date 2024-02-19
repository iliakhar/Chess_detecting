import math
import random
import cv2
import numpy as np
from UsefulFunctions import *
from Line import *


class ChessBoardDetecting:

    def __init__(self):
        self.norm = np.linalg.norm
        self.img: np.ndarray = None
        self.gray_img: np.ndarray = None
        self.lines: list[Line] = []
        self.result_lines: list[list[Line]] = []
        self.colors_list: list[tuple[int, int, int]] = []
        self.blur_koef: int = 15
        self.density_koef: float = 0.08
        self.area: int = 0

        # self.set_image(img)

    def set_image(self, img: np.ndarray):
        # self.img = img
        if type(img) is not np.ndarray:
            return
        self.img = resizing(img=img, new_height=800)
        h, w = self.img.shape[:2]
        self.area = h * w
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_board(self) -> None:
        if type(self.img) is not np.ndarray:
            return
        self.lines = self.get_lines(self.gray_img)
        if len(self.lines) == 0:
            return
        grouped_lines, self.colors_list = self.find_collinear_lines(0.9)
        grouped_points: list[list[tuple[int, int]]] = self.get_lines_dots(grouped_lines)
        self.result_lines = self.group_lines_by_points(grouped_points)

        # draw_lines(self.img, grouped_lines, colors_list)
        # draw_points(self.img, grouped_points, colors_list)
        # draw_lines(self.img, self.result_lines, colors_list)

    def get_lines(self, img: np.ndarray) -> list[Line]:
        gaussian = cv2.GaussianBlur(img, (self.blur_koef, self.blur_koef), 0)
        edges = cv2.Canny(gaussian, 40, 40, apertureSize=3)
        raw_lines: np.ndarray = np.ndarray([])
        raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)
        lines: list[Line] = []
        # print(type(raw_lines))
        if type(raw_lines) is not np.ndarray:
            return []
        for raw_line in raw_lines:
            lines.append(Line(raw_line[0]))
        return lines

    def find_collinear_lines(self, p: float) -> tuple[list, list]:
        w = ((math.pi / 2) / self.area ** 0.25)
        t = p * w

        collinear_list: list = []
        used_lines: list = []
        colors_list: list = []
        for line_ind, line in enumerate(self.lines):
            if line_ind not in used_lines:
                # collinear_list.append([line_ind])
                collinear_list.append([line])
                used_lines.append(line_ind)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # print(color)
                colors_list.append(color)
                for next_line_ind, next_line in enumerate(self.lines[line_ind + 1:]):
                    if next_line_ind + line_ind + 1 not in used_lines:
                        delta: float = (line.line_len + next_line.line_len) * t
                        v1 = line.p2 - line.p1
                        v2 = next_line.p2 - next_line.p1
                        mean_len: float = 0
                        mean_len += self.norm(np.cross(v1, line.p1 - next_line.p1)) / self.norm(v1)
                        mean_len += self.norm(np.cross(v1, line.p1 - next_line.p2)) / self.norm(v1)
                        mean_len += self.norm(np.cross(v2, next_line.p1 - line.p1)) / self.norm(v2)
                        mean_len += self.norm(np.cross(v2, next_line.p1 - line.p2)) / self.norm(v2)
                        mean_len /= 4
                        if mean_len == 0:
                            mean_len = 0.001

                        if next_line.line_len / mean_len > delta and line.line_len / mean_len > delta:
                            # collinear_list[-1].append(next_line_ind + line_ind + 1)
                            collinear_list[-1].append(next_line)
                            used_lines.append(next_line_ind + line_ind + 1)
                            # colors_list[next_line_ind + line_ind + 1] = color

        return collinear_list, colors_list

    def get_lines_dots(self, grouped_lines: list[list[Line]]) -> list[list[tuple[int, int]]]:
        pixels_per_point: int = round(self.area / self.density_koef / 640000)
        grouped_points: list[list[tuple[int, int]]] = []
        for line_group in grouped_lines:
            grouped_points.append([])
            for line in line_group:
                # print(f'{line}: ', end='')
                line_len = line.line_len
                point_count: float = line_len / pixels_per_point
                # print(line)
                x_per_point: int = round((line.p2[0] - line.p1[0]) / point_count)
                if x_per_point == 0:
                    x_per_point = 1
                x_cords: list[int] = [cord for cord in range(line.p1[0], line.p2[0], x_per_point)]
                # print(f'{x_cords}, {k}, {b}')
                for x_cord in x_cords:
                    y_cord = int(line.k * x_cord + line.b)
                    grouped_points[-1].append((x_cord, y_cord))
            # print('--------------------------------------------------------------------------------------------------')
            # print(len(grouped_points), len(grouped_lines))
        return grouped_points

    def group_lines_by_points(self, grouped_points: list[list[tuple[int, int]]]) -> list[list[Line]]:
        lines: list[list[Line]] = []
        for group in grouped_points:
            if len(group) <= 1:
                continue
            lines.append([])
            avg_x: float = 0
            avg_y: float = 0
            min_x: float = math.inf
            max_x: float = 0
            for point in group:
                avg_x += point[0]
                avg_y += point[1]
                if point[0] < min_x:
                    min_x = point[0]
                if point[0] > max_x:
                    max_x = point[0]

            avg_x /= len(group)
            avg_y /= len(group)
            # print(f'avg_x: {avg_x}, avg_y: {avg_y}')
            delta_xy: float = 0
            delta_x_square: float = 0
            for point in group:
                # print(point, avg_x, avg_y)
                delta_xy += (point[0] - avg_x)*(point[1]-avg_y)
                delta_x_square += (point[0] - avg_x)**2

            if delta_x_square == 0:
                delta_x_square = 0.00001
            k: float = delta_xy/delta_x_square
            b: float = avg_y - k*avg_x
            x_list: list[int] = [round(min_x), round(max_x)]
            y_list: list[int] = [round(k*min_x + b), round(k*max_x+b)]
            lines[-1].append(Line(np.array([x_list[0], y_list[0], x_list[1], y_list[1]])))
        return lines
