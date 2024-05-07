import numpy as np
from math import atan, pi


class Line:
    left_up_cord: tuple[int, int] = (0, 0)
    right_up_cord: tuple[int, int] = (0, 0)
    shape: tuple[int, int] = [300, 300]

    def __init__(self):
        self.norm = np.linalg.norm
        self.p1: np.array = np.array([])
        self.p2: np.array = np.array([])
        self.line_len: float = -1
        self.k, self.b = 0.0, 0.0
        self.angle: float = 0
        self.left_normal: float = 0.0
        self.right_normal: float = 0.0

    def set_by_raw_line(self, raw_line: np.array):
        x1, y1, x2, y2 = raw_line
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        self.k, self.b = get_line_solution([self.p1, self.p2])
        self.p1 = np.array(self.check_point_out(x1, y1))
        self.p2 = np.array(self.check_point_out(x2, y2))
        self.line_len = self.norm(self.p2 - self.p1)
        self.angle = atan(self.k) * 180 / pi
        self.find_normals()

    def check_point_out(self, x, y):
        if x < 0:
            x = 2
            y = round(self.k * x + self.b)
        elif x >= Line.shape[1]:
            x = Line.shape[1] - 2
            y = round(self.k * x + self.b)
        if y < 0:
            y = 2
            x = round((y - self.b)/self.k)
        elif x >= Line.shape[1]:
            y = Line.shape[0] - 2
            x = round((y - self.b)/self.k)
        return x, y

    def set_by_point_k(self, cord: tuple[int, int], k: float):
        self.k = k
        self.b = cord[1] - k * cord[0]

        x1, y1 = 2, round(self.k * 2 + self.b)
        x2, y2 = Line.shape[0] - 2, round(self.k * (Line.shape[0] - 2) + self.b)
        # print(f'k = {self.k}, b = {self.b}, cord = {cord}')
        # print(f'1) {x1}, {y1}; {x2}, {y2}')
        # print(f'1) {x1}, {y1}; {x2}, {y2}\n')
        self.p1 = np.array(self.check_point_out(x1, y1))
        self.p2 = np.array(self.check_point_out(x2, y2))

        self.angle = atan(self.k) * 180 / pi
        self.find_normals()

    def find_normals(self):
        p21 = self.p2 - self.p1
        p21_len = self.norm(p21)
        if p21_len == 0:
            p21_len = 0.0001
        self.left_normal = self.norm(np.cross(p21, self.p1 - Line.left_up_cord)) / p21_len
        self.right_normal = self.norm(np.cross(p21, self.p1 - Line.right_up_cord)) / p21_len

    def __str__(self):
        return f'({self.p1}, {self.p2}, k = {"{:.3f}".format(self.k)}, b = {"{:.3f}".format(self.b)}, angle = {"{:.3f}".format(self.angle)})'


def get_line_solution(points: list[np.array, np.array]) -> tuple[float, float]:
    x_coords, y_coords = zip(*points)
    a = np.vstack([x_coords, np.ones(len(x_coords))]).T
    k, b = np.linalg.lstsq(a, y_coords, rcond=None)[0]
    return k, b


def get_intersection_point(line1: Line, line2: Line) -> tuple[int, int]:
    if line1.k == line2.k:
        return None
    x: float = (line2.b - line1.b) / (line1.k - line2.k)
    y: float = line1.k * x + line1.b
    return round(x), round(y)


def get_lines_interseption(line1: Line, line2: Line) -> tuple[float, float]:
    max_x = 1000000
    angle_def = abs(line1.angle - line2.angle)
    if angle_def > 90:
        angle_def = 180 - angle_def
    if angle_def < 0.01:
        return max_x, line1.k * max_x + line1.b
    x = (line2.b - line1.b) / (line1.k - line2.k)
    y = line1.k * x + line1.b
    return x, y
