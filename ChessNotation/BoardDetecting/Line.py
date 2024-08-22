import numpy as np
from math import atan, pi


class Line:
    left_up_cord: tuple[int, int] = (0, 0)
    right_up_cord: tuple[int, int] = (0, 0)
    shape: tuple[int, int] = [300, 300]

    def __init__(self, is_img_size_matter: bool = True):
        self.is_img_size_matter = is_img_size_matter
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
        if x1 == x2:
            x2 += 1
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        self.k, self.b = get_line_solution([self.p1, self.p2])
        self.angle = atan(self.k) * 180 / pi
        if self.is_img_size_matter:
            self.check_point_out()
        self.line_len = self.norm(self.p2 - self.p1)
        self.find_normals()

    def set_is_img_size_matter(self, is_img_size_matter: bool):
        if is_img_size_matter:
            self.is_img_size_matter = True
            self.check_point_out()
        else:
            self.is_img_size_matter = False

    def check_point_out(self):#########################################
        # if len(self.p1) != 2 and len(self.p2)!= 2:
        #     return
        if not (0 <= self.p1[0] < self.shape[1] and 0 <= self.p1[1] < self.shape[0]) or \
            not (0 <= self.p2[0] < self.shape[1] and 0 <= self.p2[1] < self.shape[0]):
            points = [self.p1, self.p2]
            if self.k == 0:
                self.k = 0.00001
            inter_border_points = [round(self.b), round(self.k*self.shape[1]+self.b),
                                   round(-self.b/self.k), round((self.shape[0]-self.b)/self.k)]
            if 0<inter_border_points[0]<self.shape[0]:
                points.append(np.array([0, inter_border_points[0]]))
            if 0 < inter_border_points[1] < self.shape[0]:
                points.append(np.array([self.shape[1], inter_border_points[1]]))
            if 0<inter_border_points[2]<self.shape[1]:
                points.append(np.array([inter_border_points[2], 0]))
            if 0 < inter_border_points[3] < self.shape[1]:
                points.append(np.array([inter_border_points[3], self.shape[1]]))
            if abs(self.k) < 1:
                points = sorted(points, key=lambda p: p[0])
            else:
                points = sorted(points, key=lambda p: p[1])
            if len(points) == 4:
                self.p1, self.p2 = points[1], points[2]
            elif len(points) == 2:
                self.p1, self.p2 = points[0], points[1]

    def set_by_point_k(self, cord: tuple[int, int], k: float):
        self.k = k
        self.b = cord[1] - k * cord[0]

        x1, y1 = 2, round(self.k * 2 + self.b)
        x2, y2 = Line.shape[0] - 2, round(self.k * (Line.shape[0] - 2) + self.b)

        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        # print(f'k = {self.k}, b = {self.b}, cord = {cord}')
        # print(f'1) {x1}, {y1}; {x2}, {y2}')
        # print(f'1) {x1}, {y1}; {x2}, {y2}\n')
        self.check_point_out()

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


def get_intersection_point(line1: Line, line2: Line) -> tuple[int, int] | None:
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
