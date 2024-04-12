import numpy as np

class Line:
    left_up_cord: tuple[int, int] = (0, 0)
    right_up_cord: tuple[int, int] = (0, 0)
    def __init__(self, raw_line: np.array):
        self.norm = np.linalg.norm

        x1, y1, x2, y2 = raw_line
        self.p1: np.array = np.array([x1, y1])
        self.p2: np.array = np.array([x2, y2])
        self.line_len: float = self.norm(self.p2 - self.p1)
        self.k, self.b = get_line_solution([self.p1, self.p2])
        p21 = self.p2 - self.p1
        p21_len = self.norm(p21)
        if p21_len == 0:
            p21_len = 0.0001
        self.left_normal = self.norm(np.cross(p21, self.p1 - Line.left_up_cord)) / p21_len
        self.right_normal = self.norm(np.cross(p21, self.p1 - Line.right_up_cord)) / p21_len
    def __str__(self):
        return f'({self.p1}, {self.p2}, k = {"{:.3f}".format(self.k)}, b = {"{:.3f}".format(self.b)})'

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
