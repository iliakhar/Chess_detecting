import numpy as np

class Line:
    def __init__(self, raw_line: np.array):
        self.norm = np.linalg.norm

        x1, y1, x2, y2 = raw_line
        self.p1: np.array = np.array([x1, y1])
        self.p2: np.array = np.array([x2, y2])
        self.line_len: float = self.norm(self.p2 - self.p1)
        self.k, self.b = get_line_solution([self.p1, self.p2])

    def __str__(self):
        return f'({self.p1}, {self.p2}, len: {"{:.3f}".format(self.line_len)})'


def get_line_solution(points: list[np.array, np.array]) -> tuple[float, float]:
    x_coords, y_coords = zip(*points)
    a = np.vstack([x_coords, np.ones(len(x_coords))]).T
    k, b = np.linalg.lstsq(a, y_coords, rcond=None)[0]
    return k, b