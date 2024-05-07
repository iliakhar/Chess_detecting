from random import randint
from joblib import Parallel, delayed

from UsefulFunctions import *
from Line import *


class LinesGroups:

    def __init__(self, blur_koef: int, density_koef: float):
        self.norm = np.linalg.norm
        self.lines: list[Line] = []
        self.lines_inds_for_skipping: set = set()
        self.result_lines: list[Line] = []
        self.colors_list: list[tuple[int, int, int]] = []
        self.blur_koef: int = blur_koef
        self.density_koef: float = density_koef
        self.area: int = 0
        self.img: np.ndarray = np.ndarray([])

    def find_lines(self, img: np.ndarray):
        self.img = img
        Line.right_up_cord = (0, img.shape[1])
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # gray_img = self.img
        self.lines = self.get_lines(gray_img)
        if len(self.lines) == 0:
            return
        # draw_lines(self.img, [self.lines], [(30, 120, 60)])
        # self.remove_unnecessary_lines(0.999)
        h, w = img.shape[:2]
        self.area = h * w
        grouped_lines, self.colors_list = self.find_collinear_lines(0.8)
        grouped_points: list[list[tuple[int, int]]] = self.get_lines_dots(grouped_lines)
        self.result_lines = self.group_lines_by_points(grouped_points)
        # self.remove_unnecessary_lines(self.result_lines, 0.8)

        # draw_lines(self.img, [self.lines], self.colors_list)
        # draw_points(self.img, grouped_points, self.colors_list)
        # draw_lines(self.img, grouped_lines, self.colors_list)
        # draw_lines(self.img, [self.result_lines], [(22, 173, 61)])

    def get_lines(self, img: np.ndarray) -> list[Line]:
        gaussian = cv2.GaussianBlur(img, (self.blur_koef, self.blur_koef), 0)
        edges = cv2.Canny(gaussian, 20, 30, apertureSize=3)  # 20 30
        # raw_lines: np.ndarray = np.ndarray([])
        raw_lines: np.ndarray = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=7, maxLineGap=10)
        lines: list[Line] = []
        # print(type(raw_lines))
        if type(raw_lines) is not np.ndarray:
            return []
        for raw_line in raw_lines:
            lines.append(Line())
            lines[-1].set_by_raw_line(raw_line[0])
        lines.sort(key=lambda x: x.angle)
        return lines

    def find_collinear_lines(self, p: float) -> tuple[list, list]:
        w = ((math.pi / 2) / self.area ** 0.25)
        lim_angle = 4
        collinear_list: list = []
        used_lines: list = []
        colors_list: list = []
        # cnt = 0
        for line_ind, line in enumerate(self.lines):
            left_border = bisect_left(self.lines, line.angle - lim_angle, key=lambda x: x.angle)
            right_border = bisect_right(self.lines, line.angle + lim_angle, key=lambda x: x.angle)
            if line_ind not in used_lines:
                collinear_list.append([line])
                used_lines.append(line_ind)
                # -----------------------------------------------------------------------------------------------CHANGE
                normal_delta: float = 8
                left_normal_min, left_normal_max = line.left_normal - normal_delta, line.left_normal + normal_delta
                right_normal_min, right_normal_max = line.right_normal - normal_delta, line.right_normal + normal_delta
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                colors_list.append(color)
                for next_line_ind, next_line in enumerate(self.lines[left_border:right_border + 1]):
                    if next_line_ind + left_border not in used_lines:
                        if left_normal_min <= next_line.left_normal <= left_normal_max and \
                                right_normal_min <= next_line.right_normal <= right_normal_max:
                            collinear_list[-1].append(next_line)
                            used_lines.append(next_line_ind + left_border)

        return collinear_list, colors_list

    def get_lines_dots(self, grouped_lines: list[list[Line]]) -> list[list[tuple[int, int]]]:

        pixels_per_point: int = round(self.area / self.density_koef / 640000)

        def get_line_dots_from_group(line_group: list[Line]):
            group_of_points: list[tuple[int, int]] = []
            for line in line_group:
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
                    group_of_points.append((x_cord, y_cord))
            return group_of_points

        grouped_points = Parallel(n_jobs=1)(
            delayed(get_line_dots_from_group)(line_group) for line_group in grouped_lines)

        return grouped_points

    def group_lines_by_points(self, grouped_points: list[list[tuple[int, int]]]) -> list[Line]:
        lines: list[Line] = []
        for group in grouped_points:
            if len(group) <= 1:
                continue
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
            delta_xy: float = 0
            delta_x_square: float = 0
            for point in group:
                delta_xy += (point[0] - avg_x) * (point[1] - avg_y)
                delta_x_square += (point[0] - avg_x) ** 2

            if delta_x_square == 0:
                delta_x_square = 0.00001
            k: float = delta_xy / delta_x_square
            b: float = avg_y - k * avg_x
            x_list: list[int] = [round(min_x), round(max_x)]
            y_list: list[int] = [round(k * min_x + b), round(k * max_x + b)]
            lines.append(Line())
            lines[-1].set_by_raw_line(np.array([x_list[0], y_list[0], x_list[1], y_list[1]]))
        return lines
