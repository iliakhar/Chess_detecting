from random import randint
from joblib import Parallel, delayed

from UsefulFunctions import *
from Line import *


class LinesGroups:

    def __init__(self, blur_koef: int, density_p: float):
        self.norm = np.linalg.norm
        self.lines: list[Line] = []
        self.lines_inds_for_skipping: set = set()
        self.result_lines: list[Line] = []
        self.colors_list: list[tuple[int, int, int]] = []
        self.blur_koef: int = blur_koef
        self.density_p: float = density_p
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
        grouped_lines, self.colors_list = self.find_collinear_lines()
        grouped_points: list[list[tuple[int, int]]] = self.get_lines_dots(grouped_lines)
        self.result_lines = self.group_lines_by_points(grouped_points)

        # points: list[list[Point]] = []
        # for group in grouped_points:
        #     points.append([])
        #     for point in group:
        #         points[-1].append(Point((point[0], point[1])))
        #
        # draw_lines(self.img, [self.lines], [(22, 173, 61)])
        # draw_lines(self.img, grouped_lines, self.colors_list)
        # draw_points(self.img, points, self.colors_list)
        # draw_lines(self.img, [self.result_lines], [(22, 173, 61)])

    def get_lines(self, img: np.ndarray) -> list[Line]:
        gaussian = cv2.GaussianBlur(img, (self.blur_koef, self.blur_koef), 0)
        edges = cv2.Canny(gaussian, 150, 180, apertureSize=3)  # 90, 120 | 120, 160 | 180, 370 | 180, 220
        # cv2.imshow('i', edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        raw_lines: np.ndarray = cv2.HoughLinesP(edges, 1, np.pi / 180, 70,
                                                minLineLength=10, maxLineGap=15)
        lines: list[Line] = []
        if type(raw_lines) is not np.ndarray:
            return []
        for raw_line in raw_lines:
            lines.append(Line())
            lines[-1].set_by_raw_line(raw_line[0])
        lines.sort(key=lambda x: x.angle)
        # color = (22, 173, 61)
        # for l in lines:
        #     print(l)
        #     draw_lines(self.img, [[l]], [color])

        return lines

    def find_collinear_lines(self) -> tuple[list, list]:
        lim_angle = 3
        standard_area = 640800
        normal_delta: float = 18 * (self.area / standard_area) ** 0.5
        collinear_list: list = []
        used_lines: list = []
        colors_list: list = []
        for line_ind, line in enumerate(self.lines):
            left_border = bisect_left(self.lines, line.angle - lim_angle, key=lambda x: x.angle)
            right_border = bisect_right(self.lines, line.angle + lim_angle, key=lambda x: x.angle)
            if line_ind not in used_lines:
                collinear_list.append([line])
                used_lines.append(line_ind)
                left_normal_min, left_normal_max = line.left_normal - normal_delta, line.left_normal + normal_delta
                right_normal_min, right_normal_max = line.right_normal - normal_delta, line.right_normal + normal_delta
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                colors_list.append(color)
                for next_line_ind, next_line in enumerate(self.lines[left_border:right_border]):
                    if next_line_ind + left_border not in used_lines:
                        if left_normal_min <= next_line.left_normal <= left_normal_max and \
                                right_normal_min <= next_line.right_normal <= right_normal_max:
                            collinear_list[-1].append(next_line)
                            used_lines.append(next_line_ind + left_border)

        return collinear_list, colors_list

    def get_lines_dots(self, grouped_lines: list[list[Line]]) -> list[list[tuple[int, int]]]:
        standard_area = 640800
        standard_pixel_per_point = 10
        pixels_per_point = standard_pixel_per_point * (self.area / standard_area) ** 0.5
        pixels_per_point /= self.density_p
        # pixels_per_point: int = round((self.area / 640000) * (self.density_p) * 100)
        grouped_points = []
        # color = (22, 173, 61)
        for line_group in grouped_lines:
            # for l in line_group:
            #     print(l)
            # print()
            # draw_lines(self.img, [line_group], [color])
            group_of_points: list[tuple[int, int]] = []
            for line in line_group:
                line_len = line.line_len
                point_count: float = line_len / pixels_per_point
                axis = 0
                if -90 <= line.angle < -45 or 45 < line.angle <= 90:
                    axis = 1
                axis_px_per_point: int = round((line.p2[axis] - line.p1[axis]) / point_count)
                if axis_px_per_point == 0:
                    axis_px_per_point = 1
                axis_cords: list[int] = [cord for cord in range(line.p1[axis], line.p2[axis], axis_px_per_point)]
                if axis == 0:
                    for x_cord in axis_cords:
                        y_cord = int(line.k * x_cord + line.b)
                        group_of_points.append((x_cord, y_cord))
                else:
                    for y_cord in axis_cords:
                        x_cord = int((y_cord - line.b) / line.k)
                        group_of_points.append((x_cord, y_cord))
            grouped_points.append(group_of_points)

        return grouped_points

    def group_lines_by_points(self, grouped_points: list[list[tuple[int, int]]]) -> list[Line]:
        lines: list[Line] = []
        min_line_len = (self.area**0.5)*0.03
        for group in grouped_points:
            # x_nd = np.array([x for x, _ in group])
            # y_nd = np.array([y for _, y in group])
            # if x_nd.size < 2:
            #     continue
            # k, b = np.polyfit(x_nd, y_nd, 1)
            min(group, key=lambda x: x[0])
            k, b = self.get_k_b_by_points(group)
            left_x, right_x = min(group, key=lambda x: x[0])[0], max(group, key=lambda x: x[0])[0]
            left_y, right_y = k * left_x + b, k * right_x + b
            line = Line()
            line.set_by_raw_line(np.array([round(left_x), round(left_y), round(right_x), round(right_y)]))
            # points = [Point((round(x), round(y))) for x, y in group]
            # print(group)
            # print(f'len group: {len(group)}, len line: {line.line_len}, min line: {min_line_len}')
            # draw_points(self.img, [points], [(22, 173, 61)], is_wait=False)
            # draw_lines(self.img, [[line]], [(22, 173, 61)])
            if len(group) <= 1:
                if line.line_len < min_line_len:
                    continue
            lines.append(line)
        # draw_lines(self.img, [lines], [(22, 173, 61)])
        return lines

    def get_k_b_by_points(self, points_group):
        avg_x, avg_y = 0.0, 0.0
        min_x, max_x = math.inf, 0
        for point in points_group:
            avg_x += point[0]
            avg_y += point[1]
            if point[0] < min_x:
                min_x = point[0]
            if point[0] > max_x:
                max_x = point[0]

        avg_x /= len(points_group)
        avg_y /= len(points_group)
        delta_xy: float = 0
        delta_x_square: float = 0
        for point in points_group:
            delta_xy += (point[0] - avg_x) * (point[1] - avg_y)
            delta_x_square += (point[0] - avg_x) ** 2

        if delta_x_square == 0:
            delta_x_square = 0.00001
        k: float = delta_xy / delta_x_square
        b: float = avg_y - k * avg_x
        return k, b
