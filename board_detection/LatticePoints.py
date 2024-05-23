from bisect import bisect_right

import numpy as np

from lattice_points_ml.ConvNet import ConvNet
from LatticeDetectFuncs import *


class LatticePoints:
    conv_model: ConvNet | None = None
    number_of_frame: int = 15
    frame_ind: int = 0
    board_center_list: list[Point] = []
    const_board_center: Point = Point((5, 5))
    borders_info: list[list] = [[], [], [], []]
    const_borders: list[Line] = []

    def __init__(self, img: np.ndarray, intersection_points: list[Point], lines: LinesGroups):

        self.lattice_points: list[Point] = []
        self.border_points: list[Point] = []
        self.bp: list[Point] = []
        self.vert_lines: list[Line] = []
        self.horiz_lines: list[Line] = []
        self.borders: list[Line] = []
        self.img = img
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.color1: tuple[int, int, int] = (22, 173, 61)
        self.color2: tuple[int, int, int] = (180, 130, 70)

        self.get_lattice_points(intersection_points, lines)
        self.get_board_center()
        self.get_border_info()
        LatticePoints.frame_ind += 1
        LatticePoints.frame_ind %= LatticePoints.number_of_frame

    def get_border_info(self):
        if len(self.lattice_points) == 0:
            return
        if LatticePoints.borders_info[0] is None or LatticePoints.borders_info[2] is None:
            return
        if LatticePoints.frame_ind == LatticePoints.number_of_frame - 1:
            border_info = []
            for border in LatticePoints.borders_info:
                x, y, k, ratio = 0, 0, 0, 0
                for ind, vals in enumerate(border):
                    if vals is None:
                        vals = [1, 1, 1, 1]
                    x, y, k, ratio = x+vals[0], y+vals[1], k+vals[2], ratio+vals[3]
                #     print(f'[ {vals[0]}, {vals[1]}, {"{:.2f}".format(vals[2])} ], ', end='')
                # print()
                x, y, k, ratio = round(x / len(border)), round(y / len(border)), k / len(border), ratio / len(border)
                border_info.append((x, y, k, ratio))
            LatticePoints.const_borders, _ = self.get_border_lines_points(border_info, LatticePoints.const_board_center)
            ratio_list: list[float] = [info[3] if ind % 2 == 1 else 1 / info[3] for ind, info in enumerate(border_info)]
            LatticePoints.const_borders += self.get_all_grid(LatticePoints.const_borders, ratio_list)
            if LatticePoints.const_borders is None:
                LatticePoints.const_borders = []
            # LatticePoints.const_borders += self.get_all_grid(LatticePoints.const_borders)
            LatticePoints.borders_info = [[], [], [], []]
            # print()
        if len(self.vert_lines) == 0 or len(self.horiz_lines) == 0:
            return
        center = LatticePoints.board_center_list[-1]
        line_ind = int(len(self.horiz_lines) / 2)
        line_points = get_line_points(self.lattice_points, line_ind, 0)
        l_points = line_points[::-1]
        r_points = line_points
        LatticePoints.borders_info[0].append(
            get_border_cord_and_angle(self.img, l_points, self.vert_lines, 0))
        LatticePoints.borders_info[1].append(
            get_border_cord_and_angle(self.img, r_points, self.vert_lines, 0))

        line_ind = int(len(self.vert_lines) / 2)
        line_points = get_line_points(self.lattice_points, line_ind, 1)
        u_points = line_points[::-1]
        d_points = line_points
        LatticePoints.borders_info[2].append(
            get_border_cord_and_angle(self.img, u_points, self.horiz_lines, 1))
        LatticePoints.borders_info[3].append(
            get_border_cord_and_angle(self.img, d_points, self.horiz_lines, 1))
        border_info = []
        for border in LatticePoints.borders_info:
            # print("{:.2f}".format(border[-1][2]), end=' ')
            border_info.append(border[-1])
        # print()
        self.borders, self.bp = self.get_border_lines_points(border_info, center)
        if self.borders is None:
            self.borders = []
        ratio_list: list[float] = []
        for ind, info in enumerate(border_info):
            if info is None:
                info = (1, 1, 1, 1)
            if ind % 2 == 1:
                ratio_list.append(info[3])
            else:
                ratio_list.append(1/info[3])
            # ratio_list = [info[3] if ind % 2 == 1 else 1/info[3] for ind, info in enumerate(border_info)]
        self.borders += self.get_all_grid(self.borders, ratio_list)
        # print()

    def get_all_grid(self, borders: list[Line], ratio_list: list[float]):
        # print(ratio_list)
        if len(borders) == 0:
            return []
        lines: list[Line] = []
        points: list = []
        connected_lines = [(0, 1), (2, 3)]
        for ind, line in enumerate(borders):
            start = min(line.p1[ind//2], line.p2[ind//2])
            ratio_power_sum = 1
            for power in range(1, 8):
                ratio_power_sum += ratio_list[ind] ** (-power)
            step = line.line_len / ratio_power_sum
            points.append([])
            if ind in connected_lines[0]:
                step = abs(step * math.cos(line.angle*pi/180))
                for i in range(1, 8):
                    x = start+step
                    y = line.k * x + line.b
                    start = x
                    step /= ratio_list[ind]
                    points[-1].append([round(x), round(y)])
            else:
                step = abs(step * math.sin(line.angle*pi/180))
                for i in range(1, 8):
                    y = start + step
                    x = (y - line.b) / line.k
                    start = y
                    step /= ratio_list[ind]
                    points[-1].append([round(x), round(y)])

        for connect in connected_lines:
            for ind in range(len(points[connect[0]])):
                x1, y1 = points[connect[0]][ind]
                x2, y2 = points[connect[1]][ind]
                lines.append(Line())
                lines[-1].set_by_raw_line(np.asarray([x1, y1, x2, y2]))
        return lines

    def get_border_lines_points(self, border_info: list[tuple], center: Point):
        borders = []
        border_points_center = []
        line_order_list = [(0, 1), (2, 3), (0, 2), (1, 3)]
        if border_info[0] is None or border_info[2] is None:
            return [], []
        for border in border_info:
            cord = list(border[:2])
            cord[0] += center.x
            cord[1] += center.y
            border_points_center.append(Point(cord))
            borders.append(Line())
            borders[-1].set_by_point_k(cord, border[2])
        border_points = find_intersection_lattice_points(self.img, borders[2:], borders[:2], self.img.shape, True)
        if len(border_points) != 4:
            return borders, border_points_center
        for ind, pnts_ind in enumerate(line_order_list):
            points: np.ndarray = np.asarray(
                [border_points[pnts_ind[0]].x, border_points[pnts_ind[0]].y, border_points[pnts_ind[1]].x,
                 border_points[pnts_ind[1]].y])
            borders[ind].set_by_raw_line(points)
        return borders, border_points_center

    def get_lattice_points(self, intersection_points: list[Point], lines: LinesGroups):
        tmp_lattice_points: list[Point] = []

        for point in intersection_points:
            x1, y1 = point.x - 10, point.y - 10
            edges = self.gray_img[y1:y1 + 21, x1:x1 + 21]
            if type(edges) is np.ndarray:
                if edges.shape == (21, 21):
                    predicted_val = self.conv_model.predict_model(edges)
                    if predicted_val == 1:
                        tmp_lattice_points.append(point)
                    if predicted_val == 2:
                        self.border_points.append(point)
        # draw_points(self.img, [tmp_lattice_points], [(180, 130, 70)])
        # draw_lines(self.img, [lines.result_lines], [(22, 173, 61), (180, 130, 70)])
        vert_lines_with_count, horiz_lines_with_count = get_lines_with_count(self.img, tmp_lattice_points, lines)
        self.vert_lines = exclude_the_wrong_lines(self.img, vert_lines_with_count, 0, 15, 10)
        self.horiz_lines = exclude_the_wrong_lines(self.img, horiz_lines_with_count, 1, 15, 10)
        # print('vert')
        # for ln in self.vert_lines:
        #     print(f'line: {ln}')
        # print('horiz')
        # for ln in self.horiz_lines:
        #     print(f'line: {ln}')
        # draw_lines(self.img, [self.horiz_lines, self.vert_lines], [(22, 173, 61), (180, 130, 70)])
        self.lattice_points = tmp_lattice_points
        self.lattice_points = find_intersection_lattice_points(self.img, self.horiz_lines, self.vert_lines,
                                                               self.gray_img.shape)

        # self.lattice_points = self.fill_missing_points()

    def get_board_center(self):
        if LatticePoints.frame_ind == LatticePoints.number_of_frame - 1:
            x, y = 0, 0
            for point in LatticePoints.board_center_list:
                x += point.x
                y += point.y
            x = int(x / len(LatticePoints.board_center_list))
            y = int(y / len(LatticePoints.board_center_list))
            LatticePoints.const_board_center = Point((x, y))
            LatticePoints.board_center_list = []

        if len(self.lattice_points) != 0:
            x, y = 0.0, 0.0
            for point in self.lattice_points:
                x += point.x
                y += point.y
            point = Point((int(x / len(self.lattice_points)), int(y / len(self.lattice_points))))
            LatticePoints.board_center_list.append(point)
        elif len(LatticePoints.board_center_list) == 0:
            LatticePoints.board_center_list.append(LatticePoints.const_board_center)

    def fill_missing_points(self) -> list[Point]:
        # draw_lines(img, [h_lines, v_lines], [color1, color2], False)
        # draw_points(img, [lattice_points], [color2], '12345', False)

        if len(self.lattice_points) == 0:
            return self.lattice_points
        line_ind = int(len(self.horiz_lines) / 2)
        check_line_points = get_line_points(self.lattice_points, line_ind, 0)
        self.vert_lines = self.get_extra_lines(check_line_points, line_ind, 0)

        line_ind = int(len(self.vert_lines) / 2)
        check_line_points = get_line_points(self.lattice_points, line_ind, 1)
        self.horiz_lines = self.get_extra_lines(check_line_points, line_ind, 1)

        return find_intersection_lattice_points(self.img, self.horiz_lines, self.vert_lines, self.img.shape)

    def get_extra_lines(self, line_points: list[Point], center_line_ind: int, line_type: int) -> list[Line]:
        if line_type == 0:
            lines = self.vert_lines
        else:
            lines = self.horiz_lines

        if len(line_points) < 2:
            return []
        dif_list, ratio_list, mean_ratio_lst = get_dif_list_and_ratio(self.img, line_points)
        mean_ratio = mean_ratio_lst[2]
        lines_ind_to_del = clear_points(self.img, dif_list, ratio_list, line_points, line_type)
        # lines_ind_to_del = []
        start_dif: tuple[float, float, float] = get_start_dif(dif_list, ratio_list)
        ind = 0
        ratio: float = 1
        high_border: float = 2.2
        low_border: float = 0.4
        while ind < len(dif_list) - 1:
            if dif_list[ind + 1][2] != 0:
                ratio = dif_list[ind][2] / dif_list[ind + 1][2]
            else:
                ind += 1
                continue
            ind_val = 0
            if 0 < ratio < low_border:
                ind_val = 1

            if 0 < ratio < 0.45 or 1.45 < ratio < high_border:
                if line_type == 0:
                    k = get_mean_k(lines[line_points[ind + ind_val].line_ind_v].k,
                                   lines[line_points[ind + 1 + ind_val].line_ind_v].k)
                else:
                    k = get_mean_k(lines[line_points[ind + ind_val].line_ind_h].k,
                                   lines[line_points[ind + 1 + ind_val].line_ind_h].k)
                dif = dif_list[ind + 1 - ind_val]
                x_start, y_start = line_points[ind + ind_val].x, line_points[ind + ind_val].y
                lines.append(Line())
                x, y = x_start + dif[0] / mean_ratio, y_start + dif[1] / mean_ratio
                lines[-1].set_by_point_k((round(x), round(y)), k)
                if line_type == 0:
                    line_points.insert(ind + 1 + ind_val, Point((round(x), round(y)), center_line_ind, len(lines) - 1))
                else:
                    line_points.insert(ind + 1 + ind_val, Point((round(x), round(y)), len(lines) - 1, center_line_ind))
                dif_list[ind + ind_val] = get_points_dist(line_points[ind + 1 + ind_val], line_points[ind + ind_val])
                dif_list.insert(ind + 1 + ind_val,
                                get_points_dist(line_points[ind + 2 + ind_val], line_points[ind + 1 + ind_val]))
                ind += 1
            ind += 1

        lines_ind_to_del = sorted(lines_ind_to_del, reverse=True)
        for ind in lines_ind_to_del:
            if ind >= len(lines):
                break
            lines.pop(ind)
        if start_dif[0] != -1:
            lines = self.lines_postprocessing(line_type, dif_list, mean_ratio, start_dif)
            # print(len(lines), '\n')
        return lines

    def lines_postprocessing(self, line_type: int, dif_list, mean_ratio: float, start_dif):
        dif_list = [val[2] for val in dif_list]

        if line_type == 0:
            lines = sorted(self.vert_lines, key=lambda x: x.p1[0])
        else:
            lines = sorted(self.horiz_lines, key=lambda x: x.p1[1])
        # print(f'start len: {len(lines)}')
        ind = 0
        dif = start_dif[2]
        # final_lines: list[Line] = [lines[0]]
        while ind < len(dif_list) - 1:
            dif = dif / mean_ratio
            dif_sum = dif_list[ind + 1]
            end_ind = len(dif_list) - 1
            for i, cur_dif in enumerate(dif_list[ind + 2:]):
                if abs(dif - dif_sum) > abs(dif - (dif_sum + cur_dif)):
                    dif_sum += cur_dif
                else:
                    end_ind = ind + i + 1
                    dif_list[end_ind] = dif_sum
                    break
            dif_list = dif_list[:ind + 1] + dif_list[end_ind:]
            lines = lines[:ind + 2] + lines[end_ind + 1:]
            ind += 1
        # print(f'end len: {len(lines)}')
        return lines
