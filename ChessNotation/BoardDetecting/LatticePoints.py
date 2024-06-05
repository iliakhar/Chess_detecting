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
                    # cv2.imshow('imagew', edges1)
                    # cv2.imshow('imagee', edges2)
                    # draw_points(self.img, [[point]], [(180, 130, 70)], is_wait=True)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # tmp_lattice_points += self.border_points

        # draw_points(self.img, [tmp_lattice_points, self.border_points], [(180, 130, 70), (0, 123, 255)])
        # draw_lines(self.img, [lines.result_lines], [(22, 173, 61), (180, 130, 70)], img_name='123456')
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
        self.lattice_points = find_intersection_lattice_points(self.horiz_lines, self.vert_lines,
                                                               self.img.shape)

        # self.lattice_points = self.fill_missing_points()

    def fill_missing_points(self) -> list[Point]:
        # draw_lines(img, [h_lines, v_lines], [color1, color2], False)
        # draw_points(img, [lattice_points], [color2], '12345', False)

        if len(self.lattice_points) == 0:
            return self.lattice_points
        line_ind = int(len(self.horiz_lines) / 2)
        check_line_points = get_line_points(self.lattice_points, line_ind, 'vert')
        self.vert_lines = self.get_extra_lines(check_line_points, line_ind, 'vert')

        line_ind = int(len(self.vert_lines) / 2)
        check_line_points = get_line_points(self.lattice_points, line_ind, 'horiz')
        self.horiz_lines = self.get_extra_lines(check_line_points, line_ind, 'horiz')

        return find_intersection_lattice_points(self.horiz_lines, self.vert_lines, self.img.shape)

    def get_extra_lines(self, line_points: list[Point], center_line_ind: int, line_type: str) -> list[Line]:
        if line_type == 'vert':
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
                if line_type == 'vert':
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
                if line_type == 'vert':
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

    def lines_postprocessing(self, line_type: str, dif_list, mean_ratio: float, start_dif):
        dif_list = [val[2] for val in dif_list]

        if line_type == 'vert':
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

    def shift_points_and_lines(self, x_shift: int, y_shift: int):
        self.lattice_points = self.shift_points(self.lattice_points, x_shift, y_shift)
        # self.border_points = self.shift_points(self.border_points, x_shift, y_shift)
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
