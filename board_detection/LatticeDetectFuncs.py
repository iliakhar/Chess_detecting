from bisect import bisect_right
from random import randint

import cv2
import numpy as np

from Line import *
from Point import *
from board_detection.UsefulFunctions import draw_lines, draw_points


def get_point_neighborhood(img: np.ndarray, point: Point) -> np.ndarray:
    x1, y1 = point.x - 10, point.y - 10
    sub_img = img[y1:y1 + 21, x1:x1 + 21]
    ret, thresh1 = cv2.threshold(sub_img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)
    return cv2.Canny(thresh1, 20, 30, apertureSize=3)


def delete_border_lines(lines_lst: list, lines_type: int):
    if lines_type == 0:  # vert
        lines_lst = sorted(lines_lst, key=lambda x: x[0].p1[0])
    else:  # horiz
        lines_lst = sorted(lines_lst, key=lambda x: x[0].p1[1])

    while len(lines_lst) > 0:
        if lines_lst[0][1] < 2:
            lines_lst.pop(0)
        else:
            break
    while len(lines_lst) > 0:
        if lines_lst[-1][1] < 2:
            lines_lst.pop(-1)
        else:
            break
    return [val[0] for val in lines_lst]


def exclude_the_wrong_lines(img, lines_lst: list, lines_type: int, angle_lim=5) -> list[Line]:
    lines_lst = delete_border_lines(lines_lst, lines_type)
    lines_lst = sorted(list(lines_lst), key=lambda x: x.angle)
    tmp_lines: list[list[Line]] = [[]]
    if len(lines_lst) != 0:
        lines_lst.append(lines_lst[-1])
    for ind in range(len(lines_lst) - 1):
        delta = lines_lst[ind + 1].angle - lines_lst[ind].angle
        if delta < angle_lim:
            tmp_lines[-1].append(lines_lst[ind])
        else:
            tmp_lines[-1].append(lines_lst[ind])
            tmp_lines.append([])
    if len(tmp_lines) > 1:
        if 180 - (tmp_lines[-1][-1].angle - tmp_lines[0][0].angle) < angle_lim:
            tmp_lines[0] += tmp_lines[-1]
            tmp_lines = tmp_lines[:-1]
    lines = max(tmp_lines, key=lambda x: len(x))
    if lines_type == 0:
        lines = sorted(lines, key=lambda x: x.p1[0])
    else:
        lines = sorted(lines, key=lambda x: x.p1[1])
    ind = 0
    while ind < len(lines) - 1:

        dif_l = abs(lines[ind].left_normal - lines[ind + 1].left_normal)
        dif_r = abs(lines[ind].right_normal - lines[ind + 1].right_normal)
        if dif_l < 10 and dif_r < 10:
            lines.pop(ind)
        else:
            ind += 1
    if len(lines) > 1:
        dif_l = abs(lines[0].left_normal - lines[-1].left_normal)
        dif_r = abs(lines[0].right_normal - lines[-1].right_normal)
        if dif_l < 15 and dif_r < 15:
            lines.pop(0)
    return lines


def find_intersection_lattice_points(img, h_lines: list[Line], v_lines: list[Line], shape: tuple) -> list[Point]:
    points_set: set = set()
    for h_ind, h_ln in enumerate(h_lines):
        for v_ind, v_ln in enumerate(v_lines):
            point = get_intersection_point(h_ln, v_ln)
            if type(point) is tuple:
                if 0 < point[0] < shape[1] and 0 < point[1] < shape[0]:
                    points_set.add(Point(point, h_ind, v_ind))

    return list(points_set)


def fill_missing_points(img, lattice_points: list[Point], h_lines: list[Line], v_lines: list[Line]) -> list[Point]:
    # color1: tuple[int, int, int] = (22, 173, 61)
    # color2: tuple[int, int, int] = (180, 130, 70)
    # draw_lines(img, [h_lines, v_lines], [color1, color2], False)
    # draw_points(img, [lattice_points], [color2], '12345', False)

    if len(lattice_points) == 0:
        return lattice_points

    lattice_points = sorted(lattice_points, key=lambda x: x.line_ind_h)
    line_end = bisect_right(lattice_points, 0, key=lambda x: x.line_ind_h)
    first_line = sorted(lattice_points[:line_end], key=lambda x: x.x)
    get_extra_lines(img, first_line, v_lines, img.shape, 0)

    lattice_points = sorted(lattice_points, key=lambda x: x.line_ind_v)
    line_end = bisect_right(lattice_points, 0, key=lambda x: x.line_ind_v)
    first_line = sorted(lattice_points[:line_end], key=lambda x: x.y)
    get_extra_lines(img, first_line, h_lines, img.shape, 1)

    return find_intersection_lattice_points(img, h_lines, v_lines, img.shape)


def get_extra_lines(img, line_points: list[Point], lines: list[Line], shape, line_type: int) -> list[Line]:
    color1: tuple[int, int, int] = (22, 173, 61)
    color2: tuple[int, int, int] = (180, 130, 70)
    # draw_lines(img, [lines], [color1])
    if len(line_points) < 2:
        return []
    dif_list, ratio_list, mean_ratio = get_dif_list_and_ratio(img, line_points)
    lines_ind_to_del = clear_points(img, dif_list, ratio_list, line_points, line_type)
    start_dif: tuple[float, float, float] = get_start_dif(dif_list, ratio_list)

    ind = 0
    ratio: float = 1
    high_border: float = 4
    low_border: float = 0.25
    # print(dif_list, mean_ratio)
    while ind < len(dif_list) - 1:
        if dif_list[ind + 1][2] != 0:
            ratio = dif_list[ind][2] / dif_list[ind + 1][2]
        else:
            ind += 1
            continue

        if 0.1 < ratio < low_border:
            if line_type == 0:
                k = get_mean_k(lines[line_points[ind + 1].line_ind_v].k, lines[line_points[ind + 2].line_ind_v].k)
            else:
                k = get_mean_k(lines[line_points[ind + 1].line_ind_h].k, lines[line_points[ind + 2].line_ind_h].k)
            dif = dif_list[ind]
            x_start, y_start = line_points[ind + 1].x, line_points[ind + 1].y
            lines.append(Line())
            x, y = x_start + dif[0] / mean_ratio, y_start + dif[1] / mean_ratio
            lines[-1].set_by_point_k((round(x), round(y)), k)
            if line_type == 0:
                line_points.insert(ind + 2, Point((round(x), round(y)), 0, len(lines) - 1))
            else:
                line_points.insert(ind + 2, Point((round(x), round(y)), len(lines) - 1, 0))
            dif_list[ind + 1] = get_points_dist(line_points[ind + 2], line_points[ind + 1])
            dif_list.insert(ind + 2, get_points_dist(line_points[ind + 3], line_points[ind + 2]))
            # ind += 1
        elif 1.4 < ratio < high_border:
            if line_type == 0:
                k = get_mean_k(lines[line_points[ind].line_ind_v].k, lines[line_points[ind + 1].line_ind_v].k)
            else:
                k = get_mean_k(lines[line_points[ind].line_ind_h].k, lines[line_points[ind + 1].line_ind_h].k)
            dif = dif_list[ind + 1]
            x_start, y_start = line_points[ind].x, line_points[ind].y
            lines.append(Line())
            x, y = x_start + dif[0] / mean_ratio, y_start + dif[1] / mean_ratio
            lines[-1].set_by_point_k((round(x), round(y)), k)
            if line_type == 0:
                line_points.insert(ind + 1, Point((round(x), round(y)), 0, len(lines) - 1))
            else:
                line_points.insert(ind + 1, Point((round(x), round(y)), len(lines) - 1, 0))
            dif_list[ind] = get_points_dist(line_points[ind + 1], line_points[ind])
            dif_list.insert(ind + 1, get_points_dist(line_points[ind + 2], line_points[ind + 1]))
            # ind += 1

        ind += 1
    lines_ind_to_del = sorted(lines_ind_to_del, reverse=True)
    for ind in lines_ind_to_del:
        if ind >= len(lines):
            break
        lines.pop(ind)
    if start_dif[0] != -1:
        lines_postprocessing(lines, line_type, dif_list, mean_ratio, start_dif)


def get_start_dif(dif_list, ratio_list) -> tuple[float, float, float]:
    for ind, ratio in enumerate(ratio_list):
        if 0.8 < ratio < 1.2:
            return dif_list[ind]
    return -1, -1, -1


def lines_postprocessing(lines: list[Line], line_type: int, dif_list, mean_ratio: float, start_dif):
    dif_list = [val[2] for val in dif_list]
    print(f'start len: {len(lines)}')
    if line_type == 0:
        lines = sorted(lines, key=lambda x: x.p1[0])
    else:
        lines = sorted(lines, key=lambda x: x.p1[1])
    ind = 0
    dif = start_dif[2]
    # final_lines: list[Line] = [lines[0]]
    while ind < len(dif_list) - 1:
        dif = dif / mean_ratio
        dif_sum = 0
        end_ind = len(dif_list)
        for i, cur_dif in enumerate(dif_list[ind + 1:]):
            if abs(dif - dif_sum) > abs(dif - (dif_sum + cur_dif)):
                dif_sum += cur_dif
            else:
                end_ind = ind + i + 1
                dif_list[end_ind] = dif_sum
                break
        dif_list = dif_list[:ind + 1] + dif_list[end_ind:]
        lines = lines[:ind + 2] + lines[end_ind + 1:]
        ind += 1
    print(f'end len: {len(lines)}\n')


def clear_points(img, dif_list, ratio_list, line_points, line_type):
    color1: tuple[int, int, int] = (22, 173, 61)
    color2: tuple[int, int, int] = (180, 130, 70)
    high_border: float = 2
    low_border: float = 0.5
    ind: int = 0
    lines_ind_to_del: list[int] = []
    if len(ratio_list) == 0:
        return []
    while ind < len(ratio_list) - 1:
        if ratio_list[ind] >= high_border and ratio_list[ind + 1] <= low_border:
            # draw_points(img, [line_points], [color1])
            if line_type == 0:
                lines_ind_to_del.append(line_points[ind + 2].line_ind_v)
            else:
                lines_ind_to_del.append(line_points[ind + 2].line_ind_h)
            dif_list.pop(ind + 1)
            line_points.pop(ind + 2)
            dif_list[ind + 1] = get_points_dist(line_points[ind + 2], line_points[ind + 1])
            ratio = dif_list[ind][2] / dif_list[ind + 1][2]
            ratio_list[ind] = ratio
            ratio_list.pop(ind + 1)
        elif ratio_list[ind] >= high_border and ind == 0:
            if line_type == 0:
                lines_ind_to_del.append(line_points[ind].line_ind_v)
            else:
                lines_ind_to_del.append(line_points[ind].line_ind_h)
            dif_list.pop(ind)
            line_points.pop(ind)
            ratio_list.pop(ind + 1)
        else:
            ind += 1

    if ratio_list[0] <= low_border:
        if line_type == 0:
            lines_ind_to_del.append(line_points[0].line_ind_v)
        else:
            lines_ind_to_del.append(line_points[0].line_ind_h)
        dif_list.pop(0)
        line_points.pop(0)

    if ratio_list[-1] >= high_border:
        if line_type == 0:
            lines_ind_to_del.append(line_points[-1].line_ind_v)
        else:
            lines_ind_to_del.append(line_points[-1].line_ind_h)
        dif_list.pop(-1)
        line_points.pop(-1)
        ratio_list.pop(-1)

    return lines_ind_to_del


def get_dif_list_and_ratio(img, first_line_points: list[Point]):
    dif_list: list[tuple[float, float, float]] = []
    for ind in range(1, len(first_line_points)):
        dif_list.append(get_points_dist(first_line_points[ind], first_line_points[ind - 1]))

    mean_ratio = 0
    ratio_list: list[float] = []
    number_of_ratio = 0
    for ind, _ in enumerate(dif_list[:-1]):
        if dif_list[ind + 1][2] != 0:
            ratio = dif_list[ind][2] / dif_list[ind + 1][2]
            ratio_list.append(ratio)
            if 0.6 <= ratio <= 1.4:
                mean_ratio += ratio
                number_of_ratio += 1
        else:
            ratio_list.append(0)
    if number_of_ratio == 0:
        return dif_list, ratio_list, 1
    else:
        return dif_list, ratio_list, mean_ratio / number_of_ratio


def get_mean_k(k1: float, k2: float) -> float:
    k: float = 100
    if k1 * k2 > 0:
        k = (k1 + k2) / 2
    return k


def get_points_dist(point1: Point, point2: Point) -> tuple[float, float, float]:
    x_dist = point1.x - point2.x
    y_dist = point1.y - point2.y
    dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
    return x_dist, y_dist, dist
