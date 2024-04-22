from bisect import bisect_right

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


def exclude_the_wrong_lines(lines_lst: list[Line], angle_lim=5) -> list[Line]:
    lines_lst = sorted(list(lines_lst), key=lambda x: x.angle)
    tmp_horiz = [[]]
    if len(lines_lst) != 0:
        lines_lst.append(lines_lst[-1])
    for ind in range(len(lines_lst) - 1):
        delta = lines_lst[ind + 1].angle - lines_lst[ind].angle
        if delta < angle_lim:
            tmp_horiz[-1].append(lines_lst[ind])
        else:
            tmp_horiz[-1].append(lines_lst[ind])
            tmp_horiz.append([])
    if len(tmp_horiz) > 1:
        if 180 - (tmp_horiz[-1][-1].angle - tmp_horiz[0][0].angle) < angle_lim:
            tmp_horiz[0] += tmp_horiz[-1]
            tmp_horiz = tmp_horiz[:-1]
    # for h in tmp_horiz:
    #     for ln in h:
    #         print(ln)
    #     print('\n')
    # print('--------------------------------------------------------')
    return max(tmp_horiz, key=lambda x: len(x))


def find_intersection_lattice_points(img, h_lines: list[Line], v_lines: list[Line], shape: tuple) -> list[Point]:
    points_set: set = set()
    for h_ind, h_ln in enumerate(h_lines):
        for v_ind, v_ln in enumerate(v_lines):
            point = get_intersection_point(h_ln, v_ln)
            if type(point) is tuple:
                # color1: tuple[int, int, int] = (22, 173, 61)
                # color2: tuple[int, int, int] = (180, 130, 70)
                # draw_lines(img, [[h_ln], [v_ln]], [color1, color2])
                if 0 < point[0] < shape[1] and 0 < point[1] < shape[0]:
                    points_set.add(Point(point, h_ind, v_ind))

    return list(points_set)


def fill_missing_points(img, lattice_points: list[Point], h_lines: list[Line], v_lines: list[Line]) -> list[Point]:
    if len(lattice_points) == 0:
        return lattice_points

    lattice_points = sorted(lattice_points, key=lambda x: x.line_ind_h)
    line_end = bisect_right(lattice_points, 0, key=lambda x: x.line_ind_h)
    first_line = sorted(lattice_points[:line_end], key=lambda x: x.x)
    v_lines += get_extra_lines(img, first_line, v_lines, img.shape, 0)

    lattice_points = sorted(lattice_points, key=lambda x: x.line_ind_v)
    line_end = bisect_right(lattice_points, 0, key=lambda x: x.line_ind_v)
    first_line = sorted(lattice_points[:line_end], key=lambda x: x.y)
    h_lines += get_extra_lines(img, first_line, h_lines, img.shape, 1)

    # print('------------------------------------')

    return find_intersection_lattice_points(img, h_lines, v_lines, img.shape)


def get_extra_lines(img, line_points: list[Point], lines: list[Line], shape, line_type: int) -> list[Line]:
    # color1: tuple[int, int, int] = (22, 173, 61)
    # draw_lines(img, [lines], [color1])
    l = lines.copy()
    if len(line_points) < 2:
        return []
    dif_list, mean_ratio = get_dif_list_and_ratio(img, line_points)
    ind = 0
    while ind < len(dif_list):
        if dif_list[ind][2] < 7:
            if line_type == 0:
                lines.pop(line_points[ind].line_ind_v)
            else:
                lines.pop(line_points[ind].line_ind_h)
            line_points.pop(ind)
            dif_list.pop(ind)
        else:
            ind += 1

    # color1: tuple[int, int, int] = (22, 173, 61)
    # draw_lines(img, [lines], [color1])

    extra_lines: list[Line] = []
    ind = 0
    ratio: float = 0
    while ind < len(dif_list) - 1:
        if dif_list[ind + 1][2] != 0:
            ratio = dif_list[ind][2] / dif_list[ind + 1][2]
        else:
            ind += 1
            continue

        if 0.1 < ratio < 0.6:
            if line_type == 0:
                k = get_mean_k(lines[line_points[ind + 1].line_ind_v].k, lines[line_points[ind + 2].line_ind_v].k)
            else:
                k = get_mean_k(lines[line_points[ind + 1].line_ind_h].k, lines[line_points[ind + 2].line_ind_h].k)
            dif = dif_list[ind]
            x_start, y_start = line_points[ind + 1].x, line_points[ind + 1].y
            extra_lines.append(Line())
            x, y = x_start + dif[0] * mean_ratio, y_start + dif[1] * mean_ratio
            extra_lines[-1].set_by_point_k((round(x), round(y)), k)
            line_points.insert(ind + 2, Point((round(x), round(y))))
            dif_list[ind + 1] = get_points_dist(line_points[ind + 2], line_points[ind + 1])
            dif_list.insert(ind + 2, get_points_dist(line_points[ind + 3], line_points[ind + 2]))
        elif 1.4 < ratio < 10:
            if line_type == 0:
                k = get_mean_k(lines[line_points[ind].line_ind_v].k, lines[line_points[ind + 1].line_ind_v].k)
            else:
                k = get_mean_k(lines[line_points[ind].line_ind_h].k, lines[line_points[ind + 1].line_ind_h].k)
            dif = dif_list[ind]
            x_start, y_start = line_points[ind].x, line_points[ind].y
            extra_lines.append(Line())
            x, y = x_start + dif[0] * mean_ratio, y_start + dif[1] * mean_ratio
            extra_lines[-1].set_by_point_k((round(x), round(y)), k)
            line_points.insert(ind + 1, Point((round(x), round(y))))
            dif_list[ind] = get_points_dist(line_points[ind + 1], line_points[ind])
            dif_list.insert(ind + 1, get_points_dist(line_points[ind + 2], line_points[ind + 1]))

        ind += 1
    # color1: tuple[int, int, int] = (22, 173, 61)
    # color2: tuple[int, int, int] = (180, 130, 70)
    # draw_lines(img, [lines, extra_lines], [color1, color2])
    return extra_lines


def get_dif_list_and_ratio(img, first_line_points: list[Point]):
    dif_list: list[tuple[float, float, float]] = []
    # color1: tuple[int, int, int] = (22, 173, 61)
    for ind in range(1, len(first_line_points)):
        dif_list.append(get_points_dist(first_line_points[ind], first_line_points[ind - 1]))
        # print(dif_list[-1])
        # draw_points(img, [[first_line_points[ind], first_line_points[ind - 1]]], [color1])

    mean_ratio = 0
    number_of_ratio = 0
    for ind, _ in enumerate(dif_list[:-1]):
        if dif_list[ind + 1][2] != 0:
            ratio = dif_list[ind][2] / dif_list[ind + 1][2]
            if 0.6 <= ratio <= 1.4:
                mean_ratio += ratio
                number_of_ratio += 1
    if number_of_ratio == 0:
        number_of_ratio = 1
    return dif_list, mean_ratio / number_of_ratio


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
