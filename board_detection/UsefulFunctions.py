import math

import cv2
from bisect import bisect_left, bisect_right
from Line import *
from Point import *


def resizing(img, new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    # print(type(img))
    h, w = img.shape[:2]

    if new_width is None and new_height is None:
        return img
    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)
    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))
    return cv2.resize(img, dimension, interpolation=interp)


def draw_lines(img: np.ndarray, grouped_lines: list[list[Line]], colors: list[tuple], is_wait: bool = True,
               img_name='im') -> None:
    tmp_img = img.copy()
    for group_ind, group in enumerate(grouped_lines):
        for line in group:
            x1, y1 = line.p1
            x2, y2 = line.p2
            if 0 < x1 < img.shape[1] and 0 < y1 < img.shape[0]:
                cv2.line(tmp_img, (x1, y1), (x2, y2), colors[group_ind % len(colors)], 2)
            # cv2.line(tmp_img, (x1, y1), (x2, y2), colors[ind], 2)
    cv2.imshow(img_name, tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_points(img: np.ndarray, grouped_points: list[list[Point]], colors: list[tuple],
                img_name: str = 'image0', is_wait: bool = True) -> None:
    tmp_img = img.copy()
    for group_ind, group in enumerate(grouped_points):
        for point in group:
            cv2.circle(tmp_img, center=(point.x, point.y), radius=6, color=colors[group_ind % len(colors)],
                       thickness=-1, )
    cv2.imshow(img_name, tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def split_lines_by_orientation(lines: list[Line]):
    max_dif = [0, 0]
    max_dif_inds = [0, 0]
    for ind in range(len(lines) - 1):
        dif = abs(normalize_angle(lines[ind].angle - lines[ind + 1].angle))
        print(dif)
        if dif > max_dif[0]:
            max_dif_inds[1] = max_dif_inds[0]
            max_dif[1] = max_dif[0]
            max_dif_inds[0] = ind + 1
            max_dif[0] = dif
        elif dif > max_dif[1]:
            max_dif[1] = dif
            max_dif_inds[1] = ind + 1
    dif = abs(normalize_angle(lines[-1].angle - lines[0].angle))
    if dif > max_dif[1]:
        max_dif[1] = dif
        max_dif_inds[1] = 0
    for ind, ln in enumerate(lines):
        print(f'{ind}) {ln}')
    print(max_dif, max_dif_inds)
    max_dif_inds = sorted(max_dif_inds)
    vert_lines = lines[max_dif_inds[0]:max_dif_inds[1]]
    horiz_lines = lines[max_dif_inds[1]:]
    if len(vert_lines) == 0 or len(horiz_lines) == 0:
        return [], [], -1, -1
    if max_dif_inds[0] != 0:
        horiz_lines += lines[0:max_dif_inds[0]]
    angle1 = vert_lines[len(vert_lines)//2].angle
    angle2 = horiz_lines[len(horiz_lines)//2].angle
    # print(angle1, angle2)
    if abs(abs(angle1) - abs(angle2)) < 25:
        if angle1 < angle2:
            vert_lines, horiz_lines = horiz_lines, vert_lines
            max_dif_inds[0], max_dif_inds[1] = max_dif_inds[1], max_dif_inds[0]
    elif abs(angle1) < abs(angle2):
        vert_lines, horiz_lines = horiz_lines, vert_lines
        max_dif_inds[0], max_dif_inds[1] = max_dif_inds[1], max_dif_inds[0]
    # print(angle1, angle2)
    return vert_lines, horiz_lines, max_dif_inds[0], max_dif_inds[1]


# def find_intersection_points(img, lines: list[Line]) -> list[Point]:
#     if len(lines) == 0:
#         return []
#     lines.sort(key=lambda line: line.angle)
#     vert_lines, horiz_lines, vert_pos, horiz_pos = split_lines_by_orientation(lines)
#     draw_lines(img, [horiz_lines, vert_lines], [(22, 173, 61), (180, 130, 70)])
#     if vert_pos == -1:
#         return []
#     points_set: set = set()
#     for ind_v, vert in enumerate(vert_lines):
#         for ind_h, horiz in enumerate(horiz_lines):
#             point: tuple[int, int] = get_intersection_point(vert, horiz)
#             if type(point) is tuple:
#
#                 h_ind_in_lines = (ind_h+horiz_pos) % len(lines)
#                 v_ind_in_lines = (ind_v+vert_pos) % len(lines)
#                 pnt = Point(point, h_ind_in_lines, v_ind_in_lines)
#                 points_set.add(pnt)
#     return list(points_set)


def find_intersection_points(img, lines: list[Line]) -> list[Point]:
    lim_angle = 30
    points_set: set = set()
    lines.sort(key=lambda x: x.angle)
    line_h = []
    line_v = []
    for ind, line1 in enumerate(lines):
        left_border = bisect_left(lines, line1.angle - lim_angle, key=lambda x: x.angle)
        right_border = bisect_right(lines, line1.angle + lim_angle, key=lambda x: x.angle)
        for line_ind in range(ind + 1, len(lines)):
            if left_border < line_ind < right_border:
                continue
            point: tuple[int, int] = get_intersection_point(line1, lines[line_ind])
            if type(point) is tuple:
                if check_is_point_in_sections(point, line1, lines[line_ind]):
                    angle1 = lines[ind].angle
                    angle2 = lines[line_ind].angle
                    # print(angle1, angle2)
                    if abs(abs(angle1) - abs(angle2)) < 15:
                        pnt = Point(point, ind, line_ind) if angle1 < angle2 else Point(point, line_ind, ind)
                    else:
                        pnt = Point(point, ind, line_ind) if abs(angle1) < abs(angle2) else Point(point, line_ind, ind)
                    line_v.append(lines[pnt.line_ind_v])
                    line_h.append(lines[pnt.line_ind_h])
                    points_set.add(pnt)
    # print(len(line_v), len(line_h))
    # for pnt in points_set:
    #     print(lines[pnt.line_ind_h].angle, lines[pnt.line_ind_v].angle)
    #     draw_lines(img, [[lines[pnt.line_ind_h]], [lines[pnt.line_ind_v]]], [(22, 173, 61), (180, 130, 70)])  # r b
    # draw_lines(img, [line_h, line_v], [(22, 173, 61), (180, 130, 70)])  # r b
    # draw_lines(img, [line_h], [(22, 173, 61)])  # r b
    # draw_lines(img, [line_v], [(22, 173, 61)])  # r b
    return list(points_set)

def check_is_point_in_sections(point: tuple[int, int], line1: Line, line2: Line) -> bool:
    return (check_range(point[0], line1.p1[0], line1.p2[0]) and check_range(point[1], line1.p1[1],
            line1.p2[1]) and check_range(point[0], line2.p1[0], line2.p2[0])
            and check_range(point[1], line2.p1[1], line2.p2[1]))


def check_range(num: int, left: int, right: int):
    if right > left:
        return left <= num <= right
    else:
        return right <= num <= left


def get_split_inds(left_border: int, right_border: int, numper_of_parts) -> tuple[int, list[int]]:
    lst_size: int = right_border - left_border + 1
    block_size: int = math.ceil(lst_size / numper_of_parts)
    return block_size, [i for i in range(left_border, right_border + 1, block_size)]


def normalize_angle(angle: float) -> float:
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def get_xy_dist(p1: Point, p2: Point) -> tuple[int, int]:
    return abs(p1.x - p2.x), abs(p1.y - p2.y)
