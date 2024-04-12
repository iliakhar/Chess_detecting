import math

import cv2
import bisect
from Line import *


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


def draw_lines(img: np.ndarray, grouped_lines: list[list[Line]], colors: list[tuple], is_wait: bool = True) -> None:
    tmp_img = img.copy()
    for group_ind, group in enumerate(grouped_lines):
        for line in group:
            x1, y1 = line.p1
            x2, y2 = line.p2
            cv2.line(tmp_img, (x1, y1), (x2, y2), colors[group_ind % len(colors)], 2)
            # cv2.line(tmp_img, (x1, y1), (x2, y2), colors[ind], 2)
    cv2.imshow('image', tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_points(img: np.ndarray, grouped_points: list[list[tuple[int, int]]], colors: list[tuple],
                img_name: str = 'image0', is_wait: bool = True) -> None:
    tmp_img = img.copy()
    for group_ind, group in enumerate(grouped_points):
        for point in group:
            cv2.circle(tmp_img, center=point, radius=6, color=colors[group_ind % len(colors)], thickness=-1, )
    cv2.imshow(img_name, tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_intersection_points(lines: list[Line]) -> list[tuple[int, int]]:
    points_list: set = set()
    lines.sort(key=lambda x: x.k)
    for ind, line1 in enumerate(lines):
        left_border = bisect.bisect_left(lines, line1.k-0.8, key=lambda x: x.k)
        right_border = bisect.bisect_right(lines, line1.k+0.8, key=lambda x: x.k)

        for line_ind in range(ind+1, len(lines)):
            if left_border < line_ind < right_border:
                continue
            point: tuple[int, int] = get_intersection_point(line1, lines[line_ind])
            if type(point) is tuple:
                if (check_range(point[0], line1.p1[0], line1.p2[0]) and check_range(point[1], line1.p1[1],
                                                                                    line1.p2[1]) and check_range(
                    point[0], lines[line_ind].p1[0], lines[line_ind].p2[0])
                        and check_range(point[1], lines[line_ind].p1[1], lines[line_ind].p2[1])):
                    points_list.add(point)

    return list(points_list)


def check_range(num: int, left: int, right: int):
    if right > left:
        return left <= num <= right
    else:
        return right <= num <= left


def get_split_inds(left_border: int, right_border: int, numper_of_parts) -> tuple[int, list[int]]:
    lst_size: int = right_border - left_border + 1
    block_size: int = math.ceil(lst_size / numper_of_parts)
    return block_size, [i for i in range(left_border, right_border + 1, block_size)]
