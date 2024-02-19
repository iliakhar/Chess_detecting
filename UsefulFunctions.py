import cv2
import numpy as np
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
    cv2.imshow('image', tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_points(img: np.ndarray, grouped_points: list[list[tuple[int, int]]], colors: list[tuple], is_wait: bool = True) -> None:
    tmp_img = img.copy()
    for group_ind, group in enumerate(grouped_points):
        for point in group:
            cv2.circle(tmp_img, center=point, radius=2, color=colors[group_ind % len(colors)], thickness=2)
    cv2.imshow('image', tmp_img)
    if is_wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
