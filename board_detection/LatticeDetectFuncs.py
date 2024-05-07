from math import ceil

from board_detection.LinesGroups import *

def get_lines_with_count(tmp_lattice_points: list[Point], lines: LinesGroups):
    horiz_lines: list = []
    vert_lines: list = []
    horiz_dict: dict[int: int] = {}
    vert_dict: dict[int: int] = {}
    for point in tmp_lattice_points:
        if point.line_ind_h in horiz_dict:
            horiz_lines[horiz_dict[point.line_ind_h]][1] += 1
        else:
            horiz_dict[point.line_ind_h] = len(horiz_lines)
            horiz_lines.append([lines.result_lines[point.line_ind_h], 1])
        if point.line_ind_v in vert_dict:
            vert_lines[vert_dict[point.line_ind_v]][1] += 1
        else:
            vert_dict[point.line_ind_v] = len(vert_lines)
            vert_lines.append([lines.result_lines[point.line_ind_v], 1])
    return vert_lines, horiz_lines


def get_border_cord_and_angle(img, points: list[Point], lines: list[Line], line_type: int):
    color1: tuple[int, int, int] = (22, 173, 61)
    # draw_points(img, [points], [color1])
    if len(points) == 0:
        return None
    dif_list, ratio_list, mean_ratio = get_dif_list_and_ratio(None, points)
    for ind in range(len(mean_ratio)):
        if mean_ratio[ind] == 0:
            mean_ratio[ind] = 1
    # result_dif = list(get_start_dif(dif_list, ratio_list))
    result_dif = [0,0,0]
    if len(dif_list) != 0:
        if len(dif_list[0]) != 0:
            for dif in dif_list:
                for i in range(3):
                    result_dif[i] += dif[i]
            for i in range(3):
                result_dif[i] /= len(dif_list)
    # mean_angle_dif: float = 0
    # center_point_ind = len(points) // 2
    # prev_angle = lines[points[center_point_ind].line_ind_v].angle if line_type == 0 else lines[points[center_point_ind].line_ind_h].angle
    # result_angle = prev_angle
    # print(result_angle)
    # for pnt in points[1:]:
    #     angle = lines[pnt.line_ind_v].angle if line_type == 0 else lines[pnt.line_ind_h].angle
    #     angle_dif = angle - prev_angle
    #     if angle_dif > 90:
    #         angle_dif -= 180
    #     if angle_dif < -90:
    #         angle_dif += 180
    #     mean_angle_dif += angle_dif
    #     prev_angle = angle
    # mean_angle_dif /= len(points)
    cur_dif = result_dif.copy()
    number_of_cells = 4
    for i in range(number_of_cells - 1):
        for ind in range(3):
            cur_dif[ind] /= mean_ratio[ind]
            result_dif[ind] += cur_dif[ind]
    #     result_angle += mean_angle_dif
    # if result_angle > 90:
    #     result_angle -= 180
    # elif result_angle < -90:
    #     result_angle += 180
    x, y = result_dif[0], result_dif[1]

    # print(f'len: {result_dif}, x: {x}, y: {y}    |   angle: {result_angle}, anlge dif: {mean_angle_dif}')
    # k = math.tan(result_angle * math.pi / 180)
    k = lines[points[-1].line_ind_v].k if line_type == 0 else lines[points[-1].line_ind_h].k
    x, y = round(x), round(y)
    return x, y, k


######################################
def exclude_the_wrong_lines(img, lines_lst: list, lines_type: int, angle_lim=5) -> list[Line]:
    lines_lst = delete_border_lines(lines_lst, lines_type)
    lines_lst = sorted(list(lines_lst), key=lambda x: x.angle)
    l1 = lines_lst.copy()
    color1: tuple[int, int, int] = (22, 173, 61)
    color2: tuple[int, int, int] = (180, 130, 70)
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
    # print(len(tmp_lines))
    # colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(len(tmp_lines))]
    # draw_lines(img, tmp_lines, colors)
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


def delete_border_lines(lines_lst: list, lines_type: int):
    # if lines_type == 0:  # vert
    #     lines_lst = sorted(lines_lst, key=lambda x: x[0].p1[0])
    # else:  # horiz
    #     lines_lst = sorted(lines_lst, key=lambda x: x[0].p1[1])
    #
    # while len(lines_lst) > 0:
    #     if lines_lst[0][1] < 2:
    #         lines_lst.pop(0)
    #     else:
    #         break
    # while len(lines_lst) > 0:
    #     if lines_lst[-1][1] < 2:
    #         lines_lst.pop(-1)
    #     else:
    #         break
    return [val[0] for val in lines_lst]


def get_dif_list_and_ratio(img, line_points: list[Point]):
    dif_list: list[tuple[float, float, float]] = []
    for ind in range(1, len(line_points)):
        dif_list.append(get_points_dist(line_points[ind], line_points[ind - 1]))

    mean_ratio = (0, 0, 0)
    ratio_list: list[float] = []
    number_of_ratio = 0
    for ind, _ in enumerate(dif_list[:-1]):
        ratio = [0, 0, 0]
        if dif_list[ind + 1][2] != 0:
            for i in range(3):
                ratio[i] = 0 if dif_list[ind + 1][i] == 0 else dif_list[ind][i] / dif_list[ind + 1][i]
            # ratio = dif_list[ind][0] / dif_list[ind + 1][0], dif_list[ind][1] / dif_list[ind + 1][1], dif_list[ind][0] / dif_list[ind + 1][2]
            ratio_list.append(ratio[2])
            if 0.6 <= ratio[2] <= 1.4:
                mean_ratio = mean_ratio[0] + ratio[0], mean_ratio[1] + ratio[1], mean_ratio[2] + ratio[2]
                number_of_ratio += 1
        else:
            ratio_list.append(0)
    if number_of_ratio == 0:
        return dif_list, ratio_list, (1, 1, 1)
    else:
        mean_ratio = mean_ratio[0] / number_of_ratio, mean_ratio[1] / number_of_ratio, mean_ratio[2] / number_of_ratio
        return dif_list, ratio_list, list(mean_ratio)


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


def get_point_neighborhood(img: np.ndarray, point: Point) -> np.ndarray:
    x1, y1 = point.x - 10, point.y - 10
    sub_img = img[y1:y1 + 21, x1:x1 + 21]
    ret, thresh1 = cv2.threshold(sub_img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)
    return cv2.Canny(thresh1, 20, 30, apertureSize=3)


def find_intersection_lattice_points(img, h_lines: list[Line], v_lines: list[Line], shape: tuple, is_out: bool = False) -> list[Point]:
    points_list = []
    for h_ind, h_ln in enumerate(h_lines):
        for v_ind, v_ln in enumerate(v_lines):
            point = get_intersection_point(h_ln, v_ln)
            if type(point) is tuple:
                if (0 < point[0] < shape[1] and 0 < point[1] < shape[0]) or is_out:
                    points_list.append(Point(point, h_ind, v_ind))
    return points_list


def get_start_dif(dif_list, ratio_list) -> tuple[float, float, float]:
    for ind, ratio in enumerate(ratio_list):
        if 0.8 < ratio < 1.2:
            return dif_list[ind]
    return -1, -1, -1


def get_line_points(points: list[Point], line_ind: int, line_type: int) -> list[Point]:
    if line_type == 0:
        lmbd1, lmbd2 = lambda x: x.line_ind_h, lambda x: x.x

    else:
        lmbd1, lmbd2 = lambda x: x.line_ind_v, lambda x: x.y
    points = sorted(points, key=lmbd1)
    line_start = bisect_left(points, line_ind, key=lmbd1)
    line_end = bisect_right(points, line_ind, key=lmbd1)
    return sorted(points[line_start:line_end], key=lmbd2)


def clear_points(img, dif_list, ratio_list, line_points, line_type):
    high_border: float = 3
    low_border: float = 0.33
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
