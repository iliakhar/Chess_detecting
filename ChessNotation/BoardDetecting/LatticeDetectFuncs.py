from math import ceil

from ChessNotation.BoardDetecting.LinesGroups import *


def get_lines_with_count(img, tmp_lattice_points: list[Point], lines: LinesGroups):
    horiz_lines: list = []
    vert_lines: list = []
    horiz_dict: dict[int: int] = {}
    vert_dict: dict[int: int] = {}
    for point in tmp_lattice_points:
        # draw_points(img, [[point]], [(180, 130, 70)], is_wait=False)
        # draw_lines(img, [[lines.result_lines[point.line_ind_h]], [lines.result_lines[point.line_ind_v]]], [(22, 173, 61), (180, 130, 70)])
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


def get_border_cord_and_angle(points: list[Point], lines: list[Line], line_type: str):
    color1: tuple[int, int, int] = (22, 173, 61)
    # draw_lines(img, [lines], [color1])
    # draw_points(img, [points], [color1])
    # for point in points:
    #     draw_points(img, [[point]], [color1])
    if len(points) == 0:
        return None
    dif_list, _, mean_ratio = get_dif_list_and_ratio(None, points)

    for ind in range(len(mean_ratio)):
        if mean_ratio[ind] == 0:
            mean_ratio[ind] = 1
    result_dif = [0, 0, 0]
    if len(dif_list) != 0:
        if len(dif_list[0]) != 0:
            mean_dif = 0
            for dif in dif_list:
                mean_dif += dif[2]
            mean_dif /= len(dif_list)
            min_dif, max_dif = mean_dif / 1.4, mean_dif * 1.4
            dif_len = 0
            for dif in dif_list:
                if not (min_dif < dif[2] < max_dif):
                    continue
                dif_len += 1
                for i in range(3):
                    result_dif[i] += dif[i]
            if dif_len != 0:
                for i in range(3):
                    result_dif[i] /= dif_len
    mean_angle_dif: float = 0
    center_point_ind = ceil(len(points) / 2)
    if center_point_ind >= len(points):
        center_point_ind -= 1
    if line_type == 'vert':
        prev_angle = lines[points[0].line_ind_v].angle
        result_angle = lines[points[center_point_ind].line_ind_v].angle
    else:
        prev_angle = lines[points[0].line_ind_h].angle
        result_angle = lines[points[center_point_ind].line_ind_h].angle
    for pnt in points[1:]:
        angle = lines[pnt.line_ind_v].angle if line_type == 'vert' else lines[pnt.line_ind_h].angle
        angle_dif = angle - prev_angle
        angle_dif = normalize_angle(angle_dif)
        mean_angle_dif += angle_dif
        prev_angle = angle
    mean_angle_dif /= len(points)
    cur_dif = result_dif.copy()
    number_of_cells = 4
    for i in range(number_of_cells - 1):
        for ind in range(3):
            cur_dif[ind] /= mean_ratio[ind]
            result_dif[ind] += cur_dif[ind]
        result_angle += mean_angle_dif
    result_angle = normalize_angle(result_angle)
    x, y = result_dif[0], result_dif[1]
    k = math.tan(result_angle * math.pi / 180)
    # k = lines[points[-1].line_ind_v].k if line_type == 0 else lines[points[-1].line_ind_h].k
    x, y = round(x), round(y)
    # print(f'ratio: {mean_ratio}')
    return x, y, k, mean_ratio[2]


######################################
def exclude_the_wrong_lines(img, lines_lst: list, lines_type: int, angle_lim=5, dif_lim=20) -> list[Line]:
    # ln = [val[0] for val in lines_lst]
    # draw_lines(img, [ln], [(22, 173, 61), (180, 130, 70)])
    lines_lst = delete_lines_with_one_point(lines_lst)
    # print(lines_type)
    # draw_lines(img, [lines_lst], [(22, 173, 61), (180, 130, 70)])
    lines_lst = sorted(list(lines_lst), key=lambda x: x.angle)
    tmp_lines: list[list[Line]] = [[]]
    if len(lines_lst) != 0:
        lines_lst.append(lines_lst[-1])
    for ind in range(len(lines_lst) - 1):
        delta = lines_lst[ind + 1].angle - lines_lst[ind].angle
        # print(delta)
        if delta < angle_lim:
            tmp_lines[-1].append(lines_lst[ind])
        else:
            tmp_lines[-1].append(lines_lst[ind])
            tmp_lines.append([])
        # draw_lines(img, [[lines_lst[ind + 1]], [lines_lst[ind]]], [(22, 173, 61), (180, 130, 70)])
    if len(tmp_lines) > 1:
        if 180 - (tmp_lines[-1][-1].angle - tmp_lines[0][0].angle) < angle_lim:
            tmp_lines[0] += tmp_lines[-1]
            tmp_lines = tmp_lines[:-1]
    lines = max(tmp_lines, key=lambda x: len(x))
    fix_cord = 100
    if lines_type == 0:
        coords = [(fix_cord-ln.b)/ln.k for ln in lines]
    else:
        coords = [ln.k*fix_cord+ln.b for ln in lines]
    ind = 0
    lines_and_cord = list(zip(lines, coords))
    if len(lines_and_cord) != 0:
        lines_and_cord.sort(key=lambda x: x[1])
        lines, _ = list(zip(*lines_and_cord))
        lines = list(lines)
    # print(f'[{img.shape[1]*0.2}, {img.shape[1]*0.8}]  ,  [{img.shape[0]*0.2}, {img.shape[0]*0.8}]')
    while ind < len(lines) - 1:
        diff = abs(lines_and_cord[ind][1] - lines_and_cord[ind+1][1])
        # diff = get_lines_dif(lines[ind], lines[ind + 1], lines_type)
        cords = get_intersection_point(lines[ind], lines[ind + 1])
        if cords is None:
            cords = [-1, -1]
            cords = [5000, 5000]
        # print(diff, cords, lines[ind].line_len)
        # print(lines[ind])
        # draw_lines(img, [[lines[ind]], [lines[ind+1]]], [(22, 173, 61), (180, 130, 70)])
        if (diff < dif_lim) or (img.shape[1]*0.2<cords[0]<img.shape[1]*0.8 and img.shape[0]*0.2<cords[-1]<img.shape[0]*0.8):
            # print('delete')
            if ind != 0:
                dif1 = abs(normalize_angle(lines[ind].angle - lines[ind-1].angle))
                dif2 = abs(normalize_angle(lines[ind+1].angle - lines[ind - 1].angle))
                if dif1 > dif2:
                    lines.pop(ind)
                    lines_and_cord.pop(ind)
                else:
                    lines.pop(ind+1)
                    lines_and_cord.pop(ind+1)
            else:
                lines.pop(ind)
                lines_and_cord.pop(ind)
        else:
            ind += 1
    if len(lines) > 1:
        diff = abs(lines_and_cord[0][1] - lines_and_cord[-1][1])
        # diff = get_lines_dif(lines[0], lines[-1], lines_type)
        if diff < dif_lim:
            lines.pop(0)
    # if lines_type == 0:
    #     print('vert')
    # else:
    #     print('horiz')
    # for ln in lines:
    #     print(f'line: {ln}')
    # draw_lines(img, [lines], [(22, 173, 61)])
    return lines


def delete_lines_with_one_point(lines_lst: list):
    # ind = 0
    # while ind < len(lines_lst):
    #     if lines_lst[0][1] < 2:
    #         lines_lst.pop(ind)
    #     else:
    #         ind += 1
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


def find_intersection_lattice_points(h_lines: list[Line], v_lines: list[Line], shape: tuple,
                                     is_out: bool = False) -> list[Point]:
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


def get_line_points(points: list[Point], line_ind: int, line_type: str) -> list[Point]:
    if line_type == 'vert':
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
