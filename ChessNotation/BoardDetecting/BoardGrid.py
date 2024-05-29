from ChessNotation.BoardDetecting.ChessBoardDetecting import *
from ChessNotation.BoardDetecting.LatticeDetectFuncs import *

def change_lines_size(lines: list[Line], size_kx: float, size_ky: float):
    new_lines: list[Line] = []
    for ln in lines:
        x1, y1 = round(ln.p1[0]*size_kx), round(ln.p1[1]*size_ky)
        x2, y2 = round(ln.p2[0]*size_kx), round(ln.p2[1]*size_ky)
        ln.set_by_raw_line(np.asarray([x1, y1, x2, y2]))
        new_lines.append(ln)
    return new_lines


class BoardGrid:
    board_center_list: list[Point] = []
    const_board_center: Point = Point((5, 5))
    borders_info: list[list] = [[], [], [], []]
    const_grid: list[Line] = []
    const_vert_lines: list[Line] = []
    const_horiz_lines: list[Line] = []
    const_img_size: tuple[int, int] = (1, 1)
    number_of_frame: int = 15
    frame_ind: int = 0

    def __init__(self, img_size: tuple[int, int], vert_lines: list[Line], horiz_lines: list[Line], lattice_points: list[Point]):
        self.img_size: tuple[int, int] = img_size
        self.lattice_points = lattice_points
        self.vert_lines = vert_lines
        self.horiz_lines = horiz_lines
        self.color1: tuple[int, int, int] = (22, 173, 61)  # g
        self.color2: tuple[int, int, int] = (180, 130, 70)  # b

        self.grid: list[Line] = []
        self.bp: list[Point] = []

        self._get_board_center()
        if BoardGrid.frame_ind == BoardGrid.number_of_frame - 1:
            self._get_const_border_info()
        self._get_border_info()
        BoardGrid.frame_ind += 1
        BoardGrid.frame_ind %= BoardGrid.number_of_frame

    def _get_const_border_info(self):
        border_info = []
        for border in BoardGrid.borders_info:
            x, y, k, ratio = 0, 0, 0, 0
            for ind, vals in enumerate(border):
                if vals is None:
                    vals = [1, 1, 1, 1]
                x, y, k, ratio = x + vals[0], y + vals[1], k + vals[2], ratio + vals[3]
            if len(border) != 0:
                x, y, k, ratio = round(x / len(border)), round(y / len(border)), k / len(border), ratio / len(border)
            border_info.append((x, y, k, ratio))
        BoardGrid.const_grid, _ = self._get_border_lines_and_centers(border_info, BoardGrid.const_board_center)
        ratio_list: list[float] = self._pull_ratio_from_border_info(border_info)
        BoardGrid.const_vert_lines, BoardGrid.const_horiz_lines = self._get_all_grid(BoardGrid.const_grid, ratio_list)
        BoardGrid.const_grid = BoardGrid.const_vert_lines + BoardGrid.const_horiz_lines
        BoardGrid.const_img_size = self.img_size
        if BoardGrid.const_grid is None:
            BoardGrid.const_grid = []
        BoardGrid.borders_info = [[], [], [], []]

    def _get_border_info(self):
        if len(self.lattice_points) == 0:
            return
        if BoardGrid.borders_info[0] is None or BoardGrid.borders_info[2] is None:
            return
        if len(self.vert_lines) == 0 or len(self.horiz_lines) == 0:
            return
        center = BoardGrid.board_center_list[-1]
        info = self._get_opposite_borders_info(self.vert_lines, int(len(self.horiz_lines) / 2), 'vert')  # l, r
        info += self._get_opposite_borders_info(self.horiz_lines, int(len(self.vert_lines) / 2), 'horiz')  # u, d
        for ind in range(len(info)):
            BoardGrid.borders_info[ind].append(info[ind])

        border_info = []
        for border in BoardGrid.borders_info:
            border_info.append(border[-1])
        self.grid, self.bp = self._get_border_lines_and_centers(border_info, center)
        if self.grid is None:
            self.grid = []
        ratio_list: list[float] = self._pull_ratio_from_border_info(border_info)
        self.vert_lines, self.horiz_lines = self._get_all_grid(self.grid, ratio_list)
        self.grid = self.horiz_lines + self.vert_lines
        if len(BoardGrid.const_vert_lines) == 0 or len(BoardGrid.const_horiz_lines) == 0:
            BoardGrid.const_vert_lines = self.vert_lines
            BoardGrid.const_horiz_lines = self.horiz_lines
            BoardGrid.const_grid = BoardGrid.const_vert_lines + BoardGrid.const_horiz_lines
            BoardGrid.const_img_size = self.img_size
        # draw_lines(self.img, [self.vert_lines, self.horiz_lines], [self.color1, self.color2])
        # print()

    def _pull_ratio_from_border_info(self, border_info):
        ratio_list: list[float] = []
        for ind, info in enumerate(border_info):
            if info is None:
                info = (1, 1, 1, 1)
            if info[3] == 0:
                ratio_list.append(1)
            elif ind % 2 == 1:
                ratio_list.append(info[3])
            else:
                ratio_list.append(1 / info[3])
        return ratio_list

    def _get_opposite_borders_info(self, parallel_lines: list[Line], perpendicular_lines_ind: int, line_type: str):
        line_ind = perpendicular_lines_ind
        line_points = get_line_points(self.lattice_points, line_ind, line_type)
        direct_points_order = line_points
        reverse_points = line_points[::-1]
        info = []
        info.append(get_border_cord_and_angle(reverse_points, parallel_lines, line_type))
        info.append(get_border_cord_and_angle(direct_points_order, parallel_lines, line_type))
        return info

    def _get_all_grid(self, borders: list[Line], ratio_list: list[float]):
        # print(ratio_list)
        if len(borders) == 0:
            return [], []
        points: list = []
        connected_lines = [(0, 1), (2, 3)]
        for ind, line in enumerate(borders):
            line_type = 'horiz' if ind//2 == 0 else 'vert'
            points.append(self._get_border_lattice_points(line, ratio_list[ind], line_type))

        vert_horiz_lines: list[list[Line]] = []
        for connect in connected_lines:
            lines: list[Line] = []
            for ind in range(len(points[connect[0]])):
                x1, y1 = points[connect[0]][ind]
                x2, y2 = points[connect[1]][ind]
                lines.append(Line(False))
                lines[-1].set_by_raw_line(np.asarray([x1, y1, x2, y2]))
            vert_horiz_lines.append(lines)
        vert_horiz_lines[0] = [borders[2]] + vert_horiz_lines[0] + [borders[3]]
        vert_horiz_lines[1] = [borders[0]] + vert_horiz_lines[1] + [borders[1]]
        # for i in range(len(vert_horiz_lines)):
        #     for ind in range(len(vert_horiz_lines[i])):
        #         vert_horiz_lines[i][ind].set_is_img_size_matter(True)
        return vert_horiz_lines[0], vert_horiz_lines[1]

    def _get_border_lattice_points(self, line: Line, ratio: float, line_type: str):
        points = []
        if line_type == 'horiz':
            start = min(line.p1[0], line.p2[0])
        else:
            start = min(line.p1[1], line.p1[1])
        ratio_power_sum = 1
        for power in range(1, 8):
            ratio_power_sum += ratio ** (-power)
        step = line.line_len / ratio_power_sum
        if line_type == 'horiz':
            step = abs(step * math.cos(line.angle * pi / 180))
            for i in range(1, 8):
                x = start + step
                y = line.k * x + line.b
                start = x
                step /= ratio
                points.append([round(x), round(y)])
        else:
            step = abs(step * math.sin(line.angle * pi / 180))
            for i in range(1, 8):
                y = start + step
                k = line.k if line.k != 0 else 0.0001
                x = (y - line.b) / k
                start = y
                step /= ratio
                points.append([round(x), round(y)])
        return points

    def _get_border_lines_and_centers(self, border_info: list[tuple], center: Point):
        borders = []
        border_points_center = []
        line_order_list = [(0, 1), (2, 3), (0, 2), (1, 3)]
        if border_info[0] is None or border_info[2] is None:
            return [], [], []
        for border in border_info:
            cord = list(border[:2])
            cord[0] += center.x
            cord[1] += center.y
            border_points_center.append(Point(cord))
            borders.append(Line(False))
            borders[-1].set_by_point_k(cord, border[2])
        border_points = find_intersection_lattice_points(borders[2:], borders[:2], self.img_size, True)
        if len(border_points) != 4:
            return borders, border_points_center
        for ind, pnts_ind in enumerate(line_order_list):
            points: np.ndarray = np.asarray(
                [border_points[pnts_ind[0]].x, border_points[pnts_ind[0]].y, border_points[pnts_ind[1]].x,
                 border_points[pnts_ind[1]].y])
            borders[ind].set_by_raw_line(points)
        return borders, border_points_center

    def _get_board_center(self):
        if BoardGrid.frame_ind == BoardGrid.number_of_frame - 1:
            x, y = 0, 0
            for point in BoardGrid.board_center_list:
                x += point.x
                y += point.y
            x = int(x / len(BoardGrid.board_center_list))
            y = int(y / len(BoardGrid.board_center_list))
            BoardGrid.const_board_center = Point((x, y))
            BoardGrid.board_center_list = []

        if len(self.lattice_points) != 0:
            x, y = 0.0, 0.0
            for point in self.lattice_points:
                x += point.x
                y += point.y
            point = Point((int(x / len(self.lattice_points)), int(y / len(self.lattice_points))))
            BoardGrid.board_center_list.append(point)
        elif len(BoardGrid.board_center_list) == 0:
            BoardGrid.board_center_list.append(BoardGrid.const_board_center)

    @staticmethod
    def change_const_grid_size(new_size: tuple[int, int]):
        if new_size[0] == BoardGrid.const_img_size[0] and new_size[1] == BoardGrid.const_img_size[1]:
            return
        size_kx, size_ky = new_size[1] / BoardGrid.const_img_size[1], new_size[0] / BoardGrid.const_img_size[0]
        BoardGrid.const_img_size = new_size
        BoardGrid.const_vert_lines = change_lines_size(BoardGrid.const_vert_lines, size_kx, size_ky)
        BoardGrid.const_horiz_lines = change_lines_size(BoardGrid.const_horiz_lines, size_kx, size_ky)
        BoardGrid.const_grid = BoardGrid.const_vert_lines + BoardGrid.const_horiz_lines

