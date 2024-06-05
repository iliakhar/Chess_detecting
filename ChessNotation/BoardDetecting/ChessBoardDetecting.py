import uuid
from os import getcwd

from pygments.formatters import img

from ChessNotation.BoardDetecting.LinesGroups import *
from ChessNotation.BoardDetecting.Point import *
from lattice_points_ml.ConvNet import ConvNet
from ChessNotation.BoardDetecting.LatticePoints import LatticePoints
from ChessNotation.BoardDetecting.BoardGrid import BoardGrid
from ultralytics import YOLO

class ChessBoardDetecting:

    def __init__(self):
        self.img: np.ndarray | None = None
        self.chessboard_img: np.ndarray | None = None
        self.blur_koef: int = 3
        self.density_p: float = 0.3
        self.img_min_size: int = 416
        self.neural_img_shape: tuple[int, int] = (416, 416)
        self.board_detect_model = YOLO('ChessNotation\\BoardDetecting\\models\\board_detect.pt')

        self.conv_model: ConvNet = ConvNet()
        self.conv_model = self.conv_model.to(self.conv_model.device)
        self.conv_model.load_model(getcwd() + '\\lattice_points_ml\\model\\lattice_rotate_detect_6_100.pt')  # model_40_150.pt model_bigger_100
        LatticePoints.conv_model = self.conv_model

        self.lines: LinesGroups = LinesGroups(self.blur_koef, self.density_p)
        self.intersection_points: list[Point] = []
        self.lattice_points: LatticePoints | None = None
        self.board_grid: BoardGrid | None = None

    def set_image(self, img: np.ndarray):
        self.border_points = []
        # self.img = img
        if type(img) is not np.ndarray:
            return

        x = round(self.neural_img_shape[1] * img.shape[1] / 1440)
        y = round(self.neural_img_shape[0] * img.shape[0] / 1080)

        if img.shape[0] > img.shape[1]:
            self.img = resizing_for_nn(img, y, x, new_height=self.neural_img_shape[0])
        else:
            self.img = resizing_for_nn(img, y, x, new_width=self.neural_img_shape[1])

        Line.shape = self.img.shape[:2]
        # self.gray_img = self.img
        # _, self.mono_image = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)

    def get_chessboard_img(self):
        result = self.board_detect_model(self.img, conf=0.5, verbose=False)[0]

        borders = result.boxes.xyxyn.cpu().numpy()
        if len(borders) == 0:
            return (0,0,0,0), []
        borders_xywh = result.boxes.xywh.cpu().numpy()
        board_ind = 0
        max_area = 0
        for ind, border in enumerate(borders_xywh):
            if border[2]*border[3] > max_area:
                max_area = border[2]*border[3]
                board_ind = ind

        x1, y1 = round(borders[board_ind][0] * self.img.shape[1]), round(borders[board_ind][1] * self.img.shape[0])
        x2, y2 = round(borders[board_ind][2] * self.img.shape[1]), round(borders[board_ind][3] * self.img.shape[0])
        return (x1, y1, x2, y2), self.img[y1:y2, x1:x2]

    def detect_board(self) -> BoardGrid | None:

        cords, chessboard_img = self.get_chessboard_img()
        if type(chessboard_img) is not np.ndarray:
            return None
        self.lines = LinesGroups(self.blur_koef, self.density_p)
        self.lines.find_lines(chessboard_img)
        # _, self.img = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_OTSU)
        self.intersection_points = find_intersection_points(chessboard_img, self.lines.result_lines)
        # draw_points(self.img, [self.intersection_points], [(22, 173, 61)])
        # self.save_lattice_points_img(chessboard_img)

        self.lattice_points = LatticePoints(chessboard_img, self.intersection_points, self.lines)
        self.lattice_points.shift_points_and_lines(cords[0], cords[1])
        lp = self.lattice_points
        self.board_grid = BoardGrid(self.img, lp.vert_lines, lp.horiz_lines, lp.lattice_points)

        for ind in range(len(self.intersection_points)):
            self.intersection_points[ind].x += cords[0]
            self.intersection_points[ind].y += cords[1]

        return self.board_grid
        # BoardGrid.change_const_grid_size((self.img.shape[0]//2, self.img.shape[1]//2))

    def save_lattice_points_img(self, chessboard_img: np.ndarray):
        color: tuple[int, int, int] = (22, 173, 61)
        draw_points(chessboard_img, [self.intersection_points], [color], 'fasfa')
        print('Len:', len(self.intersection_points))
        filename_ok = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\ok_0\\'
        filename_no = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\no_0\\'
        filename_border = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\3\\border_0\\'
        self.intersection_points.sort(key=lambda x: x.y, reverse=True)
        gray_img = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
        for point in self.intersection_points:
            # edges = get_point_neighborhood(self.gray_img, point)
            x1, y1 = point.x - 10, point.y - 10
            edges = gray_img[y1:y1 + 21, x1:x1 + 21]
            color = (22, 173, 61)  # another
            if type(edges) is np.ndarray:
                if edges.shape == (21, 21):
                    predicted_val = self.conv_model.predict_model(edges)
                    if predicted_val == 1:
                        color = (180, 130, 70)  # lattice
                    if predicted_val == 2:
                        color = (0, 123, 255)  # border
                    draw_points(chessboard_img, [[point]], [color], 'img0', False)
                    cv2.imshow('image', edges)
                    pressed_key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if pressed_key == ord('1'):
                        cv2.imwrite(filename_ok + str(uuid.uuid4()) + '.jpg', edges)
                    elif pressed_key == ord('2'):
                        cv2.imwrite(filename_no + str(uuid.uuid4()) + '.jpg', edges)
                    elif pressed_key == ord('3'):
                        cv2.imwrite(filename_border + str(uuid.uuid4()) + '.jpg', edges)
                    elif pressed_key == ord('q'):
                        break

    def show_lines(self, is_wait: bool = True):
        color1: tuple[int, int, int] = (22, 173, 61)
        # color2: tuple[int, int, int] = (255, 255, 0)
        colors = [color1]
        draw_lines(self.img, [self.lines.result_lines], colors, is_wait)
        # draw_lines(self.img, [self.lines.result_lines], clr, is_wait)

    def show_borders(self, is_wait: bool = True):
        if self.board_grid is None:
            return
        color1: tuple[int, int, int] = (22, 173, 61)
        color2: tuple[int, int, int] = (255, 255, 0)  # const lines
        colors = [color1, color2]
        draw_lines(self.img, [self.board_grid.grid, BoardGrid.const_grid], colors, is_wait, '1')

    def show_all_points(self, is_wait: bool = True, img_name: str = 'image1',
                        color: tuple[int, int, int] = (22, 173, 61)):
        draw_points(self.img, [self.intersection_points], [color], img_name, is_wait)

    def show_lattice_points(self, is_wait: bool = True, img_name: str = 'image2',
                            color: tuple[int, int, int] = (180, 130, 70)):
        color1: tuple[int, int, int] = (0, 0, 139)
        lt = self.lattice_points
        # draw_points(self.img, [lt.lattice_points], [color], img_name, is_wait)
        draw_points(self.img, [lt.lattice_points, [BoardGrid.board_center_list[-1]]], [color, color1], img_name, is_wait)

    def show_grupped_points(self, is_wait: bool = True, img_name: str = 'image3'):
        color1: tuple[int, int, int] = (22, 173, 61)  # all
        color2: tuple[int, int, int] = (180, 130, 70)  # lattice
        color3: tuple[int, int, int] = (0, 0, 139)  # center
        color4: tuple[int, int, int] = (0, 123, 255)  # border
        color5: tuple[int, int, int] = (255, 255, 0)  # const center
        color6: tuple[int, int, int] = (225, 0, 221)
        colors = [color1, color2, color3, color4, color6, color5]
        lt = self.lattice_points
        if lt is None:
            return
        bg = self.board_grid.bp if self.board_grid is not None else []
        points = [self.intersection_points, lt.lattice_points, [BoardGrid.board_center_list[-1]], bg, lt.border_points, [BoardGrid.const_board_center]]
        # points = [lt.lattice_points]
        draw_points(self.img, points, colors, img_name, is_wait)
