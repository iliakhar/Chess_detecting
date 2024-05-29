from os import getcwd
from ChessNotation.BoardDetecting.LinesGroups import *
from ChessNotation.BoardDetecting.Point import *
from lattice_points_ml.ConvNet import ConvNet
from ChessNotation.BoardDetecting.LatticePoints import LatticePoints
from ChessNotation.BoardDetecting.BoardGrid import BoardGrid

class ChessBoardDetecting:

    def __init__(self):
        self.img: np.ndarray | None = None
        self.gray_img: np.ndarray | None = None
        # self.mono_image: np.ndarray | None = None
        self.blur_koef: int = 3
        self.density_p: float = 0.3
        self.img_min_size: int = 450

        self.conv_model: ConvNet = ConvNet()
        self.conv_model = self.conv_model.to(self.conv_model.device)
        self.conv_model.load_model(getcwd() + '\\lattice_points_ml\\model\\model_bigger1_300.pt')  # model_40_150.pt model_bigger_100
        LatticePoints.conv_model = self.conv_model

        self.lines: LinesGroups = LinesGroups(self.blur_koef, self.density_p)
        self.intersection_points: list[Point] = []
        self.lattice_points: LatticePoints | None = None
        self.board_grid: BoardGrid | None = None
        # self.border_points: list[Point] = []

        self.start_ok = 1706
        self.start_no = 2395
        self.start_border = 1570

    def set_image(self, img: np.ndarray):
        self.border_points = []
        # self.img = img
        if type(img) is not np.ndarray:
            return

        if img.shape[0] > img.shape[1]:
            self.img = resizing(img=img, new_width=self.img_min_size)
        else:
            self.img = resizing(img=img, new_height=self.img_min_size)
        Line.shape = self.img.shape[:2]
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # self.gray_img = self.img
        # _, self.mono_image = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)

    def detect_board(self) -> BoardGrid | None:
        if type(self.img) is not np.ndarray:
            return None
        self.lines = LinesGroups(self.blur_koef, self.density_p)
        self.lines.find_lines(self.img)
        # _, self.img = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_OTSU)
        self.intersection_points = find_intersection_points(self.img, self.lines.result_lines)
        # draw_points(self.img, [self.intersection_points], [(22, 173, 61)])
        # self.save_lattice_points_img()

        self.lattice_points = LatticePoints(self.img, self.intersection_points, self.lines)
        lp = self.lattice_points
        self.board_grid = BoardGrid(self.img.shape[:2], lp.vert_lines, lp.horiz_lines, lp.lattice_points)
        return self.board_grid
        # BoardGrid.change_const_grid_size((self.img.shape[0]//2, self.img.shape[1]//2))

    def save_lattice_points_img(self):
        color: tuple[int, int, int] = (22, 173, 61)
        draw_points(self.img, [self.intersection_points], [color], 'fasfa')
        print('Len:', len(self.intersection_points))
        filename_ok = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\ok_points_train\\'
        filename_no = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\no_points_train\\'
        filename_border = getcwd() + '\\lattice_points_ml\\latchess21\\photos\\border_points_train\\'
        self.intersection_points.sort(key=lambda x: x.y, reverse=True)
        for point in self.intersection_points:
            # edges = get_point_neighborhood(self.gray_img, point)
            x1, y1 = point.x - 10, point.y - 10
            edges = self.gray_img[y1:y1 + 21, x1:x1 + 21]
            color = (22, 173, 61)  # another
            if type(edges) is np.ndarray:
                if edges.shape == (21, 21):
                    predicted_val = self.conv_model.predict_model(edges)
                    if predicted_val == 1:
                        color = (180, 130, 70)  # lattice
                    if predicted_val == 2:
                        color = (0, 123, 255)  # border
                    draw_points(self.img, [[point]], [color], 'img0', False)
                    cv2.imshow('image', edges)
                    pressed_key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if pressed_key == ord('1'):
                        cv2.imwrite(filename_ok + str(self.start_ok) + '.jpg', edges)
                        self.start_ok += 1
                    elif pressed_key == ord('2'):
                        cv2.imwrite(filename_no + str(self.start_no) + '.jpg', edges)
                        self.start_no += 1
                    elif pressed_key == ord('3'):
                        cv2.imwrite(filename_border + str(self.start_border) + '.jpg', edges)
                        self.start_border += 1
                    elif pressed_key == ord('q'):
                        break

    def show_lines(self, is_wait: bool = True):
        color1: tuple[int, int, int] = (22, 173, 61)
        # color2: tuple[int, int, int] = (255, 255, 0)
        colors = [color1]
        draw_lines(self.img, [self.lines.result_lines], colors, is_wait)
        # draw_lines(self.img, [self.lines.result_lines], clr, is_wait)

    def show_borders(self, is_wait: bool = True):
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
        bg = self.board_grid
        points = [self.intersection_points, lt.lattice_points, [BoardGrid.board_center_list[-1]], bg.bp, lt.border_points, [BoardGrid.const_board_center]]
        # points = [lt.lattice_points]
        draw_points(self.img, points, colors, img_name, is_wait)
