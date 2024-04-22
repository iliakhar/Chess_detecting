from os import getcwd
from LinesGroups import *
from Point import *
from lattice_points_ml.ConvNet import ConvNet
from LatticeDetectFuncs import *


class ChessBoardDetecting:

    def __init__(self):
        self.img: np.ndarray | None = None
        self.gray_img: np.ndarray | None = None
        # self.mono_image: np.ndarray | None = None
        self.blur_koef: int = 3
        self.density_koef: float = 0.08
        self.img_min_size: int = 350

        self.conv_model: ConvNet = ConvNet()
        self.conv_model = self.conv_model.to(self.conv_model.device)
        self.conv_model.load_model(getcwd() + '\\lattice_points_ml\\model\\model_6.pt')

        self.lines: LinesGroups = LinesGroups(self.blur_koef, self.density_koef)
        self.intersection_points: list[Point] = []
        self.lattice_points: list[Point] = []
        self.border_points: list[Point] = []

        self.board_center: Point = Point((0, 0))

        self.start_ok = 783
        self.start_no = 154
        self.start_border = 82

    def set_image(self, img: np.ndarray):
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

    def detect_board(self) -> None:
        if type(self.img) is not np.ndarray:
            return
        self.lines = LinesGroups(self.blur_koef, self.density_koef)
        self.lines.find_lines(self.img)
        # _, self.img = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_OTSU)
        self.intersection_points = find_intersection_points(self.lines.result_lines)

        # self.save_lattice_points_img()

        self.lattice_points = self.get_lattice_points()

    def get_lattice_points(self) -> list[Point]:
        lattice_points: list[Point] = []
        # g_img = cv2.GaussianBlur(self.gray_img, (5, 5), 0)
        for point in self.intersection_points:
            edges = get_point_neighborhood(self.gray_img, point)
            if type(edges) is np.ndarray:
                if edges.shape == (21, 21):
                    if self.conv_model.predict_model(edges) == 1:
                        lattice_points.append(point)

        horiz_lines: set = set()
        vert_lines: set = set()
        for point in lattice_points:
            horiz_lines.add(self.lines.result_lines[point.line_ind_h])
            vert_lines.add(self.lines.result_lines[point.line_ind_v])

        horiz_lines_lst = exclude_the_wrong_lines(list(horiz_lines), 20)
        vert_lines_lst = exclude_the_wrong_lines(list(vert_lines), 20)
        # print(horiz_lines_lst)
        # print(vert_lines_lst)

        # for h in horiz_lines_lst:
        #     print(self.lines.result_lines[h])
        # print('\n')
        # for v in vert_lines_lst:
        #     print(self.lines.result_lines[v])

        # color1: tuple[int, int, int] = (22, 173, 61)
        # color2: tuple[int, int, int] = (180, 130, 70)
        # draw_lines(self.img, [horiz_lines_lst, vert_lines_lst], [color1, color2])
        lattice_points = find_intersection_lattice_points(self.img, horiz_lines_lst, vert_lines_lst, self.gray_img.shape)
        lattice_points = fill_missing_points(self.img, lattice_points, horiz_lines_lst, vert_lines_lst)
        self.board_center = self.get_board_center(lattice_points)

        return lattice_points

    def save_lattice_points_img(self):
        color: tuple[int, int, int] = (22, 173, 61)
        draw_points(self.img, [self.intersection_points], [color], 'fasfa')
        print('Len:', len(self.intersection_points))
        filename_ok = getcwd() + '\\lattice_points_ml\\latchess21\\ok_train_my\\'
        filename_no = getcwd() + '\\lattice_points_ml\\latchess21\\no_train_my1\\'
        filename_border = getcwd() + '\\lattice_points_ml\\latchess21\\border_train_my1\\'
        for point in self.intersection_points:
            edges = get_point_neighborhood(self.gray_img, point)
            if type(edges) == np.ndarray:
                if edges.shape == (21, 21):
                    draw_points(self.img, [[point]], [(22, 173, 61)], 'img0', False)
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

    def get_board_center(self, lattice_points):
        if len(lattice_points) == 0:
            return Point((5, 5))

        x, y = 0.0, 0.0
        for point in lattice_points:
            x += point.x
            y += point.y
        return Point((int(x / len(lattice_points)), int(y / len(lattice_points))))

    def show_lines(self, is_wait: bool = True, color: tuple[int, int, int] = (22, 173, 61)):
        # clr = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(self.lines.result_lines))]
        draw_lines(self.img, [self.lines.result_lines], [color], is_wait)
        # draw_lines(self.img, [self.lines.result_lines], clr, is_wait)

    def show_all_points(self, is_wait: bool = True, img_name: str = 'image1',
                        color: tuple[int, int, int] = (22, 173, 61)):
        draw_points(self.img, [self.intersection_points], [color], img_name, is_wait)

    def show_lattice_points(self, is_wait: bool = True, img_name: str = 'image2',
                            color: tuple[int, int, int] = (180, 130, 70)):
        color1: tuple[int, int, int] = (0, 0, 139)
        draw_points(self.img, [self.lattice_points, [self.board_center]], [color, color1], img_name, is_wait)

    def show_grupped_points(self, is_wait: bool = True, img_name: str = 'image3'):
        color1: tuple[int, int, int] = (22, 173, 61)
        color2: tuple[int, int, int] = (180, 130, 70)
        # color3: tuple[int, int, int] = (0, 123, 255)
        color4: tuple[int, int, int] = (0, 0, 139)
        colors = [color1, color2, color4]
        points = [self.intersection_points, self.lattice_points, [self.board_center]]
        draw_points(self.img, points, colors, img_name, is_wait)
