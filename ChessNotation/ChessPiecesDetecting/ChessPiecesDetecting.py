from bisect import bisect_left, bisect_right
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from ChessNotation.BoardDetecting.BoardGrid import Line
from ChessNotation.BoardDetecting.BoardGrid import BoardGrid
from ChessNotation.ChessPiecesDetecting.ChessPiece import ChessPiece
from ChessNotation.ChessPiecesDetecting.ChessNotation import ChessNotation


def resizing_for_nn(img, h: int, w: int, new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    if new_width is None and new_height is None:
        return img
    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)
    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))
    return cv2.resize(img, dimension, interpolation=interp)


class ChessPiecesDetecting:
    def __init__(self):
        self.board_grid: BoardGrid | None = None
        self.img: np.ndarray | None = None
        self.resized_img: np.ndarray | None = None
        self.chess_pieces: list[ChessPiece] = []
        self.neural_img_shape: tuple[int, int] = (960, 960)
        # self.yolo_model_pieces = YOLO('ChessNotation\\ChessPiecesDetecting\\models\\yolo8n_obb_4_120.pt')
        self.yolo_model_pieces = YOLO('ChessNotation\\ChessPiecesDetecting\\models\\yolo8s960_0_40.pt')# yolo8s960_mix_50, yolo8s960_0_40
        self.boards_list: list[list[list[tuple[int, float]]]] = []
        self.is_first_detecting: bool = True
        self.number_of_transpose: int = 0
        self.number_of_check_frames = 3
        self.check_frames_num = 0
        self.notation = ChessNotation()

    def set_image(self, img: np.ndarray):
        self.img = img
        ChessPiece.img_size = self.img.shape[:2]

        x = round(self.neural_img_shape[1] * self.img.shape[1] / 1440)
        y = round(self.neural_img_shape[0] * self.img.shape[0] / 1080)

        if self.img.shape[0] > self.img.shape[1]:
            self.resized_img = resizing_for_nn(self.img, y, x, new_height=self.neural_img_shape[0])
        else:
            self.resized_img = resizing_for_nn(self.img, y, x, new_width=self.neural_img_shape[1])

    def set_board_grid(self, board_grid: BoardGrid):
        self.board_grid = board_grid
        Line.shape = self.img.shape[:2]
        BoardGrid.change_const_grid_size(self.img.shape[:2])

    def find_chess_pieces_positions(self):
        if self.board_grid is None:
            return
        if self.board_grid.is_hand_under_board:
            return
        self.get_chess_pieces()
        self._fix_position()

    def _fix_position(self):
        if self.board_grid is None:
            return
        if self.check_frames_num == self.number_of_check_frames - 1:
            self.check_frames_num = 0
            result_board = self._find_mean_board()
            result_board = self._transpose_board(result_board)
            self.notation.set_board(result_board)
            self.boards_list = []
            # print(self.notation)
        else:
            self.check_frames_num += 1

    def rotate_board(self):
        self.number_of_transpose = (self.number_of_transpose + 1) % 4
        self.notation.rotate_board()

    def get_chess_pieces(self):

        result = self.yolo_model_pieces(self.resized_img, conf=0.5, verbose=False)[0]
        raw_box = result.obb.xyxyxyxyn.cpu().numpy()
        cls = result.obb.cls.cpu().numpy()
        probs = result.obb.conf.cpu().numpy()

        self.chess_pieces = []
        for ind in range(len(cls)):
            self.chess_pieces.append(ChessPiece(raw_box[ind], cls[ind], probs[ind]))
        board = self.find_chess_pieces_coord_on_board()
        # board = self._transpose_board(board)
        self.boards_list.append(board)

    def _find_mean_board(self):
        mean_board: list[list[int]] = []
        for row in range(8):
            mean_board.append([])
            for col in range(8):
                pieces_list: list[int] = [self.boards_list[ind][row][col][0] for ind in range(len(self.boards_list))]
                piece = get_most_frequent_item(pieces_list)
                mean_board[-1].append(piece)
        return mean_board

    def _transpose_board(self, board):
        if self.is_first_detecting:
            self.number_of_transpose = self._get_number_of_transpose(board)
            self.is_first_detecting = False
        return np.rot90(board, k=self.number_of_transpose).tolist()

    @staticmethod
    def _get_number_of_transpose(board):
        king_coord = (0, 0)
        is_find = False
        for row in range(8):
            for col in range(8):
                if board[row][col] == 1:  # k
                    king_coord = (row, col)
                    is_find = True
                    break
            if is_find:
                break
        border_dists = [(0, king_coord[0]), (1, 7 - king_coord[1]), (2, 7 - king_coord[0]), (3, king_coord[1])]
        number_of_transpose = min(border_dists, key=lambda x: x[1])[0]
        return number_of_transpose

    def find_chess_pieces_coord_on_board(self):
        board = self.get_empty_board()
        for chess_piece in self.chess_pieces:
            x, y = chess_piece.coord
            pos_x = bisect_left(BoardGrid.const_vert_lines, x, key=lambda ln: (y - ln.b) / ln.k)
            pos_y = bisect_left(BoardGrid.const_horiz_lines, y, key=lambda ln: ln.k * x + ln.b)
            if 0 < pos_x < 9 and 0 < pos_y < 9:
                if board[pos_y - 1][pos_x - 1][0] != -1 and board[pos_y - 1][pos_x - 1][1] > chess_piece.prob:
                    continue
                board[pos_y - 1][pos_x - 1] = (chess_piece.class_num, chess_piece.prob)
        return board

    def draw_detect_chess_pieces(self, is_board_draw=True, is_piece_draw=True, is_wait=True):
        tmp_img = self.get_detect_chess_pieces_img(is_board_draw, is_piece_draw)
        cv2.imshow('detect', tmp_img)
        if is_wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_detect_chess_pieces_img(self, is_board_draw=True, is_piece_draw=True):
        tmp_img = self.img.copy()
        if is_board_draw:
            color = (180, 130, 70)
            for line in BoardGrid.const_grid:
                line.set_is_img_size_matter(True)
                x1, y1 = line.p1
                x2, y2 = line.p2
                if 0 <= x1 <= tmp_img.shape[1] and 0 <= y1 <= tmp_img.shape[0]:
                    cv2.line(tmp_img, (x1, y1), (x2, y2), color, 2)

        if is_piece_draw:
            for piece in self.chess_pieces:
                color = ChessPiece.classes[piece.class_num][1]
                cl_name = ChessPiece.classes[piece.class_num][0]
                prob_str = f"{piece.prob:.3f}"
                tmp_img = cv2.polylines(tmp_img, [piece.box], isClosed=True, color=color, thickness=2)
                cv2.putText(tmp_img, cl_name, (piece.coord[0], piece.coord[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(tmp_img, prob_str, (piece.coord[0] - 20, piece.coord[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return tmp_img

    def get_2d_chess(self):
        return self.notation.chess_2d_img

    @staticmethod
    def get_empty_board():
        board = []
        for i in range(8):
            board.append([(-1, 0) for _ in range(8)])
        return board


def get_most_frequent_item(lst):
    occurence_count = Counter(lst)
    return occurence_count.most_common(1)[0][0]
