from bisect import bisect_left, bisect_right

import cv2
import numpy as np
from ultralytics import YOLO

from ChessNotation.BoardDetecting.BoardGrid import Line
from ChessNotation.BoardDetecting.UsefulFunctions import draw_lines
from ChessNotation.BoardDetecting.BoardGrid import BoardGrid
from ChessNotation.ChessPiecesDetecting.ChessPiece import ChessPiece


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
        self.chess_pieces: list[ChessPiece] = []
        self.neural_img_shape: tuple[int, int] = (640, 640)
        self.yolo_model = YOLO('ChessNotation\\ChessPiecesDetecting\\models\\yolo8obb_650.pt')
        self.board: list[list[int]] = [[]]

    def set_image(self, img: np.ndarray):
        self.img = img
        ChessPiece.img_size = self.img.shape[:2]

    def set_board_grid(self, board_grid: BoardGrid):
        self.board_grid = board_grid
        Line.shape = self.img.shape[:2]
        BoardGrid.change_const_grid_size(self.img.shape[:2])

    def find_chess_pieces_positions(self):
        self.get_chess_pieces()

    def get_chess_pieces(self):
        x = round(self.neural_img_shape[1] * self.img.shape[1] / 1440)
        y = round(self.neural_img_shape[0] * self.img.shape[0] / 1080)
        tmp_img = cv2.resize(self.img.copy(), (x, y), interpolation=cv2.INTER_LINEAR)
        result = self.yolo_model(tmp_img, conf=0.5, verbose=False)[0]

        raw_box = result.obb.xyxyxyxyn.cpu().numpy()
        cls = result.obb.cls.cpu().numpy()
        probs = result.obb.conf.cpu().numpy()

        self.chess_pieces = []
        for ind in range(len(cls)):
            self.chess_pieces.append(ChessPiece(raw_box[ind], cls[ind], probs[ind]))
        self.find_chess_pieces_coord_on_board()

    def find_chess_pieces_coord_on_board(self):
        self.fill_empty_board()
        # draw_lines(self.img, [BoardGrid.const_horiz_lines, BoardGrid.const_vert_lines], [(180, 130, 70), (22, 173, 61)])
        for chess_piece in self.chess_pieces:
            x, y = chess_piece.coord
            pos_x = bisect_left(BoardGrid.const_vert_lines, x, key=lambda ln: (y - ln.b) / ln.k)
            pos_y = bisect_left(BoardGrid.const_horiz_lines, y, key=lambda ln: ln.k * x + ln.b)
            if 0 < pos_x < 9 and 0 < pos_y < 9:
                self.board[pos_y-1][pos_x-1] = chess_piece.class_num

    def draw_detect_chess_pieces(self, is_board_draw=True, is_piece_draw=True, is_wait=True):
        tmp_img = self.img.copy()
        if is_board_draw:
            color = (180, 130, 70)
            for line in BoardGrid.const_grid:
                x1, y1 = line.p1
                x2, y2 = line.p2
                if 0 < x1 < tmp_img.shape[1] and 0 < y1 < tmp_img.shape[0]:
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
        cv2.imshow('detect', tmp_img)
        if is_wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def fill_empty_board(self):
        self.board = []
        for i in range(8):
            self.board.append([-1 for _ in range(8)])

    def __str__(self):
        str_board: str = ''
        for y in range(8):
            even = int(y % 2 == 0)
            for x in range(8):
                if self.board[y][x] == -1:
                    str_board += '⛂' if x % 2 == even else '⛀'
                else:
                    str_board += ChessPiece.classes[self.board[y][x]][2]
            str_board += '\n'
        return str_board
