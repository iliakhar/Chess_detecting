import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ChessNotation.BoardDetecting.BoardGrid import BoardGrid
from ChessNotation.BoardDetecting.ChessBoardDetecting import ChessBoardDetecting
from ChessNotation.ChessPiecesDetecting.ChessPiecesDetecting import ChessPiecesDetecting
from ChessNotation.Resizing import resizing


class VideoThread(QThread):
    signal_change_pixmap = pyqtSignal(np.ndarray, np.ndarray)
    signal_change_notation = pyqtSignal(int, str, str)

    def __init__(self):
        super().__init__()
        self.board_detect: ChessBoardDetecting = ChessBoardDetecting()
        self.chess_detect: ChessPiecesDetecting = ChessPiecesDetecting()
        self.media_name = ''
        self.media_type: str = 'video'
        self.is_show_board = True
        self.is_show_pieces = True
        self.is_video_stop = False
        self.is_check_once = False
        self.is_search_board = True

    def stop_running(self):
        """Terminate the thread."""
        self.terminate()
        self.wait()

    def run(self):
        self.board_detect = ChessBoardDetecting()
        self.chess_detect = ChessPiecesDetecting()

        if self.media_type == 'video':
            self.video_processing()
        else:
            self.img_processing()


    def video_processing(self):
        media_name = int(self.media_name) if self.media_name.isdigit() else self.media_name
        cap = cv2.VideoCapture(media_name)
        ret, frame = cap.read()
        self.board_detect.set_image(frame)
        board_grid: BoardGrid = self.board_detect.detect_board()
        frame_tmp = frame
        while True:
            if not self.is_video_stop:
                ret, frame_tmp = cap.read()
            if frame_tmp is not None:
                frame = frame_tmp

            if self.is_check_once:
                if self.is_search_board:
                    self.board_detect.set_image(frame)
                    board_grid = self.board_detect.detect_board()
                    if BoardGrid.if_const_border_find:
                        self.is_search_board = False
                        BoardGrid.if_const_border_find = False
                little_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
                board_grid.check_for_hand(little_frame, 'const')
            else:
                self.board_detect.set_image(frame)
                board_grid: BoardGrid = self.board_detect.detect_board()

            move_describe = self.find_chess_pieces(frame, board_grid)
            if move_describe == 0 or move_describe == 1:
                move_list = self.chess_detect.notation.alg_not_list
                move = move_list[-1][0] if move_describe == 0 else move_list[-1][0] + '\t' + move_list[-1][1]
                move = str(len(move_list)) + '. ' + move
                fen = self.chess_detect.notation.get_fen_notation()
                self.signal_change_notation.emit(move_describe, move, fen)
                self.chess_detect.notation.move_describe = -1
            elif move_describe == 2:
                fen = self.chess_detect.notation.get_fen_notation()
                self.signal_change_notation.emit(0, '', fen)
                self.chess_detect.notation.move_describe = -1

    def img_processing(self):
        frame = cv2.imread(self.media_name)
        self.board_detect.set_image(frame)
        board_grid: BoardGrid = self.board_detect.detect_board()
        while True:
            if self.is_search_board:
                self.board_detect.set_image(frame)
                board_grid = self.board_detect.detect_board()
                if BoardGrid.if_const_border_find:
                    self.is_search_board = False
                    BoardGrid.if_const_border_find = False

            move_describe = self.find_chess_pieces(frame, board_grid)
            if move_describe != -1:
                fen = self.chess_detect.notation.get_fen_notation()
                self.signal_change_notation.emit(0, '', fen)
                self.chess_detect.notation.move_describe = -1

    def find_chess_pieces(self, frame, board_grid):
        self.chess_detect.set_image(frame)
        if BoardGrid.is_board_fixed:
            self.chess_detect.set_board_grid(board_grid)
        self.chess_detect.find_chess_pieces_positions()
        detect_img = self.chess_detect.get_detect_chess_pieces_img(self.is_show_board, self.is_show_pieces, )
        chess_2d_img = self.chess_detect.get_2d_chess()
        self.signal_change_pixmap.emit(detect_img, chess_2d_img)
        move_describe = self.chess_detect.notation.move_describe
        return move_describe