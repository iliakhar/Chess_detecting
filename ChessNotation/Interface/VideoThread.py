import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ChessNotation.BoardDetecting.BoardGrid import BoardGrid
from ChessNotation.BoardDetecting.ChessBoardDetecting import ChessBoardDetecting
from ChessNotation.ChessPiecesDetecting.ChessPiecesDetecting import ChessPiecesDetecting


class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.board_detect: ChessBoardDetecting = ChessBoardDetecting()
        self.chess_detect: ChessPiecesDetecting = ChessPiecesDetecting()
        self.media_name = ''
        self.is_run = True

    def stop_running(self):
        """Terminate the thread."""
        self.terminate()
        self.wait()

    def run(self):
        media_name = int(self.media_name) if self.media_name.isdigit() else self.media_name
        cap = cv2.VideoCapture(media_name)
        self.is_run = True
        while self.is_run:
            ret, frame = cap.read()
            if frame is None:
                continue
            self.board_detect.set_image(frame)
            board_grid: BoardGrid = self.board_detect.detect_board()

            self.chess_detect.set_image(frame)
            if BoardGrid.is_board_fixed:
                self.chess_detect.set_board_grid(board_grid)
            self.chess_detect.find_chess_pieces_positions()
            detect_img = self.chess_detect.get_detect_chess_pieces_img(is_wait=False, is_piece_draw=True)
            chess_2d_img = self.chess_detect.get_2d_chess()
            self.change_pixmap.emit(detect_img, chess_2d_img)

