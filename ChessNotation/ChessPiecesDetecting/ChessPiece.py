import cv2
import numpy as np

from ChessNotation.BoardDetecting.Line import Line


class ChessPiece:
    classes = [('b', (0, 255, 206), '♗'), ('k', (254, 0, 86), '♔'), ('n', (255, 128, 0), '♘'),
               ('p', (134, 34, 255), '♙'), ('q', (14, 122, 254), '♕'), ('r', (0, 183, 235), '♖'),
               ('B', (0, 0, 255), '♝'), ('K', (139, 0, 139), '♚'), ('N', (72, 61, 139), '♞'),
               ('P', (255, 171, 171), '♟'), ('Q', (128, 128, 0), '♛'), ('R', (160, 82, 45), '♜')]
    img_size: tuple[int, int] = (-1, -1)

    def __init__(self, raw_box, class_num: int, prob: float):
        self.class_num: int = int(class_num)
        self.prob: float = prob
        self.coord: tuple[int, int] = (-1, -1)
        self.box: np.array = np.array([])
        self.find_box_and_coord_by_raw_box(raw_box)

    def find_box_and_coord_by_raw_box(self, raw_box):
        img_size = ChessPiece.img_size
        box: list = []
        for pnt_ind in range(len(raw_box)):
            x1, y1 = round(raw_box[pnt_ind][0] * img_size[1]), round(raw_box[pnt_ind][1] * img_size[0])
            box.append([x1, y1])
        self.coord = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2 - 10)
        self.box = np.array(box)

    def __str__(self):
        info: str = f'box: {self.box.tolist()} \nclass: {ChessPiece.classes[self.class_num][0]} \n'
        info += f'prob: {self.prob} \nposition: {self.coord} \n'
        return info
