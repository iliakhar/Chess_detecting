
import cv2
import time

import requests
from numpy.linalg import norm
from ChessNotation.BoardDetecting.ChessBoardDetecting import *
from ChessNotation.ChessPiecesDetecting.ChessPiecesDetecting import *


def brightness(img):
    if len(img.shape) == 3:
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        return np.average(img)


def detect_photo():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()
    chess_detect: ChessPiecesDetecting = ChessPiecesDetecting()

    while True:
        frame = cv2.imread('ChessNotation/source/img/line9.jpg')
        board_detect.set_image(frame)
        board_grid: BoardGrid = board_detect.detect_board()

        chess_detect.set_image(frame)
        chess_detect.set_board_grid(board_grid)
        chess_detect.find_chess_pieces_positions()
        print(chess_detect)
        chess_detect.draw_detect_chess_pieces(is_wait=False, is_piece_draw=True)

        # board_detect.show_borders(False)
        # board_detect.show_grupped_points(False, 'img123')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def detect_video():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()
    chess_detect: ChessPiecesDetecting = ChessPiecesDetecting()
    # url = "http://192.168.0.66:8080/shot.jpg"
    # url = "http://192.168.0.66:8080/video"
    url = 2


    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture('ChessNotation\\source\\video\\video3.mp4')
    # ind = 0
    while True:
        # print('a', ind)
        # ind+=1
        # ind%=100
        st = time.time()
        # response = requests.get(url)
        # frame = np.array(bytearray(response.content), dtype=np.uint8)
        # frame = cv2.imdecode(frame, -1)

        # print('b')

        ret, frame = cap.read()
        if frame is None:
            continue

        # print(frame.shape)

        board_detect.set_image(frame)
        board_grid: BoardGrid = board_detect.detect_board()

        if BoardGrid.is_board_fixed:
            chess_detect.set_image(frame)
            chess_detect.set_board_grid(board_grid)
            chess_detect.find_chess_pieces_positions()
            chess_detect.draw_detect_chess_pieces(is_wait=False, is_piece_draw=True)


        # board_detect.show_borders(False)
        # board_detect.show_grupped_points(False, 'img123')
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

        et = time.time()
        elapsed_time = et - st
        # print('Время исполнения:', elapsed_time, 'секунд')
    # cap.release()
    cv2.destroyAllWindows()


def main():
    # detect_photo()
    detect_video()


if __name__ == '__main__':
    main()
    # l = [0,2,2,3,4,5,5,6,7,8]
    # print(bisect_left(l, 10), bisect_right(l, 10))
