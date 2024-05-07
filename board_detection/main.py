import cv2

from ChessBoardDetecting import *


def detect_photo():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()


    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY +
    #                              cv2.THRESH_OTSU)
    # cv2.imshow('adad', thresh1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    while True:
        frame = cv2.imread('board_detection/source/img/line1.jpg')
        # gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, frame = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY +
        #                              cv2.THRESH_OTSU)
        board_detect.set_image(frame)
        board_detect.detect_board()
        # board_detect.show_lines(False)
        # # board_detect.show_all_points(False, 'img')
        # board_detect.show_lattice_points(False, 'img1')
        board_detect.show_borders(False)
        board_detect.show_grupped_points(False, 'img123')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def detect_video():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture('board_detection/source/video/video3.mp4')
    # if cap.isOpened() == False:
    #     print("Error opening video stream or file")

    while True:
        ret, frame = cap.read()
        # gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, frame = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY +
        #                              cv2.THRESH_OTSU)
        board_detect.set_image(frame)
        board_detect.detect_board()
        # board_detect.show_lines(False)
        # # board_detect.show_all_points(False, 'img')
        # board_detect.show_lattice_points(False, 'img1')
        board_detect.show_borders(False)
        board_detect.show_grupped_points(False, 'img123')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    detect_photo()
    # detect_video()


if __name__ == '__main__':
    main()
    # l = [0,1,2,2,3,4,5,5,6,7,8]
    # print(bisect_left(l, 2), bisect_right(l, 5))