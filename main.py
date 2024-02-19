from ChessBoardDetecting import *


def main():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()

    # img: np.ndarray = cv2.imread('line5.jpg')
    # board_detect.set_image(img)
    # board_detect.detect_board()
    # board_detect.show_lines()
    # board_detect.show_points()

    board_detect.detect_board()
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        board_detect.set_image(frame)
        board_detect.detect_board()
        board_detect.show_lines(False)
        board_detect.show_points(False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
