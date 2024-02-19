from ChessBoardDetecting import *


def main():
    img: np.ndarray = cv2.imread('line5.jpg')
    # img: np.ndarray = cv2.imread('line5.jpg')
    board_detect: ChessBoardDetecting = ChessBoardDetecting()
    board_detect.set_image(img)
    board_detect.detect_board()
    draw_lines(board_detect.img, board_detect.result_lines, board_detect.colors_list)

    board_detect.detect_board()
    # cap = cv2.VideoCapture(1)
    # while True:
    #     ret, frame = cap.read()
    #     board_detect.set_image(frame)
    #     board_detect.detect_board()
    #     draw_lines(board_detect.img, board_detect.result_lines, board_detect.colors_list, False)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
