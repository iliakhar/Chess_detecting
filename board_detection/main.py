from ChessBoardDetecting import *
# from lattice_points_ml.ConvNet import *

def detect_photo():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()

    img: np.ndarray = cv2.imread('board_detection/source/img/line5.jpg')

    board_detect.set_image(img)
    board_detect.detect_board()
    board_detect.show_lines(False)
    board_detect.show_all_points(False)
    board_detect.show_lattice_points()
    # board_detect.show_points()

def detect_video():
    board_detect: ChessBoardDetecting = ChessBoardDetecting()
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture('board_detection/source/video/video3.mp4')
    # if cap.isOpened() == False:
    #     print("Error opening video stream or file")
    skip_frame: int = 0
    frame_num: int = 0

    while True:
        ret, frame = cap.read()
        board_detect.set_image(frame)
        # cv2.imshow('image0', frame)
        if frame_num == 0:
            board_detect.detect_board()
            board_detect.show_lines(False)
            # board_detect.show_all_points(False, 'img')
            # board_detect.show_lattice_points(False, 'img1')
            board_detect.show_grupped_points(False, 'img1')
        frame_num += 1
        if frame_num >= skip_frame:
            frame_num = 0


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # detect_photo()
    detect_video()


if __name__ == '__main__':
    main()