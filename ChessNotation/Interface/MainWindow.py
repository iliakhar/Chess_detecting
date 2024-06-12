from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import *
from PyQt6.QtGui import (QFont, QPixmap, QIcon, QAction)
from ChessNotation.BoardDetecting.ChessBoardDetecting import *
from ChessNotation.Interface.LoadDialog import LoadDialog
from ChessNotation.Interface.ImagePresentationWidget import ImagePresentationWidget
from ChessNotation.Interface.ChessMoveWidget import ChessMoveWidget
from ChessNotation.Interface.FENInfoWidget import FENInfoWidget
from ChessNotation.Interface.UsefulFuncs import *
from ChessNotation.Interface.VideoThread import VideoThread


class MainWindow(QWidget):
    signal_stop_running = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.big_font = QFont('Arial', 13)
        self.video_thread = VideoThread(self.signal_stop_running)

        self.workspace = ImagePresentationWidget()
        self.chess_move_widget = ChessMoveWidget()
        self.fen_widget = FENInfoWidget()
        self.alg_path = ''
        self.fen_path = ''

        self.InitUi()

    def InitUi(self):
        self.setWindowTitle('Chess detector')
        self.setWindowIcon(QIcon('icon/2.png'))
        self.setMinimumSize(1100, 900)

        self._create_menu()

        main_h_lay = QHBoxLayout()
        info_v_lay = QVBoxLayout()
        workspace_v_lay = QVBoxLayout()

        info_frame = create_frame(info_v_lay)
        info_frame.setFixedWidth(400)
        workspace_frame = create_frame(workspace_v_lay)

        workspace_v_lay.addWidget(self.workspace)
        info_v_lay.addWidget(self.chess_move_widget, 2)
        info_v_lay.addWidget(self.fen_widget, 1)
        main_h_lay.setContentsMargins(5, 30, 5, 5)
        main_h_lay.addWidget(workspace_frame, 3)
        main_h_lay.addWidget(info_frame, 1)
        self.setLayout(main_h_lay)

        self.video_thread.signal_change_pixmap.connect(self.set_image)
        self.video_thread.signal_change_notation.connect(self.change_notation)
        self.workspace.signal_check_board.connect(self.check_board_changed)
        self.workspace.signal_check_pieces.connect(self.check_pieces_changed)
        self.workspace.signal_stop_video.connect(self.stop_video)
        self.workspace.signal_find_board_grid.connect(self.find_board_grid)
        self.workspace.signal_rotate_board.connect(self.rotate_board)
        self.chess_move_widget.signal_save_alg_not.connect(self.save_alg_not)

    def _create_menu(self):
        main_menu = QMenuBar(self)
        main_menu.setFont(QFont('Arial', 13))
        main_menu.setFixedHeight(500)

        file_menu = main_menu.addMenu("File")
        file_menu.setFont(QFont('Arial', 11))
        self._create_menu_item(file_menu, "Загрузить видео", "Ctrl+Q", self._load_video)
        self._create_menu_item(file_menu, "Загрузить картинку", "Ctrl+W", self._load_img)
        file_menu.addSeparator()
        self._create_menu_item(file_menu, "Выход", "Ctrl+E", self.close_window)

    def create_new_video_thread(self):
        self.video_thread = VideoThread(self.signal_stop_running)
        self.video_thread.signal_change_pixmap.connect(self.set_image)
        self.video_thread.signal_change_notation.connect(self.change_notation)

    def _create_menu_item(self, menu: QMenu, name: str, shortcut: str, func):
        action = QAction(name, self)
        action.setShortcut(shortcut)
        action.triggered.connect(func)
        menu.addAction(action)

    def change_notation(self, move_describe: int, move: str, fen: str):
        if move != '':
            self.chess_move_widget.change_moves(move_describe, move)
        self.fen_widget.change_fen(fen)

    def rotate_board(self):
        self.video_thread.chess_detect.rotate_board()

    def find_board_grid(self):
        self.video_thread.is_search_board = True

    def stop_video(self, val):
        self.video_thread.is_video_stop = val

    def check_board_changed(self, val):
        self.video_thread.is_show_board = val

    def check_pieces_changed(self, val):
        self.video_thread.is_show_pieces = val

    def _load_video(self):
        media_type = 'video'
        self._load_media(media_type)

    def _load_img(self):
        media_type = 'img'
        self._load_media(media_type)

    def _load_media(self, media_type: str):
        dlg = LoadDialog(media_type)
        if media_type == 'img':
            dlg.setWindowTitle("Загрузка изображения")
        else:
            dlg.setWindowTitle("Загрузка видео")

        dlg.exec()
        if dlg.is_accept:
            print('afasfdaf')
            # self.video_thread.is_run = False
            self.signal_stop_running.emit(False)
            # self.video_thread.stop_running()
            self.create_new_video_thread()
            self.workspace.rotate_btn.setEnabled(True)
            self.workspace.find_board_btn.setEnabled(True)
            self.fen_widget.save_btn.setEnabled(True)
            self.fen_widget.set_media_type(media_type)

            self.video_thread.is_video_stop = False
            self.video_thread.is_search_board = True
            self.workspace.is_video_stop = False
            self.workspace.stop_btn.setText('Стоп')
            self.fen_widget.save_path = dlg.loader_fen_not.get_text()
            self.video_thread.media_name = dlg.loader_media.filename_edit.text()
            self.video_thread.chess_detect.number_of_check_frames = int(dlg.pieces_spinbox.text())
            self.chess_move_widget.moves_list.clear()
            self.fen_widget.filename_ind_fen = 0
            BoardGrid.number_of_frame = int(dlg.board_spinbox.text())

            if media_type == 'img':
                self.workspace.stop_btn.setEnabled(False)
                self.chess_move_widget.save_btn.setEnabled(False)
            else:
                self.workspace.stop_btn.setEnabled(True)
                self.video_thread.is_check_once = dlg.manual_radio_btn.isChecked()
                self.alg_path = dlg.loader_alg_not.get_text()
                self.chess_move_widget.save_btn.setEnabled(True)

            self.video_thread.media_type = media_type
            self.video_thread.start()

    def save_alg_not(self):
        self.video_thread.chess_detect.notation.save_alg_not(self.alg_path)

    def set_image(self, detect_img: np.ndarray, chess_2d_img: np.ndarray):
        self.workspace.set_images(detect_img, chess_2d_img)

    def image_processing(self):
        pass

    def close_window(self):
        self.close()
