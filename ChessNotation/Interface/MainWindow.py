from PyQt6.QtWidgets import *
from PyQt6.QtGui import (QFont, QPixmap, QIcon, QAction)
from PyQt6.QtCore import *
import copy
import sys
from PyQt6.QtGui import QStandardItemModel, QStandardItem
import sys

from ChessNotation.BoardDetecting.ChessBoardDetecting import *
from ChessNotation.ChessPiecesDetecting.ChessPiecesDetecting import *

from ChessNotation.Interface.LoadDialog import LoadDialog
from ChessNotation.Interface.ImagePresentationWidget import ImagePresentationWidget
from ChessNotation.Interface.ChessMoveWidget import ChessMoveWidget
from ChessNotation.Interface.FENInfoWidget import FENInfoWidget
from ChessNotation.Interface.UsefulFuncs import *
from ChessNotation.Interface.VideoThread import VideoThread


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.big_font = QFont('Arial', 13)
        self.video_thread = VideoThread()

        self.workspace = ImagePresentationWidget()
        self.chess_move_widget = ChessMoveWidget()
        self.fen_widget = FENInfoWidget()
        self.alg_path = ''
        self.fen_path = ''

        self.InitUi()

    def InitUi(self):
        self.setWindowTitle('Chess detector')
        # self.setWindowIcon(QIcon('icon/table_icon.png'))
        self.setMinimumSize(1100, 900)

        self._create_menu()

        main_h_lay = QHBoxLayout()
        info_v_lay = QVBoxLayout()
        workspace_v_lay = QVBoxLayout()

        info_frame = create_frame(info_v_lay)
        workspace_frame = create_frame(workspace_v_lay)

        workspace_v_lay.addWidget(self.workspace)
        info_v_lay.addWidget(self.chess_move_widget)
        info_v_lay.addWidget(self.fen_widget)
        main_h_lay.setContentsMargins(5, 30, 5, 5)
        main_h_lay.addWidget(workspace_frame, 3)
        main_h_lay.addWidget(info_frame, 1)
        self.setLayout(main_h_lay)

        self.video_thread.change_pixmap.connect(self.set_image)

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

    def _create_menu_item(self, menu: QMenu, name: str, shortcut: str, func):
        action = QAction(name, self)
        action.setShortcut(shortcut)
        action.triggered.connect(func)
        menu.addAction(action)

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
            self.video_thread.is_run = False
            self.video_thread.stop_running()
            self.alg_path = dlg.loader_alg_not.filename_edit.text()
            self.fen_path = dlg.loader_fen_not.filename_edit.text()
            self.video_thread.media_name = dlg.loader_media.filename_edit.text()
            self.video_thread.chess_detect.number_of_check_frames = int(dlg.pieces_spinbox.text())
            BoardGrid.number_of_frame = int(dlg.board_spinbox.text())
            if media_type == 'img':
                self.image_processing()
            else:
                self.video_thread.start()

    def set_image(self, detect_img: np.ndarray, chess_2d_img: np.ndarray):
        self.workspace.set_images(detect_img, chess_2d_img)

    def image_processing(self):
        pass

    def close_window(self):
        self.close()
