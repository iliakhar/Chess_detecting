import cv2
import numpy as np

from ChessNotation.ChessPiecesDetecting.ChessPiece import ChessPiece


class ChessNotation:
    def __init__(self):
        self.col_names: list[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.move_describe: int = -1  #  -1 - не было хода, 0 - ход белых, 1 - ход черных, 2 - fen без хода
        self.alg_not: str = ''
        self.fen_not: str = ''
        self.alg_not_list: list[list[str]] = []
        self.en_passant: str = '-'
        self.move_counter: int = 1
        self.half_move: int = 0
        self.cur_player: str = 'w'
        self.board: list[list[int]] = []
        self.b_castling = 'kq'
        self.w_castling = 'KQ'
        self.path_2d_pieces: str = 'ChessNotation\\ChessPiecesDetecting\\board_and_pieces\\'
        self.chess_2d_img: np.ndarray = cv2.imread(self.path_2d_pieces + 'board.png')
        self.filename_ind_alg = 0

    def set_board(self, board: list[list[int]]):
        if len(self.board) == 0:
            self.board = board.copy()
            print(self)
            self._get_2d_chess()
            self.fen_not = self.get_fen_notation()
            self.move_describe = 2
            return
        cur_move: str = ''
        if self.w_castling != '-' or self.b_castling != '-':
            cur_move = self._check_for_castling(board)
            if cur_move != '':
                self.board = board.copy()
                self.make_move(cur_move)
                return

        cur_move += self._check_for_movement(board)
        if cur_move != '' and cur_move != 'err':
            if cur_move[0] == 'P':
                cur_move = cur_move[1:]
            self.board = board.copy()
            self.make_move(cur_move)
        elif cur_move == 'err':
            # pass
            self.board = board.copy()
            self._get_2d_chess()

    def rotate_board(self):
        self.board = np.rot90(self.board, k=1).tolist()
        self.fen_not = self.get_fen_notation()
        self._get_2d_chess()
        self.move_describe = 2

    def get_fen_notation(self):
        fen_not: str = ''
        for row in range(8):
            empty_cell_counter: int = 0
            for col in range(8):
                if self.board[row][col] == -1:
                    empty_cell_counter += 1
                if self.board[row][col] != -1:
                    if empty_cell_counter != 0:
                        fen_not += str(empty_cell_counter)
                        empty_cell_counter = 0
                    fen_not += ChessPiece.classes[self.board[row][col]][0]
            if empty_cell_counter != 0:
                fen_not += str(empty_cell_counter)
            fen_not += '/'
        castling: str = self.w_castling + self.b_castling
        if self.b_castling == '-':
            castling = self.w_castling + ' -'
        elif self.w_castling == '-':
            castling = self.b_castling + ' -'
        fen_not = fen_not[:-1] + ' ' + self.cur_player + ' ' + castling + ' ' + self.en_passant
        fen_not += ' ' + str(self.half_move) + ' ' + str(self.move_counter)
        return fen_not

    def _check_for_movement(self, board: list[list[int]]) -> str:
        cur_move = ''
        start_pos, end_pos = (-1, -1), (-1, -1)
        pseudo_start = ((-1, -1), '')  # for ep
        piece, act = '', ''
        number_of_changes: int = 0
        for row in range(8):
            for col in range(8):
                prev_cell = self.board[row][col]
                cur_cell = board[row][col]
                number_of_changes += 1
                if prev_cell != -1 and cur_cell == -1:
                    if start_pos[0] != -1:
                        pseudo_start = ((row, col), ChessPiece.classes[prev_cell][0].upper())
                        act = 'x'
                    else:
                        start_pos = (row, col)
                elif prev_cell == -1 and cur_cell != -1:
                    end_pos = (row, col)
                    piece = ChessPiece.classes[cur_cell][0].upper()
                elif (0 <= cur_cell < 6 <= prev_cell < 12) or (0 <= prev_cell < 6 <= cur_cell < 12):
                    end_pos = (row, col)
                    piece = ChessPiece.classes[cur_cell][0].upper()
                    act = 'x'
                else:
                    number_of_changes -= 1
        # print('NM', number_of_changes)
        if number_of_changes > 5:
            return 'err'
        if start_pos[0] != -1 and end_pos[0] != -1:
            if piece == pseudo_start[1]:
                start_pos = pseudo_start[0]
            if piece == 'p' and start_pos[0] == 1 and end_pos[0] == 3:
                self.en_passant = self.col_names[end_pos[1]] + str(5)
            elif piece == 'P' and start_pos[0] == 6 and end_pos[0] == 4:
                self.en_passant = self.col_names[end_pos[1]] + str(3)
            else:
                self.en_passant = '-'
            if piece == 'p' or piece == 'P' or act == 'x':
                self.half_move = 0
            else:
                self.half_move += 1
            cur_move = piece + self.col_names[start_pos[1]] + str(8 - start_pos[0])
            cur_move += act + self.col_names[end_pos[1]] + str(8 - end_pos[0])
        return cur_move

    def make_move(self, move: str):
        if self.cur_player == 'w':
            self.alg_not += str(self.move_counter) + '. ' + move + ' '
            self.alg_not_list.append([move])
            self.move_describe = 0
        else:
            self.alg_not += move + ' '
            self.alg_not_list[-1].append(move)
            self.move_describe = 1
            print(f'\nAlg not: {self.alg_not}')
        print(f'Move: {move}')
        print(self)
        self.cur_player = 'w' if self.cur_player == 'b' else 'b'
        self.fen_not = self.get_fen_notation()
        print(self.fen_not)
        if self.cur_player == 'b':
            self.move_counter += 1
        self._get_2d_chess()


    def _check_for_castling(self, new_board: list[list[int]]) -> str:
        row_num, rook_ind, king_ind = 0, 5, 1
        castling_availability = self.b_castling.upper()
        if self.cur_player == 'w':
            castling_availability = self.w_castling
            row_num, rook_ind, king_ind = 7, 11, 7
        if castling_availability == '-':
            return ''
        castling: str = ''
        row_num: int = 7 if self.cur_player == 'w' else 0
        # print(self.board[row_num][7], self.board[row_num][4], new_board[row_num][5], new_board[row_num][6])
        if (self.board[row_num][0] == rook_ind and self.board[row_num][4] == king_ind) and \
                (new_board[row_num][3] == rook_ind and new_board[row_num][2] == king_ind):
            castling = 'O-O-O'
        elif (self.board[row_num][7] == rook_ind and self.board[row_num][4] == king_ind) and \
                (new_board[row_num][5] == rook_ind and new_board[row_num][6] == king_ind):
            castling = 'O-O'

        if castling != '':
            castling_availability = '-'
        elif new_board[row_num][4] != king_ind:
            castling_availability = '-'
        elif castling_availability[-1] == 'Q' and new_board[row_num][0] != rook_ind:
            castling_availability = castling_availability[:-1]
        elif castling_availability[0] == 'K' and new_board[row_num][7] != rook_ind:
            castling_availability = castling_availability[1:]
        if castling_availability == '':
            castling_availability = '-'
        if self.cur_player == 'w':
            self.w_castling = castling_availability
        else:
            self.b_castling = castling_availability.lower()
        return castling

    def _get_2d_chess(self):
        self.chess_2d_img: np.ndarray = cv2.imread(self.path_2d_pieces + 'board.png', -1)
        start_pos, offset = (53, 53), 132
        for row in range(8):
            y = start_pos[1]+row*offset
            for col in range(8):
                piece_ind: int = self.board[row][col]
                if piece_ind != -1:
                    piece_symb:str = ChessPiece.classes[piece_ind][0]
                    piece_symb = 'w'+piece_symb.lower() if piece_symb.isupper() else 'b'+piece_symb
                    piece_img = cv2.imread(self.path_2d_pieces + piece_symb + '.png', -1)
                    x = start_pos[0]+col*offset
                    self.chess_2d_img = self.put_piece_img_on_board(self.chess_2d_img, piece_img, (x, y))

        self.chess_2d_img = cv2.cvtColor(self.chess_2d_img, cv2.COLOR_BGRA2BGR)

    def put_piece_img_on_board(self, board_img: np.ndarray, piece_img: np.ndarray, pos: tuple[int, int]):
        y1, y2 = pos[1], pos[1] + piece_img.shape[0]
        x1, x2 = pos[0], pos[0] + piece_img.shape[1]

        alpha_s = piece_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            board_img[y1:y2, x1:x2, c] = (alpha_s * piece_img[:, :, c] + alpha_l * board_img[y1:y2, x1:x2, c])
        return board_img

    def save_alg_not(self, path: str):
        path += '\\'+str(self.filename_ind_alg)+'.txt'
        self.filename_ind_alg += 1
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.alg_not)


    def __str__(self):
        str_board: str = ''
        for y in range(8):
            even = int(y % 2 == 0)
            for x in range(8):
                if self.board[y][x] == -1:
                    str_board += '⛀' if x % 2 == even else '⛂'
                else:
                    str_board += ChessPiece.classes[int(self.board[y][x])][2]
            str_board += '\n'
        return str_board
