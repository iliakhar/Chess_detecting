from ChessNotation.ChessPiecesDetecting.ChessPiece import ChessPiece


class ChessNotation:
    def __init__(self):
        self.col_names: list[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.alg_not: str = ''
        self.en_passant: str = ''
        self.move_counter: int = 1
        self.half_move: int = 0
        self.cur_player: str = 'w'
        self.board: list[list[int]] = []
        self.b_castling = 'kq'
        self.w_castling = 'KQ'

    def set_board(self, board: list[list[int]]):
        if len(self.board) == 0:
            self.board = board.copy()
            print(self)
            return
        cur_move: str = ''
        if self.w_castling != '-' or self.b_castling != '-':
            cur_move = self._check_for_castling(board)
            if cur_move != '':
                self.board = board.copy()
                self.make_move(cur_move)
                return

        cur_move += self._check_for_movement(board)
        if cur_move != '':
            if cur_move[0] == 'P':
                cur_move = cur_move[1:]
            self.board = board.copy()
            self.make_move(cur_move)

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
        for row in range(8):
            for col in range(8):
                prev_cell = self.board[row][col]
                cur_cell = board[row][col]
                if prev_cell != -1 and cur_cell == -1:
                    if start_pos[0] != -1:
                        pseudo_start = ((row, col), ChessPiece.classes[prev_cell][0].upper())
                        act = 'x'
                    else:
                        start_pos = (row, col)
                elif prev_cell == -1 and cur_cell != -1:
                    end_pos = (row, col)
                    piece = ChessPiece.classes[cur_cell][0].upper()
                elif (1 <= cur_cell < 6 and 6 <= prev_cell < 12) or (1 <= prev_cell < 6 and 6 <= cur_cell < 12):
                    end_pos = (row, col)
                    piece = ChessPiece.classes[cur_cell][0].upper()
                    act = 'x'
        if start_pos[0] != -1 and end_pos[0] != -1:
            if piece == pseudo_start[1]:
                start_pos = pseudo_start[0]
            if piece == 'p' and start_pos[0] == 1 and end_pos[0] == 3:
                self.en_passant = self.col_names[end_pos[1]] + str(5)
            elif piece == 'P' and start_pos[0] == 6 and end_pos[0] == 4:
                self.en_passant = self.col_names[end_pos[1]] + str(3)
            else:
                self.en_passant = ''
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
        else:
            self.alg_not += move + ' '
            print(f'\nAlg not: {self.alg_not}')
        print(f'Move: {move}')
        print(self)
        self.cur_player = 'w' if self.cur_player == 'b' else 'b'
        print(self.get_fen_notation())
        if self.cur_player == 'b':
            self.move_counter += 1

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
