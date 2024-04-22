class Point:
    def __init__(self, cord: tuple[int, int], line_ind_h: int = -1, line_ind_v: int = -1):
        self.x: int = cord[0]
        self.y: int = cord[1]
        self.line_ind_h: int = line_ind_h
        self.line_ind_v: int = line_ind_v

    def __str__(self):
        return f'({self.x}, {self.y}, horiz = {self.line_ind_h}, vert = {self.line_ind_v})'
