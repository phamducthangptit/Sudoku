class Sudoku:
    def __init__(self, board):
        self.board = board
    def solveSudoku(self):
        n = 9
        def isValid(row, col, ch):
            row, col = int(row), int(col)

            for i in range(9):

                if self.board[i][col] == ch:
                    return False
                if self.board[row][i] == ch:
                    return False

                if self.board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == ch:
                    return False

            return True
        for i in range(0,9):
            for j in range(0,9):
                if self.board[i][j] == 0:
                    continue
                tmp = self.board[i][j]
                self.board[i][j] = 0
                if isValid(i,j,tmp):
                    self.board[i][j] = tmp
                else:
                    self.board[i][j] = tmp
                    return False


        def solve(row, col):
            if row == n:
                return True
            if col == n:
                return solve(row + 1, 0)

            if self.board[row][col] == 0:
                for i in range(1, 10):
                    if isValid(row, col, i):
                        self.board[row][col] = i

                        if solve(row, col + 1):
                            return True
                        else:
                            self.board[row][col] = 0
                return False
            else:
                return solve(row, col + 1)
        solve(0, 0)
        return True
    def printSudoku(self):
        for y in self.board:
            for x in y:
                print(x, end = "")
            print()

if __name__ == "__main__":
    board1  = [[2, 0, 0, 0, 0, 1, 4, 0, 0,],
                [7, 0, 0, 0, 9, 0, 0, 0, 0],
                 [0, 3, 0, 0, 5, 6, 0, 0, 2],
                 [0, 7, 0, 2, 1, 8 ,5 ,0, 6],
                 [1, 0, 2, 0, 0, 5,9 ,3, 0],
                 [0, 6, 0, 0, 0, 9,0 ,0, 0],
                 [6, 0, 0, 0, 8, 0 ,0, 0, 0],
                 [9, 0, 0, 5, 0, 3, 8 ,0 ,0],
                 [4 ,1, 8, 0, 2, 0 ,0 ,6 ,5]]
    sudoku= Sudoku(board1)
    sudoku.solveSudoku()
    sudoku.printSudoku()