import numpy as np


class Sumplete:
    def __init__(
        self,
        K: int = (k := 3),
        low: int = 1,
        high: int = round(4 * np.log(k * k)),
        deletion_rate: float = 1/3,
        evaluation_type: str = "absolute",
        seed: int = None,
    ):
        # Initialize parameters
        self.K = K
        self.low = low
        self.high = high
        self.deletion_rate = deletion_rate
        self.evaluation_type = evaluation_type

        # Generate the game: new board, column sums, and row sums
        self.board, self.col_sums, self.row_sums = self.new(seed=seed)

    def new(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Apply seed
        np.random.seed(seed)

        # Generate new board
        board = np.random.randint(
            self.low,
            self.high,
            size=(self.K, self.K),
        )

        # Apply deletion rate
        mask = np.random.choice(
            [0, 1],
            size=board.shape,
            p=[self.deletion_rate, 1-self.deletion_rate]
        )
        
        # Calculate column and row sums
        col_sums = np.sum(board * mask, axis=0)
        row_sums = np.sum(board * mask, axis=1)

        return board, col_sums, row_sums

    def evaluate(self, cell_mask: np.ndarray):
        # Compute the new masked board
        board = self.board * cell_mask.reshape(self.K, self.K)

        # Calculate column and row sums
        col_sums = np.sum(board, axis=0)
        row_sums = np.sum(board, axis=1)

        if self.evaluation_type == "absolute":
            # Check if all row and all col sums are equal
            is_col_eq = np.all(col_sums == self.col_sums)
            is_row_eq = np.all(row_sums == self.row_sums)
            return int(is_col_eq and is_row_eq)
        elif self.evaluation_type == "sum":
            # Compute the number of correct col and row sums
            col_correct = (col_sums == self.col_sums).sum()
            row_correct = (row_sums == self.row_sums).sum()
            return col_correct + row_correct
        elif self.evaluation_type == "distance":
            # Compute the absolute difference between the sums
            col_sums = np.abs(col_sums - self.col_sums)
            row_sums = np.abs(row_sums - self.row_sums)
            return -col_sums.sum() - row_sums.sum()

    def show(self, mask: np.ndarray = None, num_digits: int = 5):
        # Initialize variables
        k, n = self.K, num_digits

        # Generate a mask to hide certain numbers in board
        mask = np.ones(self.board.shape) if mask is None else mask
        mask = mask.reshape(self.K, self.K) if mask.ndim == 1 else mask
        
        # Show the top border
        s = "+" + "-" * (n * k + k + 1) + "+\n"

        for i in range(k):
            # Generate printable values (mask if needed)
            values = (f"{self.board[i][j]:{n}}"
                      if mask[i][j] else n * " "
                      for j in range(k))
            s += "| " + ' '.join(values) + " | " + f"{self.row_sums[i]:n}\n"
        
        # Show the bottom border and column sums
        s += "+" + "-" * (n * k + k + 1) + "+\n"
        s += "  " + " ".join(f"{self.col_sums[i]:{n}}" for i in range(k))

        # Print
        print(s)
