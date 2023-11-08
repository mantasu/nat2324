import numpy as np


class Sumplete:
    def __init__(
        self,
        K: int = (k := 3),
        low: int = 1,
        high: int = round(4 * np.log(k * k)),
        deletion_rate: float = 1 / 3,
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

        self.decay_factor = 0.95
        self.local_bests = np.zeros((1, self.K**2))
        self.stagnation_counters = np.zeros(1)
        self.local_best_fitness = 0

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
            [0, 1], size=board.shape, p=[self.deletion_rate, 1 - self.deletion_rate]
        )

        # Calculate column and row sums
        col_sums = np.sum(board * mask, axis=0)
        row_sums = np.sum(board * mask, axis=1)

        return board, col_sums, row_sums

    def evaluate(self, cell_mask: np.ndarray):
        if self.evaluation_type == "mixed":
            fitnesses = []

            for eval_type in ["distance", "row_col", "cellular"]:
                self.evaluation_type = eval_type
                fitnesses.append(self.evaluate(cell_mask))

            self.evaluation_type = "mixed"

            return np.mean(fitnesses)

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
        elif self.evaluation_type == "distance":
            # Compute the absolute difference between the sums
            col_diffs = np.abs(col_sums - self.col_sums) / np.maximum(1, self.col_sums)
            row_diffs = np.abs(row_sums - self.row_sums) / np.maximum(1, self.row_sums)
            distance = (col_diffs.sum() + row_diffs.sum()) / 2
            raw_fitness = 1 / (1 + distance)
        elif self.evaluation_type == "row_col":
            # Compute the number of correct col and row sums
            col_correct = (col_sums == self.col_sums).sum()
            row_correct = (row_sums == self.row_sums).sum()
            raw_fitness = (col_correct + row_correct) / (self.K * 2)
        elif self.evaluation_type == "cellular":
            # Compute the number of correct col and row sums
            is_col_correct = col_sums == self.col_sums
            is_row_correct = row_sums == self.row_sums

            raw_fitness = (is_row_correct & is_col_correct[:, None]).sum() / (
                self.K**2
            )
            # raw_fitness2 = (is_row_correct.sum() + is_row_correct.sum()) / (self.K * 2)
            # raw_fitness = (raw_fitness1 + raw_fitness2) / 2

        return raw_fitness

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
            values = (
                f"{self.board[i][j]:{n}}" if mask[i][j] else n * " " for j in range(k)
            )
            s += "| " + " ".join(values) + " | " + f"{self.row_sums[i]:n}\n"

        # Show the bottom border and column sums
        s += "+" + "-" * (n * k + k + 1) + "+\n"
        s += "  " + " ".join(f"{self.col_sums[i]:{n}}" for i in range(k))

        # Print
        print(s)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def __repr__(self) -> str:
        return f"{self.evaluation_type}"
