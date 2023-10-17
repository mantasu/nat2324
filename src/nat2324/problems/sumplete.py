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
        
        self.decay_factor = 0.95
        self.local_bests = np.zeros((1, self.K ** 2))
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
            raw_fitness = (col_correct + row_correct) / (self.K * 2)
        elif self.evaluation_type == "antisum":
            # Compute the number of incorrect col and row sums
            col_incorrect = (col_sums != self.col_sums).sum()
            row_incorrect = (row_sums != self.row_sums).sum()
            return -(col_incorrect + row_incorrect)
        elif self.evaluation_type == "distance":
            # Compute the absolute difference between the sums
            col_sums = np.abs(col_sums - self.col_sums)
            row_sums = np.abs(row_sums - self.row_sums)
            raw_fitness = -col_sums.sum() - row_sums.sum()
        elif self.evaluation_type == "even":
            # Compute the number of correct col and row sums
            col_correct = (col_sums == self.col_sums).sum()
            row_correct = (row_sums == self.row_sums).sum()

            if col_correct == row_correct:
                result = col_correct + row_correct
            elif col_correct - row_correct != 0:
                result = min(col_correct, row_correct)
            else:
                result = 0

            return result
        elif self.evaluation_type == "included":
            # Compute the number of correct col and row sums
            is_col_correct = (col_sums == self.col_sums)
            is_row_correct = (row_sums == self.row_sums)

            raw_fitness1 = (is_row_correct & is_col_correct[:, None]).sum() / (self.K ** 2)
            raw_fitness2 = (is_row_correct.sum() + is_row_correct.sum()) / (self.K * 2)
            raw_fitness = (raw_fitness1 + raw_fitness2) / 2

        # If this is the best solution and its fitness is less than 1
        # if (cell_mask == self.local_bests).all(axis=1).any() and (raw_fitness < 1):
        #     # Find the index of the matching local best
        #     # index = np.where((cell_mask == self.local_bests).all(axis=1))[0][0]
        #     index = np.argmax((cell_mask == self.local_bests).all(axis=1))
            
        #     # Increment the corresponding stagnation counter
        #     self.stagnation_counters[index] += 1
            
        #     # Apply decay factor
        #     fitness = np.round(raw_fitness * (self.decay_factor ** self.stagnation_counters[index]), 6)
        #     # print(self.stagnation_counters[index], raw_fitness * (self.K ** 2), fitness)
        # else:
        #     # Add the new solution to local_bests
        #     # self.local_bests = np.append(self.local_bests, [cell_mask], axis=0)
        #     # Add a new stagnation counter for this solution
        #     fitness = raw_fitness
        # print(fitness, self.local_best_fitness)
        # Update best solution and fitness
        # if fitness > self.local_best_fitness:
        #     self.local_bests = np.append(self.local_bests, [cell_mask], axis=0)
        #     self.stagnation_counters = np.append(self.stagnation_counters, [0], axis=0)
        #     self.local_best_fitness = fitness
        
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
            values = (f"{self.board[i][j]:{n}}"
                      if mask[i][j] else n * " "
                      for j in range(k))
            s += "| " + ' '.join(values) + " | " + f"{self.row_sums[i]:n}\n"
        
        # Show the bottom border and column sums
        s += "+" + "-" * (n * k + k + 1) + "+\n"
        s += "  " + " ".join(f"{self.col_sums[i]:{n}}" for i in range(k))

        # Print
        print(s)
