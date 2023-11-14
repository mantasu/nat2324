from typing import Callable, Collection

import numpy as np

from ..utils import Loss


class Sequence:
    START = {
        "fibonacci": [0, 1],
        "tribonacci": [0, 1, 1],
        "pell": [0, 1],
        "arithmetic_geometric": [0, 1],
    }

    def __init__(
        self,
        sequence_type: str = "fibonacci",
        loss_type: str = "mae",
        num_samples: int = 100,
        sample_range: tuple[int, int] = (0, 100),
        train_frac: float = 0.8,
        seed: int | None = None,
    ):
        self.sequence = getattr(self, sequence_type)
        self.start = self.START[sequence_type].copy()

        self.loss_fn = Loss(loss_type, is_inverse=True)
        self.rng = np.random.default_rng(seed=seed)

        self.name = f"Sequence({sequence_type}, {loss_type})"

        xs = self.rng.choice(range(*sample_range), num_samples, replace=False)
        ys = [self.sequence(x) for x in xs]

        num_train = int(train_frac * num_samples)
        self.x_train, self.y_train = xs[:num_train], ys[:num_train]
        self.x_test, self.y_test = xs[num_train:], ys[num_train:]

    @staticmethod
    def fibonacci(n: int) -> list[int]:
        # Initialize starting sequence of fibonacci
        series = Sequence.START["fibonacci"].copy()

        if n <= len(series):
            # Series is short
            return series[:n]

        for _ in range(len(series), n):
            # Append the sum of the last two numbers
            series.append(series[-1] + series[-2])

        return series

    @staticmethod
    def tribonacci(n: int) -> list[int]:
        # Initialize starting sequence of tribonacci
        series = Sequence.START["tribonacci"].copy()

        if n <= len(series):
            # Series is short
            return series[:n]

        for _ in range(len(series), n):
            # Append the sum of the last three numbers
            series.append(series[-1] + series[-2] + series[-3])

        return series

    @staticmethod
    def pell(n: int) -> list[int]:
        # Initialize starting sequence of pell
        series = Sequence.START["pell"].copy()

        if n <= len(series):
            # Series is short
            return series[:n]

        for _ in range(len(series), n):
            # Sum of last two numbers (multiply first)
            series.append(2 * series[-1] + series[-2])

        return series

    @staticmethod
    def arithmetic_geometric(n: int) -> list[float]:
        # Initialize starting sequence of arithmetic_geometric
        series = Sequence.START["arithmetic_geometric"].copy()

        if n <= len(series):
            # Series is short
            return series[:n]

        for _ in range(len(series), n):
            # Append the sum of the last two multiplied numbers
            series.append(2 * series[-1] + 0.5 * series[-2])

        return series

    def evaluate(
        self,
        tree: Callable[..., Collection[float]],
        is_test: bool = False,
    ) -> float:
        # Get the correct data based on the is_test flag
        xs = self.x_test if is_test else self.x_train
        ys = self.y_test if is_test else self.y_train
        fitnesses = []

        for x, y in zip(xs, ys):
            # Evaluate based on start sequence and remaining elements
            y_pred = tree(x=x - len(self.start), s=self.start.copy())
            fitness = 0

            if not isinstance(y_pred, Collection) or len(y_pred) == 0:
                # y_pred is not a list or is empty
                fitnesses.append(0)
                continue

            # Compute the format fitness (length and uniqueness)
            len_fitness = self.loss_fn([len(y)], [len(y_pred)])
            unq_fitness = self.loss_fn([len(set(y))], [len(set(y_pred))])
            fitness += 0.25 * len_fitness + 0.25 * unq_fitness

            # Compute the correctness fitness (sequence value matching)
            min_len = min(len(y_pred), len(y))
            lft_fitness = self.loss_fn(y[:min_len], y_pred[:min_len])
            rht_fitness = self.loss_fn(y[-min_len:], y_pred[-min_len:])
            fitness += 0.25 * lft_fitness + 0.25 * rht_fitness

            # Append the fitness to the list
            fitnesses.append(fitness)

        return np.mean(fitnesses)

    def __call__(
        self,
        tree: Callable[..., Collection[float]],
        is_test: bool = False,
    ) -> float:
        return self.evaluate(tree=tree, is_test=is_test)

    def __repr__(self) -> str:
        return self.name
