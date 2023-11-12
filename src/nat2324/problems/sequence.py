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

        xs = self.rng.choice(range(*sample_range), num_samples, replace=False)
        ys = [self.sequence(x) for x in xs]

        num_train = int(train_frac * num_samples)
        self.x_train, self.y_train = xs[:num_train], ys[:num_train]
        self.x_test, self.y_test = xs[num_train:], ys[num_train:]

    @staticmethod
    def recaman(n):
        seq = [0] * n

        for i in range(1, n):
            curr = seq[i - 1] - i
            if curr > 0 and curr not in seq[:i]:
                seq[i] = curr
            else:
                seq[i] = seq[i - 1] + i
        return seq

    @staticmethod
    def catalan(n):
        if n == 0 or n == 1:
            return 1

        catalan = [0 for _ in range(n + 1)]
        catalan[0] = 1
        catalan[1] = 1

        for i in range(2, n + 1):
            catalan[i] = 0
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - j - 1]

        return catalan[n]

    @staticmethod
    def primes(n):
        sieve = [True] * n

        for x in range(2, int(n**0.5) + 1):
            for y in range(x * 2, n, x):
                sieve[y] = False

        return [p for p in range(2, n) if sieve[p]]

    @staticmethod
    def mersenne_primes(n):
        def is_prime(num):
            if num < 2 or num % 2 == 0:
                return False

            sqr = int(num**0.5) + 1

            for divisor in range(3, sqr, 2):
                if num % divisor == 0:
                    return False
            return True

        primes = []
        candidate = 2

        while len(primes) < n:
            mersenne_candidate = 2**candidate - 1
            if is_prime(candidate) and is_prime(mersenne_candidate):
                primes.append(mersenne_candidate)
            candidate += 1

        return primes

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
        tree: Callable[[np.ndarray], Collection[float]],
        is_test: bool = False,
    ) -> float:
        xs = self.x_test if is_test else self.x_train
        ys = self.y_test if is_test else self.y_train
        fitnesses = []
        # tree.show()

        # print("Ultra wait")

        for x, y in zip(xs, ys):
            fitness = 0
            # Predict sequence TODO: fix this (s should be automatic)

            # print("Waiting...")
            y_pred = tree(x=x - len(self.start), s=self.start.copy())
            # print(y, y_pred)
            # print("Wit end")

            if not isinstance(y_pred, Collection):
                # print("Ultra wait end")
                return 0

            if len(y_pred) == 0:
                fitnesses.append(fitness)
                continue

            # if len(y_pred) <= len(y):

            len_fitness = self.loss_fn(np.array([len(y)]), np.array([len(y_pred)]))

            fitness += 0.25 * len_fitness
            # print("len", len_fitness)

            # std_fitness = self.loss_fn(
            #     np.std(y, keepdims=True), np.std(y_pred, keepdims=True)
            # )
            std_fitness = self.loss_fn(
                np.array([len(set(y))]), np.array([len(set(y_pred))])
            )
            fitness += 0.25 * std_fitness

            # print("std", std_fitness)

            # if std_fitness < 0.5 or len_fitness < 0.5:
            #     fitnesses.append(fitness)
            #     continue

            # if std_fitness < 0.2:
            #     fitnesses.append(fitness)
            #     continue

            # Truncate to the length of the shortest list
            min_len = min(len(y_pred), len(y))

            # y_pred_left, y_left = y_pred[:min_len], y[:min_len]
            # y_pred_right, y_right = y_pred[-min_len:], y[-min_len:]

            fitness += 0.5 * self.loss_fn(
                np.array(y[:min_len] + y[-min_len:]),
                np.array(y_pred[:min_len] + y_pred[-min_len:]),
            )

            # y_pred, y_pred_remain = y_pred[:min_len], y_pred[min_len:]
            # y, y_remain = y[:min_len], y[min_len:]
            # remain = y_pred_remain + y_remain

            # fitness += 0.25 * self.loss_fn(np.array(y), np.array(y_pred))

            # if len(remain) != 0:
            #     fitness += 0.25 * self.loss_fn(np.array(remain), np.zeros_like(remain))
            # else:
            #     fitness += 0.25 * 1

            # Encourages to find correct output structure but also if some components are good, they may also earn some fitness

            fitnesses.append(fitness)

        return np.mean(fitnesses)
