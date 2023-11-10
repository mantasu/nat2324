from typing import Callable, Collection

import numpy as np

from ..utils import Loss


class Sequence:
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
        if n == 1:
            return [0]
        elif n == 2:
            return [0, 1]

        series = [0, 1]

        for _ in range(2, n):
            series.append(series[-1] + series[-2])

        return series

    def evaluate(
        self,
        tree: Callable[[np.ndarray], Collection[float]],
        is_test: bool = False,
    ) -> float:
        xs = self.x_test if is_test else self.x_train
        ys = self.y_test if is_test else self.y_train
        fitness = 0

        for x, y in zip(xs, ys):
            # Predict sequence TODO: fix this (s should be automatic)
            y_pred = tree(x=x, s=[])

            if not isinstance(y_pred, (list, tuple, np.ndarray)):
                return 0

            # Truncate to the length of the shortest list
            min_len = min(len(y_pred), len(ys))
            y_pred, y_pred_remain = y_pred[:min_len], y_pred[min_len:]
            y, y_remain = y[:min_len], y_pred[min_len:]
            remain = y_pred_remain + y_remain

            # Calculate fitness
            fitness += self.loss_fn(np.array(y), np.array(y_pred))
            fitness += self.loss_fn(np.array(remain), np.zeros_like(remain))

        return fitness
