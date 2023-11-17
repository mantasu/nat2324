from typing import Callable, Collection

import numpy as np

from ..utils import Loss


class Sequence:
    """Sequence problem.

    This problem is to find the next element of a sequence. The sequence
    can be one of the following:

        * Fibonacci
        * Tribonacci
        * Pell
        * Arithmetic-geometric

    The sequence is given as a list of integers. The fitness is computed
    based on the following:

        * Length of the sequence
        * Number of unique elements in the sequence
        * Correctness of the sequence (value matching)

    The fitness is computed as follows:

    .. math::

        \\text{fitness} = \\frac{1}{4} \\text{length} + \\frac{1}{4}
        \\text{unique} + \\frac{1}{4} \\text{correctness}

    The fitness can be computed either for training or testing data.

    Args:
        sequence_type (str, optional): The type of sequence to use.
            Defaults to "fibonacci".
        loss_type (str, optional): The loss function to use. Defaults
            to "mae".
        num_samples (int, optional): The number of samples to generate.
            Defaults to 100.
        sample_range (tuple[int, int], optional): The range of samples
            to generate. Defaults to (0, 100).
        train_frac (float, optional): The fraction of samples to use for
            training. Defaults to 0.8.
        seed (int | None, optional): The seed to use for the random
            number generator. Defaults to ``None``.
    """

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
        # Init attributes based on given params
        self.loss_fn = Loss(loss_type, is_inverse=True)
        self.rng = np.random.default_rng(seed=seed)

        # Get the sequence and starting sequence
        self.sequence = getattr(self, sequence_type)
        self.start = self.START[sequence_type].copy()

        # Generate the name of the sequence problem (this class name)
        self.name = f"Sequence({sequence_type}, {loss_type})"

        # Generate samples: inputs and ground-truth outputs
        xs = self.rng.choice(range(*sample_range), num_samples, replace=False)
        ys = [self.sequence(x) for x in xs]

        # Split the data to training and testing
        num_train = int(train_frac * num_samples)
        self.x_train, self.y_train = xs[:num_train], ys[:num_train]
        self.x_test, self.y_test = xs[num_train:], ys[num_train:]

    @staticmethod
    def fibonacci(n: int) -> list[int]:
        """Generate the fibonacci sequence.

        This function generates the fibonacci sequence up to the given
        number of elements. Each number is the sum of the previous two
        numbers.

        Args:
            n (int): The number of elements to generate.

        Returns:
            list[int]: The generated fibonacci sequence.
        """
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
        """Generate the tribonacci sequence.

        This function generates the tribonacci sequence up to the given
        number of elements. Each number is the sum of the previous three
        numbers.

        Args:
            n (int): The number of elements to generate.

        Returns:
            list[int]: The generated tribonacci sequence.
        """
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
        """Generate the pell sequence.

        This function generates the pell sequence up to the given number
        of elements. Each number is the sum of the previous two numbers
        with the last one being multiplied by two.

        Args:
            n (int): The number of elements to generate.

        Returns:
            list[int]: The generated pell sequence.
        """
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
        """Generate the arithmetic-geometric sequence.

        This function generates the arithmetic-geometric sequence up to
        the given number of elements. Each number is the sum of the last
        two numbers: one is multiplied by two and the the other is
        multiplied by a half.

        Args:
            n (int): The number of elements to generate.

        Returns:
            list[float]: The generated arithmetic-geometric sequence.
        """
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
        """Evaluate the given tree.

        This function evaluates the given tree based on the training or
        testing data. The fitness is computed based on the following:

            * Length of the sequence
            * Number of unique elements in the sequence
            * Correctness of the sequence (value matching)

        The fitness is computed as follows:

        .. math::

            \\text{fitness} = \\frac{1}{4} \\text{length} + \\frac{1}{4}
            \\text{unique} + \\frac{1}{4} \\text{correctness}

        The fitness can be computed either for training or testing data.

        Args:
            tree (typing.Callable[..., typing.Collection[float]]): The
                tree to evaluate.
            is_test (bool, optional): Whether to evaluate on testing
                data. Defaults to ``False``.

        Returns:
            float: The fitness of the given tree.
        """
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
