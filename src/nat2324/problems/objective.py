import numpy as np


class Objective:
    """Objective function.

    This class represents an objective function. It supports the
    following types:

            * Sphere
            * Rosenbrock
            * Rastrigin
            * Ackley
            * Griewank
            * Schwefel

    Args:
        function_type (str, optional): The type of the objective
            function. Defaults to ``"rastrigin"``.
        is_maximization (bool, optional): Whether the objective function
            is a maximization problem. Defaults to ``True``.
    """

    AVAILABLE_FUNCTIONS = [
        "sphere",
        "rosenbrock",
        "rastrigin",
        "ackley",
        "griewank",
        "schwefel",
    ]

    def __init__(
        self,
        function_type: str = "rastrigin",
        is_maximization: bool = True,
    ):
        # Check which function is chosen
        self.function_type = function_type
        self.is_maximization = is_maximization
        self.function = getattr(self, self.function_type)

    @property
    def bounds(self) -> tuple[float, float]:
        match self.function_type:
            case "sphere":
                return (-5.12, 5.12)
            case "rosenbrock":
                return (-2.048, 2.048)
            case "rastrigin":
                return (-5.12, 5.12)
            case "ackley":
                return (-32.768, 32.768)
            case "griewank":
                return (-600, 600)
            case "schwefel":
                return (-500, 500)

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function.

        The Sphere function is defined as:

        ..math::

            f(x) = \sum_{i=1}^D x_i^2

        Args:
            x (numpy.ndarray): The input vector.

        Returns:
            float: The calculated value.
        """
        return np.sum(x**2)

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function.

        The Rosenbrock function is defined as:

        ..math::

            f(x) = \sum_{i=1}^D [100(x_i + 1 - x_i^2)^2 + (1 - x_i)^2]

        Args:
            x (numpy.ndarray): The input vector.

        Returns:
            float: The calculated value.
        """
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function.

        The Rastrigin function is defined as:

        ..math::

            f(x) = 10n + \sum_{i=1}^D [x_i^2 - 10 \cos(2 \pi x_i)]


        Args:
            x (numpy.ndarray): The input vector.

        Returns:
            float: The calculated value.
        """
        return 10 * len(x) + (x**2 - 10 * np.cos(2 * np.pi * x)).sum()

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        return (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x)))
            + 20
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x))
            + np.exp(1)
        )

    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function.

        The Griewank function is defined as:

        ..math::

            f(x) = 1 + \frac{1}{4000} \sum_{i=1}^D x_i^2
            - \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}})

        Args:
            x (numpy.ndarray): The input vector.

        Returns:
            float: The calculated value.
        """
        return (
            1
            + np.sum(x**2) / 4000
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        )

    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Schwefel function.

        The Schwefel function is defined as:

        ..math::

            f(x) = 418.9829D - \sum_{i=1}^D x_i \sin(\sqrt{|x_i|})

        Args:
            x (numpy.ndarray): The input vector.

        Returns:
            float: The calculated value.
        """
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate the objective function.

        This method evaluates the objective function on the given
        solution. If the objective function is a maximization problem,
        it negates the value.

        Args:
            solution (numpy.ndarray): The solution vector to evaluate.
                It should be a 1D numpy array.

        Returns:
            float: The calculated objective value.
        """
        if self.is_maximization:
            # Negate if maximization problem
            return -self.function(solution)
        else:
            # Return the value as is
            return self.function(solution)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def __repr__(self) -> str:
        return f"{self.function_type}-{'max' if self.is_maximization else 'min'}"
