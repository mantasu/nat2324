import numpy as np


class Objective:
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
    def sphere(x):
        return np.sum(x**2)

    @staticmethod
    def rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        return 10 * len(x) + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum()

    @staticmethod
    def ackley(x):
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) + 20 \
               - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + np.exp(1)

    @staticmethod
    def griewank(x):
        return 1 + np.sum(x ** 2) / 4000 \
               - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    @staticmethod
    def schwefel(x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def evaluate(self, solution: np.ndarray) -> float:
        if self.is_maximization:
            return -self.function(solution)
        else:
            return self.function(solution)
    
    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)
    
    def __repr__(self) -> str:
        return f"{self.function_type}-{'min' if self.is_maximization else 'max'}"
