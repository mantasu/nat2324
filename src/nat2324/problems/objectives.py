import numpy as np


class Objectives:
    def __init__(self, function_type: str = "rastrigin"):
        self.function = getattr(self, function_type)
    
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
        return -self.function(solution) # Convert to maximization problem