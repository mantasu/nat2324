import numpy as np


class Objectives:
    def __init__(self, function_type: str = "rastrigin"):
        pass
    
    @staticmethod
    def rastrigin(x: np.ndarray):
        return 10 * len(x) + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum()

    def evaluate(self, solution):
        pass