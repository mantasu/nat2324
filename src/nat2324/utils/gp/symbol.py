import math
import operator
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Union

import numpy as np
from anytree import Node


class Symbol(ABC):
    """Abstract base class for symbols.

    A symbol is either :class:`.Terminal` or :class:`.NonTerminal`.
    Terminals are variables (x, y, z, ...) and constants
    (0, 0.123, -5, ...), and non-terminals are functions
    (+, -, *, /, ...).

    Args:
        p (float, optional): The probability of the symbol. Defaults to
            ``None``.
    """

    def __init__(self, p: float | None = None):
        # Assign p
        self.p = p

    # @classmethod
    # @abstractmethod
    # def get_default(cls) -> set["Symbol"]:
    #     """Abstract method to get the default set of symbols.

    #     This should return the basic (common) set of symbols: terminals
    #     or non-terminals, depending on which class implements this.

    #     Returns:
    #         set[Symbol]: The default set of symbols.
    #     """
    #     ...

    @staticmethod
    def validate_ps(symbols: set["Symbol"]) -> set["Symbol"]:
        """Validates and adjusts the probabilities of a set of symbols.

        This method validates and adjust probabilities in 3 cases:

            1. If the total probability sums up to 1 and there are
               symbols with ``None`` probability, it makes the total
               probability half and distributes the other half equally
               among symbols with ``None`` probability.
            2. If the total probability does not sum up to 1 and there
               are remaining ``Nones``, it redistributes the remaining
               probability equally.
            3. If there are no symbols with ``None`` probability and the
               total probability doesn't sum to 1, it rescales the
               probabilities so that they sum up to 1.

        Args:
            symbols (set[Symbol]): The set of symbols to validate.

        Returns:
            set[Symbol]: The validated set of symbols.
        """
        # Calculate the total probability and count those with ``None``
        total_p = sum(symbol.p for symbol in symbols if symbol.p is not None)
        none_count = len([symbol for symbol in symbols if symbol.p is None])

        if total_p == 1 and none_count > 0:
            for symbol in symbols:
                if symbol.p is not None:
                    # Halve probability
                    symbol.p /= 2
                else:
                    # Halve remaining probability, distribute equally
                    symbol.p = (1 - total_p / 2) / none_count
        elif total_p < 1 and none_count > 0:
            for symbol in symbols:
                if symbol.p is None:
                    # Distribute remaining probability equally
                    symbol.p = (1 - total_p) / none_count
        elif total_p > 1 and none_count > 0:
            # TODO: rescale probabilities
            raise ValueError("The total probability is greater than 1.")
        elif none_count == 0 and total_p != 1:
            for symbol in symbols:
                # Rescale probability
                symbol.p /= total_p

        return symbols

    def __repr__(self) -> str:
        return str(self)


class Terminal(Symbol):
    TYPE = Union[int, float, bool, list, np.ndarray, np.generic]

    def __init__(
        self,
        value: str | int | float | list,
        p: float | None = None,
    ):
        super().__init__(p=p)
        self.value = value

    # @classmethod
    # def get_default(cls) -> set["Terminal"]:
    #     terminals = {
    #         Terminal("x", 1 / 6),
    #         Terminal("y", 1 / 6),
    #         Terminal("z", 1 / 6),
    #         Terminal(0.0, 1 / 6),
    #         Terminal(0.1, 1 / 6),
    #         Terminal(1.0, 1 / 6),
    #     }
    #     return terminals

    @property
    def is_variable(self) -> bool:
        return isinstance(self.value, str)

    def __str__(self) -> str:
        return str(self.value)


class NonTerminal(Symbol):
    TYPE = Callable[..., Terminal.TYPE]

    def __init__(
        self,
        function: Callable,
        arity: int,
        name: str | None = None,
        p: float | None = None,
    ):
        super().__init__(p=p)
        self.function = function
        self.arity = arity
        self.name = name

    # @classmethod
    # def get_default(cls) -> set["NonTerminal"]:
    #     non_terminals = {
    #         cls(operator.add, 2, "+", 1 / 8),
    #         cls(operator.sub, 2, "-", 1 / 8),
    #         cls(operator.mul, 2, "*", 1 / 8),
    #         cls(operator.truediv, 2, "/", 1 / 8),
    #         cls(math.pow, 2, "pow", 1 / 8),
    #         cls(math.log, 1, "log", 1 / 8),
    #         cls(math.cos, 1, "cos", 1 / 8),
    #         cls(math.sin, 1, "sin", 1 / 8),
    #     }
    #     return non_terminals

    @property
    def is_flow(self) -> bool:
        return self.name in {"->", "if", "for"}

    def is_valid(self, *args) -> bool:
        return True

    def pre_validate(self, args: tuple[Any]) -> tuple[Any]:
        # Can be Any, e.g., flow non-terminals accept any
        # Non-flow terminals accept only tuple[Terminal.TYPE]
        return args

    def post_validate(self, arg: Terminal.TYPE) -> Terminal.TYPE:
        return arg

    # def __call__(self, *args) -> int | float:
    #     # Check for division by zero
    #     if self.symbol == "/" and args[1] == 0:
    #         return args[0]

    #     # Check for invalid values for log
    #     if self.symbol == "log" and args[0] <= 0:
    #         return 1

    #     # Check for invalid values for power
    #     if self.symbol == "pow" and args[0] < 0:
    #         return args[0]

    #     # Apply the function
    #     return self.function(*args)

    def __call__(self, *args, **kwargs) -> Terminal.TYPE:
        args = self.pre_validate(args)
        result = self.function(*args, **kwargs)
        result = self.post_validate(result)

        return result

    def __str__(self) -> str:
        return str(self.function) if self.name is None else self.name
