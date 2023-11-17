from abc import ABC
from typing import Any, Callable, Union

import numpy as np


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

    @property
    def is_flow(self) -> bool:
        return self.name in {"->", "if", "for"}

    def is_valid(self, *args) -> bool:
        return True

    def validate(self, args: tuple[Any]) -> tuple[Any]:
        # Can be Any, e.g., flow non-terminals accept any
        # Non-flow terminals accept only tuple[Terminal.TYPE]
        return args

    def __call__(self, *args, **kwargs) -> Terminal.TYPE:
        args = self.validate(args)
        result = self.function(*args, **kwargs)

        return result

    def __str__(self) -> str:
        return str(self.function) if self.name is None else self.name
