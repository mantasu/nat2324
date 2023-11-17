import operator
from typing import Callable

from .converter import as_list, as_scalar, numpify
from .symbol import NonTerminal, Terminal


class Arithmetic(NonTerminal):
    @classmethod
    def add(cls) -> NonTerminal:
        return cls(operator.add, 2, "+")

    @classmethod
    def sub(cls) -> NonTerminal:
        return cls(operator.sub, 2, "-")

    @classmethod
    def mul(cls) -> NonTerminal:
        return cls(operator.mul, 2, "*")

    @classmethod
    def div(cls) -> NonTerminal:
        return cls(operator.truediv, 2, "/")

    @classmethod
    def pow(cls) -> NonTerminal:
        return cls(operator.pow, 2, "^")

    @classmethod
    def mod(cls) -> NonTerminal:
        return cls(operator.mod, 2, "%")

    def validate(
        self,
        args: tuple[Terminal.TYPE],
    ) -> tuple[int | float, int | float]:
        if len(args) == 0:
            # No arguments
            args = (0, 0)
        elif len(args) == 1:
            # One argument
            args = args * 2
        elif len(args) > 2:
            # > 2 arguments
            args = args[:2]

        # Convert to valid scalars
        args = as_scalar(args)

        if abs(args[0]) > 10e6 or abs(args[1]) > 10e6:
            # Encourage correctness
            args = (0, 0)

        if (self.name in {"/", "%"} and abs(args[1]) < 1e-6) or (
            self.name == "pow" and args[1] > 100
        ):
            # Fix division and power
            args = (0, 1)

        return args


class Comparison(NonTerminal):
    @classmethod
    def less_than(cls) -> NonTerminal:
        return cls(operator.lt, 2, "<")

    @classmethod
    def less_than_or_equal_to(cls) -> NonTerminal:
        return cls(operator.le, 2, "<=")

    @classmethod
    def greater_than(cls) -> NonTerminal:
        return cls(operator.gt, 2, ">")

    @classmethod
    def greater_than_or_equal_to(cls) -> NonTerminal:
        return cls(operator.ge, 2, ">=")

    @classmethod
    def equal_to(cls) -> NonTerminal:
        return cls(operator.eq, 2, "==")

    @classmethod
    def not_equal_to(cls) -> NonTerminal:
        return cls(operator.ne, 2, "!=")


class Logic(NonTerminal):
    @classmethod
    def and_(cls) -> NonTerminal:
        return NonTerminal(operator.and_, 2, "&&")

    @classmethod
    def or_(cls) -> NonTerminal:
        return NonTerminal(operator.or_, 2, "||")

    @classmethod
    def not_(cls) -> NonTerminal:
        return NonTerminal(operator.not_, 1, "!")


class Flow(NonTerminal):
    @staticmethod
    def before_after_clb(
        before: Callable[..., Terminal.TYPE],
        after: Callable[..., Terminal.TYPE],
        **kwargs,
    ) -> Terminal.TYPE:
        # Run before & after
        _ = before(**kwargs)
        a = after(**kwargs)

        return a

    @staticmethod
    def if_else_clb(
        condition: Callable[..., bool],
        if_true: Callable[..., Terminal.TYPE],
        if_false: Callable[..., Terminal.TYPE],
        **kwargs,
    ) -> Terminal.TYPE:
        # Get the condition
        condition = condition(**kwargs)
        condition = numpify(condition, default=0, as_bool=True).all()

        return if_true(**kwargs) if condition(**kwargs) else if_false(**kwargs)

    @staticmethod
    def for_loop_clb(
        num_iters: Callable[..., int],
        body: Callable[..., Terminal.TYPE],
        **kwargs,
    ) -> Terminal.TYPE:
        kwargs.setdefault("level", 1)

        if kwargs["level"] > 1:
            return body(**kwargs)
        else:
            kwargs["level"] += 1

        # Get the number of iterations
        num_iters = num_iters(**kwargs)
        num_iters = as_scalar(num_iters, bounds=(0, 1000), type=int)

        if num_iters == 0:
            # Encourage correctness
            return num_iters

        for _ in range(num_iters):
            # Repeat the execution
            result = body(**kwargs)

        kwargs.pop("level")

        return result

    @classmethod
    def before_after(cls) -> NonTerminal:
        return NonTerminal(cls.before_after_clb, 2, "->")

    @classmethod
    def if_else(cls) -> NonTerminal:
        return NonTerminal(cls.if_else_clb, 3, "if")

    @classmethod
    def for_loop(cls) -> NonTerminal:
        return NonTerminal(cls.for_loop_clb, 2, "for")


class Indexable:
    @staticmethod
    def get_clb(
        array: list,
        index: int,
    ) -> Terminal.TYPE:
        # Convert to proper types
        array = as_list(array)
        index = as_scalar(index, bounds=(-len(array), len(array) - 1), type=int)

        if len(array) == 0:
            # Empty array
            return index

        return array[index]

    @staticmethod
    def set_clb(
        array: list,
        index: int,
        value: Terminal.TYPE,
    ) -> list:
        if not isinstance(array, list) or len(array) == 0:
            # Array is not a list or is empty
            return array

        if not isinstance(index, (int, float, list)) or (
            (isinstance(index, list)) and len(index) == 0
        ):
            # Index is not a number or is an empty list
            return array

        if isinstance(index, (int, float)):
            index = [index]

        for i in index:
            if index > 0 and index >= len(array):
                # Index is out of bounds (too big)
                i = -1
            elif index < 0 and abs(index) > len(array):
                # Index is out of bounds (too low)
                i = 0

            # Set the element in the array at the given index
            array[int(i)] = value

        return array

    @staticmethod
    def push_clb(
        array: list,
        value: Terminal.TYPE,
    ) -> list:
        # Convert to proper types
        array = as_list(array, default=None)
        value = as_scalar(value, default=None)

        if array is None or value is None:
            # Not a list
            return array

        # Append the value
        array.append(value)

        return array

    @staticmethod
    def pop_clb(
        array: list,
    ) -> Terminal.TYPE:
        if not isinstance(array, list):
            return array

        return array.pop()

    @classmethod
    def get(cls) -> NonTerminal:
        return NonTerminal(cls.get_clb, 2, "get")

    @classmethod
    def set(cls) -> NonTerminal:
        return NonTerminal(cls.set_clb, 3, "set")

    @classmethod
    def push(cls) -> NonTerminal:
        return NonTerminal(cls.push_clb, 2, "push")

    @classmethod
    def pop(cls) -> NonTerminal:
        return NonTerminal(cls.pop_clb, 1, "pop")
