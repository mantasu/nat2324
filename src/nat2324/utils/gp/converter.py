from typing import Any, Collection

import numpy as np

from .symbol import Terminal


def is_nested_list(a: Collection) -> bool:
    """Checks whether the given list is nested.

    Args:
        a (typing.Collection): The list to check.

    Returns:
        bool: Whether the given list is nested.
    """
    return any(isinstance(i, Collection) for i in a)


def numpify(
    args: int | float | bool | np.ndarray | tuple[int | float | bool | np.ndarray],
    default: int | float = 0,
    as_bool: bool = False,
) -> int | float | bool | np.ndarray | tuple[int | float | bool | np.ndarray]:
    """Converts the given arguments to numpy arrays if they are lists.

    This functions either leaves the arguments as they are (if they are
    not lists), or converts them to numpy arrays (if at least one of
    them is a list). If an argument is not of type ``int``, ``float``,
    ``bool``, or is a nested list, it is replaced with the ``default``
    value.

    Warning:
        If some arguments are lists and they have different lengths
        (>1), the longer lists will be truncated to the length of the
        shortest list.

    Args:
        args (int | float | bool | numpy.ndarray | list[int | float | bool]
        | tuple[int | float | bool | numpy.ndarray, list[int | float | bool]]):
            The arguments to validate. Can also be a single argument.
        default (int | float, optional): The default value. Defaults to
            ``0``.
        as_bool (bool, optional): Whether to convert the arguments to
            boolean values. Defaults to ``False``.

    Returns:
        int | float | bool | numpy.ndarray | tuple[int | float | bool | numpy.ndarray]:
        The converted arguments.
    """
    # Check if the arguments are a tuple
    is_tuple = isinstance(args, tuple)

    if not is_tuple:
        # Convert the arguments to a tuple
        args = (args,)

    # init valid args
    min_len = np.inf
    valid_args = []

    # print("My args", args)

    for arg in args:
        # Convert to regular type (to normalize, e.g., int vs np.int64)
        arg = arg.tolist() if isinstance(arg, (np.ndarray, np.generic)) else arg

        if not isinstance(arg, Terminal.TYPE) or (
            isinstance(arg, Collection) and is_nested_list(arg)
        ):
            # Replace the argument with the default value
            valid_args.append(default)
        elif isinstance(arg, Collection) and len(arg) == 0:
            # Replace the argument with the default value
            valid_args.append(0)
        elif isinstance(arg, Collection):
            # Convert the argument to a numpy array
            valid_args.append(np.array([bool(a) if as_bool else a for a in arg]))
            min_len = min(min_len, len(arg))
        else:
            # Leave the argument as it is (convert maybe to bool)
            valid_args.append(bool(arg) if as_bool else arg)

    # print(min_len, valid_args)

    if min_len < np.inf:
        for i, arg in enumerate(valid_args):
            if isinstance(arg, np.ndarray):
                # Truncate the numpy array to the length of the shortest list
                valid_args[i] = arg[:min_len]
            else:
                # Repeat the argument to the length of the shortest list
                valid_args[i] = np.array([arg] * min_len)
    else:
        # Convert the arguments to numpy arrays of length 1
        valid_args = [np.array([arg]) for arg in valid_args]

    return tuple(valid_args) if is_tuple else valid_args[0]


def as_scalar(
    args: Any | tuple[Any],
    default: int | float | bool = 0,
    bounds: tuple[int, int] | tuple[float, float] | tuple[bool, bool] | None = None,
    type: type | None = None,
) -> int | float | bool | tuple[int | float | bool]:
    """Verifies the given arguments are scalars.

    This functions only considers arguments of type :class:`int`,
    :class:`float`, :class:`bool`, or :class:`numpy.generic`, and
    returns ``default`` for all other types. If ``bounds`` are given,
    the arguments are clipped to the given bounds. If ``type`` is given,
    the arguments are converted to the given type.

    Note:
        If either of the arguments are of type :class:`numpy.generic`,
        they are converted to regular types.

    Args:
        args (typing.Any | tuple[typing.Any]): The arguments to
            validate. Can also be a single argument.
        default (int | float | bool | None, optional): The default value
            to use if the argument is not of type :class:`int`,
            :class:`float`, :class:`bool`, or :class:`numpy.generic`.
            Defaults to ``0``.
        bounds (tuple[int, int] | tuple[float, float] | tuple[bool, bool] | None, optional):
            The bounds to clip the value to. Defaults to ``None``.
        type (type | None, optional): The type to convert the value to.
        This is applied before clipping. Defaults to ``None``.

    Returns:
        int | float | bool | None | tuple[int | float | bool | None]:
        A tuple of converted arguments or a single argument if ``args``
        was a single argument.
    """
    # Check if the arguments are a tuple
    is_tuple = isinstance(args, tuple)
    args = args if is_tuple else (args,)
    valid_args = []

    for arg in args:
        if isinstance(arg, np.generic):
            # Convert to regular type
            arg = arg.item()

        if not isinstance(arg, (int, float, bool)):
            # Append the default value
            valid_args.append(default)
            continue

        if type is not None:
            # Convert to the given type
            arg = type(arg)

        if bounds is not None:
            # Clip the argument to the given bounds
            arg = min(max(arg, bounds[0]), bounds[1])

        # Append the argument
        valid_args.append(arg)

    return tuple(valid_args) if is_tuple else valid_args[0]


def as_list(
    args: Any | tuple[Any],
    default: list[int | float | bool] = [],
    max_length: int | None = None,
) -> list[int | float | bool] | tuple[list[int | float | bool]]:
    """Verifies the given arguments are lists.

    This functions only considers arguments of type :class:`list` or
    :class:`numpy.ndarray`, both of which must be 1D non-nested arrays
    and returns ``default`` for all other cases. If ``max_length`` is
    given, the arguments are truncated to the given length.

    Args:
        args (typing.Any | tuple[typing.Any]): The arguments to
            validate. Can also be a single argument.
        default (list[int | float | bool], optional): The default value
            to use if the argument is not valid. Defaults to ``[]``.
        max_length (int | None, optional): The maximum length of the
            list. Defaults to ``None``.

    Returns:
        list[int | float | bool] | tuple[list[int | float | bool]]:
        A tuple of converted arguments or a single argument if ``args``
        was a single argument.
    """
    # Check if the arguments are a tuple
    is_tuple = isinstance(args, tuple)
    args = args if is_tuple else (args,)
    valid_args = []

    for arg in args:
        if isinstance(arg, np.ndarray):
            # Convert to regular type
            arg = arg.tolist()

        if not isinstance(arg, list) or is_nested_list(arg):
            # Append the default value
            valid_args.append(default)
            continue

        # TODO: check if all elements are of type int, float, or bool

        if max_length is not None:
            # Truncate the argument to the given length
            arg = arg[:max_length]

        # Append the argument
        valid_args.append(arg)

    return tuple(valid_args) if is_tuple else valid_args[0]
