import numpy as np


def is_nested_list(a: list) -> bool:
    """Checks whether the given list is nested.

    Args:
        a (list): The list to check.

    Returns:
        bool: Whether the given list is nested.
    """
    return any(isinstance(i, list) for i in a)


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

    for arg in args:
        # Convert to regular type (to normalize, e.g., int vs np.int64)
        arg = arg.tolist() if isinstance(arg, (np.ndarray, np.generic)) else arg

        if not isinstance(arg, (int, float, bool)) or (
            isinstance(arg, list) and is_nested_list(arg)
        ):
            # Replace the argument with the default value
            valid_args.append(default)
        elif isinstance(arg, list):
            # Convert the argument to a numpy array
            valid_args.append(np.array([bool(arg) if as_bool else arg]))
            min_len = min(min_len, len(arg))
        else:
            # Leave the argument as it is (convert maybe to bool)
            valid_args.append(bool(arg) if as_bool else arg)

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
