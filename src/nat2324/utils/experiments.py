import ast
import os
import random
from collections import defaultdict
from itertools import product
from multiprocessing import Manager
from typing import Any, Callable

import numpy as np
from tqdm.contrib.concurrent import process_map


def make_name(arguments, default_arguments):
    name = ""

    for key, val in default_arguments.items():
        name += key + "=" + str(val) + ";"

    name += "[" + ",".join([str(k) for k in arguments.keys()]) + "]" + ".npz"

    return name


def worker(args):
    _, i, lock, callback, kwargs, num_runs = args
    new_kwargs = {}

    for key, val in kwargs.items():
        if isinstance(key, tuple):
            new_kwargs.update(dict(zip(key, val)))
        else:
            new_kwargs[key] = val

    result_list = []
    for _ in range(num_runs):
        with lock:
            # Update seed & progress
            random.seed(i.value)
            np.random.seed(i.value)
            os.environ["PYTHONASHSEED"] = str(i.value)
            i.value += 1

        # Run the callback function; add the result to the list
        result = callback(i.value, **new_kwargs)
        result = result if isinstance(result, tuple) else [result]
        result_list.append(result)

    # Return the mean of the results
    return np.mean(result_list, axis=0).tolist()


def run_optimization_experiment(
    callback: Callable[..., float | int | tuple[float | int]],
    arguments: dict[str | tuple[str], list[Any]],
    default_arguments: dict[str, Any] = {},
    num_runs: int = 1,
    chunksize: int = 1,
    is_cartesian_product: bool = False,
    dirname: str | None = None,
) -> dict[str | tuple[str], np.ndarray]:
    if dirname is not None:
        filename = make_name(arguments, default_arguments)
        filepath = os.path.join(dirname, filename)
    else:
        filepath = None

    if filepath is not None and os.path.exists(filepath):
        # Load the results from the file if it exists
        data = np.load(filepath)
        results = {}

        for key in data.files:
            try:
                # Try to parse the key as a Python literal
                results[ast.literal_eval(key)] = data[key]
            except ValueError:
                # If it fails, use the key as a string
                results[key] = data[key]

        return results

    results = defaultdict(lambda: [])

    with Manager() as manager:
        i = manager.Value("i", 0)
        lock = manager.Lock()

        args = []
        if is_cartesian_product:
            for value_combination in product(*arguments.values()):
                arg_dict = dict(zip(arguments.keys(), value_combination))
                args.append(
                    (
                        tuple(arguments.keys()),
                        i,
                        lock,
                        callback,
                        {**default_arguments, **arg_dict},
                        num_runs,
                    )
                )
        else:
            for key, values in arguments.items():
                args.extend(
                    [
                        (
                            key,
                            i,
                            lock,
                            callback,
                            {**default_arguments, key: argument},
                            num_runs,
                        )
                        for argument in values
                    ]
                )

        result_list = process_map(worker, args, chunksize=chunksize)

        # Add the results to the dictionary
        for res, arg in zip(result_list, args):
            results[arg[0]].append(res)

        for key, val in results.items():
            results[key] = np.array(val)

            if is_cartesian_product:
                # Reshape the results into a multi-dimensional numpy array
                shape = [len(values) for values in arguments.values()]
                results[key] = np.reshape(results[key], (*shape, -1))

    if filepath is not None:
        # Save the results to the file (make dir if it needed)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, **{str(k): v for k, v in results.items()})

    return results
