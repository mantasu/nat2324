import numpy as np

from typing import Any, Callable
from collections import defaultdict
from multiprocessing import Manager
from itertools import product

from tqdm.contrib.concurrent import process_map


def worker(args):
    _, i, lock, callback, kwargs, num_runs = args
    result_list = []
    for _ in range(num_runs):
        # Run the callback function; add the result to the list
        result = callback(i.value, **kwargs)
        result = result if isinstance(result, tuple) else [result]
        result_list.append(result)
        
        # Update progress
        with lock:
            i.value += 1
            
    # Return the mean of the results
    return np.mean(result_list, axis=0).tolist()

def run_optimization_experiment(
    callback: Callable[..., float | int | tuple[float | int]],
    arguments: dict[str, list[Any]],
    default_arguments: dict[str, Any] = {},
    num_runs: int = 1,
    is_cartesian_product: bool = False,
) -> dict[str | tuple[str], np.ndarray]:
    results = defaultdict(lambda: [])

    with Manager() as manager:
        i = manager.Value('i', 0)
        lock = manager.Lock()
        
        args = []
        if is_cartesian_product:
            for value_combination in product(*arguments.values()):
                arg_dict = dict(zip(arguments.keys(), value_combination))
                args.append((tuple(arguments.keys()), i, lock, callback, {**default_arguments, **arg_dict}, num_runs))
        else:
            for key, values in arguments.items():
                args.extend([(key, i, lock, callback, {**default_arguments, key: argument}, num_runs) for argument in values])
        
        result_list = process_map(worker, args)
                
        # Add the results to the dictionary
        for res, arg in zip(result_list, args):
            results[arg[0]].append(res)
        
        for key, val in results.items():
            results[key] = np.array(val)

            if is_cartesian_product:
                # Reshape the results into a multi-dimensional numpy array
                shape = [len(values) for values in arguments.values()]
                results[key] = np.reshape(results[key], (*shape, -1))

        # if is_cartesian_product:
        #     # Reshape the results into a multi-dimensional numpy array
        #     shape = [len(values) for values in arguments.values()]
        #     for key in results.keys():
        #         results[key] = np.reshape(results[key], shape).tolist()
    
    return results
                