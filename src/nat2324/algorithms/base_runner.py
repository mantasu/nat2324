import time
import numpy as np

from tqdm.notebook import tqdm
from functools import partial
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import Callable, Collection, Any


class BaseRunner(ABC):
    def __init__(self, N: int, seed: int | None = None):
        # Set population size, seed, and RNG
        self.N = N
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def initialize_population(self) -> Collection[Any]:
        """Generate initial population.

        This function should be implemented by the child class. It
        should return a collection of individuals, e.g., a list of
        :class:`numpy.ndarray` objects. It is the initial population
        to be used as input to the optimization algorithm that will
        evolve it over generations.

        Returns:
            typing.Collection[Any]: The population of individuals
            (initial solutions) to be optimized (evolved).
        """
        ...

    @abstractmethod
    def evolve(
        self,
        population: Collection[Any],
        fitnesses: Collection[float | int],
        *cache,
    ) -> tuple[Collection[Any], Collection[float | int], tuple[Any]]:
        """Optimization method to be implemented by the child class.

        This function should be implemented by the child class. It takes
        the population (an iterable of individuals) and their fitness
        values as inputs and returns a new evolved population, along
        with new fitness scores for each individual.

        Args:
            population (typing.Collection[typing.Any]): The population
                of individuals to be optimized.
            fitnesses (typing.Collection[float | int]): The fitness
                scores of the individuals in the population. The fitness
                scores must be in the same order as the individuals in
                the population.
            *cache: Additional arguments to be passed to the
                optimization function (cache).

        Returns:
            tuple[typing.Collection[typing.Any], typing.Collection[float | int], tuple[typing.Any]]:
            A tuple containing the new population, its fitness scores,
            and any additional data that should be cached and passed to
            the next call of ``evolve``.
        """
        ...
    
    @staticmethod
    def parallel_apply(
        worker_fn: Callable[..., Any],
        iterable: Collection[Any],
        num_processes: int = cpu_count(),
        prog_bar: bool | str = False,
        **kwargs,
    ) -> list[Any]:
        """Apply worker function to each element in iterable.

        Applies worker function to each element in iterable and returns
        the results. This function can run on multiple processes
        if the device supports it. The returned results are in the same
        order as the elements in the iterable object.

        Warning:
            The worker function must be picklable, i.e., it must be
            defined at the top level of a module. So it cannot refer to
            any local variables, i.e., the variables it access must not
            be declared in any unknown local scope.

        Args:
            worker_fn (typing.Callable[..., typing.Any]): The worker
                function, e.g., fitness function. It can map from any
                input, e.g., :class:`numpy.ndarray`, to any output,
                e.g., to :class:`float` or another
                :class:`numpy.ndarray`.
            iterable (typing.Collection[typing.Any]): The elements to
                which the worker function should be applied. It must
                support protocols like ``len``, ``in``, i.e., it must be
                of type :class:`list`, :class:`numpy.ndarray` or
                similar. Each element can be of any type.
            num_processes (int, optional): The number of cpu units to
                run the function evaluations on. Defaults to
                ``cpu_count()``, which is the maximum number of CPUs on
                the machine.
            prog_bar (str | bool, optional): Whether to show the
                progress bar. If :class:`str` is provided, progress bar
                will be shown automatically with the description
                specified in the provided string (except `""`). Defaults
                to ``False``.
            **kwargs: Additional arguments to be passed to the worker
                function.

        Returns:
            list[typing.Any]: The results of the worker function
            evaluations. The results are in the same order as the
            elements in the iterable and of the same type as
            returned by ``worker_fn``.
        """
        # Wrap with partial to accept kwargs
        worker_fn = partial(worker_fn, **kwargs)

        with ThreadPool(num_processes) as pool:
            # Create imap object that will apply workers
            imap = pool.map(worker_fn, iterable)
            
            if prog_bar:
                # If description is provided, wrap progress bar around
                desc = prog_bar if isinstance(prog_bar, str) else None
                imap = tqdm.tqdm(imap, desc, len(iterable))
            
            # Actually apply workers
            results = list(imap)
        
        return results
    
    @classmethod
    def experiment_callback(
        cls,
        i: int,
        **kwargs,
    ) -> tuple[float | int]:
        """Callback function to be used in experiments.

        This function is used to run the optimization algorithm in
        parallel in experiments. It is a wrapper around the
        :meth:`run` method that accepts a dictionary of arguments
        instead of a single argument. It is used in experiments to
        run the optimization algorithm in parallel with different
        arguments.

        Note:
            Passed kwargs are mutated, thus, if they have to be reused,
            make sure to make a deep copy of them before passing to this
            method.

        Args:
            i (int): The index of the experiment.
            **kwargs: The arguments to be passed to the optimization
                algorithm and the runner.

        Returns:
            tuple[float | int]: The result of the optimization
            algorithm.
        """
        # Pop keyword arguments for the runner
        run_kwargs = {
            "max_generations": kwargs.pop("max_generations", 500),
            "patience": kwargs.pop("patience", 100),
            "extra_return": kwargs.pop("extra_return", ("score",)),
            "is_maximization": kwargs.pop("is_maximization", True),
            "verbose": kwargs.pop("verbose", False),
        }

        # By default experiments don't care about solution
        return_solution = kwargs.pop("return_solution", False)

        # Initialize the algorithm and run
        algorithm = cls(seed=i, **kwargs)
        results = algorithm.run(**run_kwargs)

        return results[0 if return_solution else 1:]

    def run(
        self,
        max_generations: int = 500,
        patience: int | None = 100,
        extra_return: tuple[str] = tuple(),
        is_maximization: bool = True,
        verbose: bool = True,
    ) -> tuple[Any | tuple]:
        """Run the optimization algorithm.

        This function runs the optimization algorithm for a given number
        of generations. It returns the best individual and, optionally,
        its fitness score, the duration it took to run the algorithm,
        and the number of generations it took until the stopping
        criteria was met.
        
        The optimization algorithm is applied to the population
        iteratively until the maximum number of generations is reached
        or the early stopping criterion is met.

        Args:
            max_generations (int, optional): The number of maximum
                generations to run. Defaults to ``500``.
            patience (int, optional): The number of generations without
                improvement before early stopping. Set to a value of
                ``None`` to run without early stopping. Defaults to
                ``100``.
            extra_return (tuple, optional): Whether to return additional
                values. The values can be:

                    * ``"score"`` - the score of the best solution
                    * ``"duration"`` - the duration (in seconds) it took
                      to run the algorithm
                    * ``"num_gens"`` - the number of total generations
                      it took until the termination condition was
                      reached
                    
                    Defaults to ``tuple()``.
            is_maximization (bool, optional): Whether to maximize or
                minimize the fitness function. Defaults to ``True``.
            verbose (bool, optional): Whether to show the progress bar.
                Defaults to ``True``.

        Returns:
            tuple[typing.Any | tuple]: The best individual and, if
            specified, its fitness score, the duration it took to run
            the algorithm, and the number of generations it took until
            the stopping criteria was met.
        """
        # Init trackable best measures
        best = {
            "solution": None,
            "score": -np.inf if is_maximization else np.inf,
            "duration": 0,
            "num_gens": 0,
        }

        # Generate initial population and init patience counter
        population, cache = self.initialize_population(), tuple()
        fitness = [(-1 if is_maximization else 1) * np.inf] * len(population)
        start_time, _patience = time.time(), 0

        # If description is provided, init and wrap progress bar around
        desc = "Current best 0.00000000" if verbose else None
        pbar = tqdm(desc=desc, total=max_generations, disable=not verbose)

        for i in range(max_generations):
            # Evolve population and get the best individual + its score
            population, fitness, cache = self.evolve(population, fitness, *cache)
            idx = np.argmax(fitness) if is_maximization else np.argmin(fitness)
            solution, score = population[idx], fitness[idx]

            if patience is not None and _patience >= patience:
                # Patience exceeded
                break
            elif (not is_maximization and score < best["score"]) \
                  or (is_maximization and score > best["score"]):
                # Update bests
                best.update({
                    "solution": solution,
                    "score": score,
                    "duration": time.time() - start_time,
                    "num_gens": i + 1,
                })

                # Reset patience counter
                _patience = 0
            else:
                # Increment patience
                _patience += 1

            if verbose:
                # Update progress bar with the current best fitness score
                pbar.set_description(f"Current best {best['score']:.8f}")
                pbar.update()
        
        # Close the progress bar
        pbar.close()
        
        if patience is None:
            # Just get the last as best
            best.update({
                "solution": solution,
                "score": score,
                "duration": time.time() - start_time,
                "num_gens": i + 1,
            })
        
        # Create returnables with solution
        returnables = [best["solution"]]
        
        if len(extra_return) == 0:
            # Only best individual
            return returnables[0]

        for returnable in extra_return:
            if returnable not in ("score", "duration", "num_gens"):
                # Invalid extra return value
                raise ValueError(f"Invalid extra return value: {returnable}")
            else:
                # Add extra return value
                returnables.append(best[returnable])
        
        return tuple(returnables)
