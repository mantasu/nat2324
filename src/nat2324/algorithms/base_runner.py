import time
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Any, Callable, Collection

import numpy as np
from tqdm.notebook import tqdm


class BaseRunner(ABC):
    def __init__(
        self,
        fitness_fn: Callable[..., int | float],
        N: int,
        parallelize_fitness: bool = False,
        seed: int | None = None,
    ):
        # Set population size, seed, and RNG
        self.fitness_fn = fitness_fn
        self.N = N
        self.parallelize_fitness = parallelize_fitness
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
            typing.Collection[typing.Any]: The population of individuals
            (initial solutions) to be optimized (evolved).
        """
        ...

    @abstractmethod
    def evolve(
        self,
        population: Collection[Any],
        fitnesses: Collection[float | int],
        *cache,
    ) -> (
        tuple[Collection[Any], Collection[float | int]]
        | tuple[Collection[Any], Collection[float | int], tuple[Any]]
    ):
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
            *cache: Additional optional arguments to be passed to the
                optimization function (cache).

        Returns:
            tuple[
                typing.Collection[typing.Any],
                typing.Collection[float | int],
            ] | tuple[
                typing.Collection[typing.Any],
                typing.Collection[float | int],
                tuple[typing.Any],
            ]: A tuple containing the new population, its fitness
            scores, and any additional data that should be cached and
            passed to the next call of :meth:`evolve`.
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

        with Pool(num_processes) as pool:
            # Create imap object that will apply workers
            imap = pool.map(worker_fn, iterable)

            if prog_bar:
                # If description is provided, wrap progress bar around
                desc = prog_bar if isinstance(prog_bar, str) else None
                imap = tqdm(imap, desc, len(iterable))

            # Actually apply workers
            results = list(imap)

        return results

    @classmethod
    def experiment_callback(
        cls,
        i: int | None = None,
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
            i (int | None, optional): The index of the experiment.
                Defaults to ``None``.
            **kwargs: The arguments to be passed to the optimization
                algorithm and the runner.

        Returns:
            tuple[float | int]: The result of the optimization
            algorithm.
        """
        # Pop keyword arguments for the runner
        run_kwargs = {
            "max_generations": kwargs.pop("max_generations", None),
            "num_evaluations": kwargs.pop("num_evaluations", None),
            "patience": kwargs.pop("patience", 100),
            "returnable": kwargs.pop("returnable", "score"),
            "is_maximization": kwargs.pop("is_maximization", True),
            "verbose": kwargs.pop("verbose", False),
        }

        # Initialize the algorithm and run it with seed i
        algorithm = cls(**{"seed": i, **kwargs})
        results = algorithm.run(**run_kwargs)

        return results

    def evaluate(self, population: Collection[Any]) -> np.ndarray:
        """Evaluate the fitness function on the population.

        This function evaluates the fitness function on the population
        and returns the fitness scores. It can be overridden in the
        child class if a different evaluation method is required.

        Args:
            population (typing.Collection[typing.Any]): The population
                of individuals to be optimized.

        Returns:
            numpy.ndarray: The fitness scores in the same order the
            individuals in the population. The shape of the array is
            ``(len(population),)`` and the dtype is either
            :class:`numpy.float64`. or :class:`numpy.int64`, unless
            ``self.fitness_fn`` returns a different type.
        """
        if self.parallelize_fitness:
            # Parallelize fitness function evaluations
            fitnesses = self.parallel_apply(self.fitness_fn, population)
        else:
            # Sequential fitness function evaluations
            fitnesses = np.apply_along_axis(self.fitness_fn, 1, population)

        return fitnesses

    def run(
        self,
        max_generations: int | None = None,
        num_evaluations: int | None = None,
        best_score: float | int | None = None,
        patience: int | None = 100,
        returnable: str | tuple[str] = "solution",
        is_maximization: bool = True,
        verbose: bool = True,
    ) -> Any | tuple[Any]:
        """Run the optimization algorithm.

        This function runs the optimization algorithm for a given number
        of generations. It returns the best individual and, optionally,
        its fitness score, the duration it took to run the algorithm,
        and the number of generations it took until the stopping
        criteria was met.

        The optimization algorithm is applied to the population
        iteratively until the maximum number of generations is reached
        or the early stopping criterion is met.

        Warning:
            If ``num_evaluations`` is provided, but it is not a multiple
            of ``self.N``, it will be rounded up to the nearest
            multiple of ``self.N``. So, if ``self.N = 10`` and
            ``num_evaluations = 15``, the algorithm will run for
            ``2 * self.N = 20`` evaluations, i.e., ``2`` generations.

        Args:
            max_generations (int, optional): The number of maximum
                generations to run. If not provided, ``num_evaluations``
                will be used instead. Defaults to ``None``.
            num_evaluations (int, optional): The number of maximum
                function evaluations to run. If both ``max_generations``
                and ``num_evaluations`` are not specified, it will
                default to ``self.N * 100``. Defaults to ``None``.
            best_score (float | int, optional): The best score of all
                time (throughout all generations). If provided, the
                algorithm will stop if it reaches this score. Defaults
                to ``None``.
            patience (int, optional): The number of generations without
                improvement before early stopping. Set to a value of
                ``None`` to run without early stopping. Defaults to
                ``100``.
            returnable (str | tuple[str], optional): The values to be
                returned. If :class:`str` is provided, only the
                corresponding value will be returned. If :class:`tuple`
                is provided, a :class:`tuple` of the corresponding
                values will be returned. Each value can be one of the
                following:

                    * ``"solution"`` - the earliest best solution of all
                      time (throughout all generations).
                    * ``"score"`` - the earliest best score of all time
                      (throughout all generations).
                    * ``"population"`` - the population at the earliest
                      best score.
                    * ``"fitnesses"`` - the fitness scores corresponding
                      to each individual in the population at the
                      earliest best score.
                    * ``"duration"`` - the duration (in seconds) it took
                      to reach the earliest best score.
                    * ``"num_generations"`` - the number of total
                      generations the earliest best score was reached.
                    * ``"num_evaluations"`` - the number of total
                      function evaluations the earliest best score was
                      reached. It is an approximate number and always a
                      multiple of ``self.N``.

                    It is also possible to prepend ``"last_"`` to any of
                    the above values to get the corresponding value at
                    the last generation. For example, ``"last_score"``
                    will return the best score at the last generation.
                    Defaults to ``"solution"``.
            is_maximization (bool, optional): Whether to maximize or
                minimize the fitness function. Defaults to ``True``.
            verbose (bool, optional): Whether to show the progress bar.
                Defaults to ``True``.

        Returns:
            typing.Any | tuple[tuple]: A single value or a tuple of
            values depending on the ``returnable`` argument. If
            ``returnable`` is a :class:`str`, a single value will be
            returned. If ``returnable`` is a :class:`tuple`, a tuple of
            values will be returned. By default, only the earliest best
            solution will be returned.
        """
        if max_generations is None and num_evaluations is None:
            # If both are not provided, default to 100 * self.N
            num_evaluations = self.N * 100

        if max_generations is None:
            # Compute max_generations from num_evaluations
            max_generations = round(num_evaluations / self.N)

        # Initialize population, fitnesses and other algorithm variables
        best_index_fn = np.argmax if is_maximization else np.argmin
        start_time, _patience, cache = time.time(), 0, tuple()
        population = self.initialize_population()
        fitnesses = self.evaluate(population)

        # Init trackable best measures
        trackable = {
            "solution": population[best_index_fn(fitnesses)],
            "score": fitnesses[best_index_fn(fitnesses)],
            "population": population,
            "fitnesses": fitnesses,
            "duration": time.time() - start_time,
            "num_generations": 0,
            "num_evaluations": 0,
        }

        # If description is provided, init and wrap progress bar around
        desc = "Current best N/A" if verbose else None
        pbar = tqdm(desc=desc, total=max_generations, disable=not verbose)

        for i in range(max_generations):
            # Evolve the population and get the next generation
            next_generation = self.evolve(population, fitnesses, *cache)

            if len(next_generation) == 2:
                # If only population and its fitness values are returned
                population, fitnesses, cache = next_generation, tuple()
            else:
                # If additional cached data is returned
                population, fitnesses, cache = next_generation

            # Get the best individual and its score
            best_idx = best_index_fn(fitnesses)
            solution = population[best_idx]
            score = fitnesses[best_idx]

            if verbose:
                # Update progress bar with the current best score
                pbar.set_description(f"Current best {score:.8f}")
                pbar.update()

            if (not is_maximization and score < trackable["score"]) or (
                is_maximization and score > trackable["score"]
            ):
                # Update bests
                trackable.update(
                    {
                        "solution": solution,
                        "score": score,
                        "population": population,
                        "fitnesses": fitnesses,
                        "duration": time.time() - start_time,
                        "num_generations": i + 1,
                        "num_evaluations": (i + 1) * self.N,
                    }
                )

                # Reset patience counter
                _patience = 0
            else:
                # Increment patience
                _patience += 1

            if (
                best_score is not None
                and (
                    (is_maximization and score >= best_score)
                    or (not is_maximization and score <= best_score)
                )
            ) or (patience is not None and _patience >= patience):
                # Patience exceeded
                break

        # Close the progress bar
        pbar.close()

        if max_generations > 0:
            # After all iterations
            trackable.update(
                {
                    "last_solution": solution,
                    "last_score": score,
                    "last_population": population,
                    "last_fitnesses": fitnesses,
                    "last_duration": time.time() - start_time,
                    "last_num_generations": i + 1,
                    "last_num_evaluations": (i + 1) * self.N,
                }
            )
        else:
            # If max_generations is 0, set last values to initial values
            trackable.update({f"last_{k}": v for k, v in trackable.items()})

        if isinstance(returnable, str):
            # Return corresponding value
            return trackable[returnable]
        else:
            # Return a tuple of corresponding multiple values
            return tuple(trackable[key] for key in returnable)
