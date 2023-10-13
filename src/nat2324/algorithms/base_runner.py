import tqdm
import numpy as np

from functools import partial
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from typing import Callable, Collection, Any


class BaseRunner(ABC):
    @abstractmethod
    def initialize_population(self) -> tuple[Collection[Any], tuple[Any]]:
        """Generate initial population.

        This function should be implemented by the child class. It
        should return a collection of individuals, e.g., a list of
        :class:`numpy.ndarray` objects. It is the initial population
        to be used as input to the optimization algorithm that will
        evolve it over generations.

        Returns:
            tuple[typing.Collection[Any], tuple[Any]]: The population of
            individuals to be optimized and any additional data that
            should be cached and passed to the optimization function.
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

    def run(
        self,
        is_maximization: bool = True,
        max_generations: int = 500,
        patience: int | None = 100,
        return_score: bool = False,
        return_duration: bool = False,
        return_num_gens: bool = False,
        prog_bar: str | bool = "Current best fitness: ",
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
            is_maximization (bool, optional): Whether to maximize or
                minimize the fitness function. Defaults to ``True``.
            max_generations (int, optional): The number of maximum
                generations to run. Defaults to ``500``.
            patience (int, optional): The number of generations without
                improvement before early stopping. Set to a value of
                ``None`` to run without early stopping. Defaults to
                ``100``.
            return_score (bool, optional): Whether to return the score
                of the best solution. Defaults to ``False``.
            return_duration (bool, optional): Whether to return the
                duration (in seconds) it took to run the algorithm.
                Defaults to ``False``.
            return_num_gens (bool, optional): The number of total
                generations it took until the termination condition was
                reached. Defaults to ``False``.
            prog_bar (str | bool, optional): Whether to show the
                progress bar. If :class:`str` is provided, progress bar
                will be shown automatically with the description
                specified in the provided string (except `""`). Defaults
                to ``"Current best fitness: "``.

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
        population, cache = self.initialize_population()
        fitness = [(-1 if is_maximization else 1) * np.inf] * len(population)
        _patience = 0

        # If description is provided, init and wrap progress bar around
        desc = prog_bar + "N/A" if isinstance(prog_bar, str) else None
        pbar = tqdm.tqdm(range(max_generations), desc, max_generations)

        if not prog_bar:
            # Disable progress bar
            pbar.disable = True

        for i in pbar:
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
                    "duration": pbar.last_print_t - pbar.start_t,
                    "num_gens": i + 1,
                })

                # Reset patience counter
                _patience = 0
            else:
                # Increment patience
                _patience += 1

            # Update progress bar with the current best fitness score
            pbar.set_description(f"Current best {best['score']:.8f}")
        
        if patience is None:
            # Just get the last as best
            best.update({
                "solution": solution,
                "score": score,
                "duration": pbar.last_print_t - pbar.start_t,
                "num_gens": i + 1,
            })
        
        # Create returnables with solution
        returnables = [best["solution"]]
        
        if not return_score and not return_duration and not return_num_gens:
            # Only best individual
            return returnables[0]
        
        if return_score:
            # Aso return best fitness score
            returnables.append(best["score"])
        
        if return_duration:
            # Return the duration it took to run the algorithm
            returnables.append(best["duration"])
        
        if return_num_gens:
            # Num generations took until termination
            returnables.append(best["num_gens"])
        
        return tuple(returnables)
