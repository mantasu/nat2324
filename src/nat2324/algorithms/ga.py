from typing import Any, Callable, Collection

import numpy as np
from scipy.special import softmax

from ..utils.decorators import override, private, submethod, utilmethod
from .base_runner import BaseRunner


class BinaryGeneticAlgorithm(BaseRunner):
    """Optimizer based on *binary* genetic algorithm.

    A binary genetic algorithm (GA) implementation for solving
    optimization problems. This implementation uses roulette wheel
    sampling for selecting individuals for crossover and mutation.

    Note:
        A better implementation would be to generalize this to a general
        Genetic Algorithm that can deal with variable discrete (and even
        continuos) cases, with different crossover and mutation types.
        However, for this assignment, it is sufficient to use a binary
        GA.

    Args:
        fitness_fn (Callable[[numpy.ndarray], float]): A fitness
            function that takes a numpy array of shape ``(D,)`` as
            input and returns a scalar value representing the fitness
            score of the individual.
        N (int, optional): The number of individuals. Defaults to
            ``100``.
        D (int, optional): The number of dimensions for each individual.
            Warning: this must be square-rootable. Defaults to ``9``.
        selection_type (str, optional): The type of selection to use.
            It must be one of the following: "roulette_wheel" or
            "tournament". Defaults to ``"tournament"``.
        mutation_type (str, optional): The type of mutation to use. It
            must be one of the following: "bit_flip", "row_col_shuffle",
            or "mixed". Defaults to ``"mixed"``.
        crossover_type (str, optional): The type of crossover to use. It
            must be one of the following: "n_point", "row_col_swap", or
            "mixed". Defaults to ``"mixed"``.
        use_refit (bool, optional): Whether to use refit operation. If
            ``True``, the fitness of the local best solutions will be
            refit (decayed over time) to avoid stagnation. Defaults to
            ``True``.
        use_escape (bool, optional): Whether to use escape operation. If
            ``True``, the algorithm will escape local optima by
            replacing the population with random individuals if it
            detects that a large fraction of individuals is close to the
            mean of the local best solution or the mean of the
            population. Defaults to ``True``.
        parallelize_fitness (bool, optional): Whether to parallelize the
            computation of the fitness scores for each individual.
            Defaults to ``False``, which should be set if the fitness
            function is not that complex, in which case running on a
            single process would be more efficient.
        seed (int | None, optional): The seed for the random number
            generator. Defaults to ``None``, which means on every run,
            the results will be random.
    """

    VALID_SELECTION_TYPES = ["roulette_wheel", "tournament"]
    VALID_CROSSOVER_TYPES = ["n_point", "row_col_swap"]
    VALID_MUTATION_TYPES = ["bit_flip", "row_col_shuffle"]

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        N: int = 1000,
        D: int = 9,
        selection_type: str = "tournament",
        mutation_type: str = "mixed",
        crossover_type: str = "mixed",
        use_refit: bool = True,
        use_escape: bool = True,
        parallelize_fitness: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(fitness_fn, N, parallelize_fitness, seed=seed)

        # Initialize variables
        self.D = D
        self.K = np.sqrt(self.D).astype(np.int64)
        self.use_refit = use_refit
        self.use_escape = use_escape
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type

        # Initialize extra parameters
        self._init_params(kwargs)

    @private
    def _init_params(self, kwargs: dict[str, Any]):
        """Initializes the parameters for the GA.

        Assigns extra optional parameters for the GA. The parameters
        that can be assigned are:

            * ``stop_at_1``: Whether to stop refitting when a fitness
              of 1 is reached. Defaults to ``True``.
            * ``decay_factor``: The decay factor for the refit
              operation. Defaults to ``0.99``.
            * ``elite_frac``: The fraction of the top individuals to
              select from the population as elites (not used for
              crossover). Defaults to ``0.0``.
            * ``tournament_size``: The number of individuals to select
              for tournament selection. Defaults to ``2``.
            * ``p_c``: The crossover probability for each non-elite
              parent pair (which is also teh fraction of pairs taken
              for crossover). Defaults to ``0.7``.
            * ``num_cross_points``: The number of points to use for
              crossover (n+1 segments will be created and every second
              one will be swapped to create 2 children). This is used if
              ``crossover_type`` is ``n_point``. Defaults to ``1``,
              meaning single-point crossover.
            * ``frac_cross_axes``: The fraction of rows and columns to
              use for crossover if ``crossover_type`` is
              ``row_col_swap``. Defaults to ``0.5``.
            * ``p_m``: The probability of an individual being mutated.
              Defaults to ``0.2``.
            * ``p_flip``: The probability of a bit being flipped if
              ``mutation_type`` is ``bit_flip``. Defaults to ``0.1``.
            * ``frac_mutate_axes``: The fraction of rows and columns to
              use for mutation if ``mutation_type`` is
              ``row_col_shuffle``. Defaults to ``0.2``.

        Args:
            kwargs (dict[str, typing.Any]): A dictionary of optional
                parameters for the GA as discussed above.
        """
        # Assign refit parameters
        self.stop_at_1 = kwargs.get("stop_at_1", True)
        self.decay_factor = kwargs.get("decay_factor", 0.99)
        self.local_bests = np.zeros((1, self.K**2))
        self.counters = np.zeros(1)  # Stagnation counters
        self.best_fitness = 0  # Best fitness so far

        # Assign escape parameters
        self.mean_counter = 0
        self.mean = np.zeros(self.D)

        # Assign selection parameters
        self.elite_frac = kwargs.get("elite_frac", 0.0)
        self.tournament_size = kwargs.get("tournament_size", 2)

        # Assign crossover parameters
        self.p_c = kwargs.get("p_c", 0.7)
        self.num_cross_points = kwargs.get("num_cross_points", self.K // 2)
        frac_cross_axes = kwargs.get("frac_cross_axes", 0.5)
        self.num_cross_axes = max(1, round(self.K * frac_cross_axes))

        # Assign mutation parameters
        self.p_m = kwargs.get("p_m", 0.2)
        self.p_flip = kwargs.get("p_flip", 0.1)
        frac_mutate_axes = kwargs.get("frac_mutate_axes", 0.2)
        self.num_mutate_axes = max(1, round(self.K * frac_mutate_axes))

    @utilmethod
    def split(
        self,
        population: np.ndarray,
        frac: float = 0.0,
        fitness: Collection[float] | None = None,
        ensure_even: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    ):
        """Splits a population into 2 groups.

        Splits a population into 2 groups based on a fraction that
        represents the proportion of the population in the first group.
        If ``fitness`` is *not* ``None``, then the population is split
        based on fitness values (best individuals go into the first
        group). Otherwise, the population is split randomly.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            frac (float, optional): The fraction of the individuals in
                the first group. Defaults to 0.0.
            fitness (typing.Collection[float] | None, optional): A
                collection of fitness values for each individual. If
                ``None``, the population is split randomly. Defaults to
                ``None``.
            ensure_even (bool, optional): Whether to ensure that the
                first group has an even number of individuals. This is
                useful for crossover. Defaults to ``False``.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]
            | tuple[tuple[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]:
            A tuple of numpy arrays representing the 2 groups of the
            population. If ``fitness`` is ``None``, then the tuple
            contains 2 numpy arrays representing the 2 groups of the
            population. Otherwise, the tuple contains 2 tuples of numpy
            arrays representing the 2 groups of the population and their
            corresponding fitness values.
        """
        # Number of individuals to select
        N, D = population.shape
        size = round(N * frac)

        if ensure_even and size % 2 != 0:
            # Ensure even
            size -= 1

        if size == 0 and fitness is None:
            # Return empty array and population
            return np.empty((0, D)), population
        elif size == 0:
            # Return empty arrays and the original un-split population
            return (np.empty((0, D)), np.empty(0)), (population, fitness)

        if fitness is None:
            # Select random individuals for the first group
            indices = self.rng.choice(N, size=size, replace=False)
        else:
            # Select the best individuals for the first group
            indices = np.argpartition(fitness, -size)[-size:]

        # Split the population into 2 groups
        population1 = np.array(population)[indices]
        population2 = np.delete(population, indices, axis=0)

        if fitness is None:
            # Return the 2 population groups
            return population1, population2

        # Split the fitness into 2 groups
        fitness1 = np.array(fitness)[indices]
        fitness2 = np.delete(fitness, indices, axis=0)

        return (population1, fitness1), (population2, fitness2)

    @utilmethod
    def mix(
        self,
        population: np.ndarray,
        callback: Callable[..., np.ndarray],
        param_name: str,
        param_values: list[str],
    ) -> np.ndarray:
        """Mixes a population by changing a parameter.

        This function splits a population into sub-populations and
        changes a parameter for each sub-population. The callback
        function is then called on each sub-population and the results
        are concatenated to form the new population.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            callback (typing.Callable[..., numpy.ndarray]): A callback
                function that takes a sub-population as input and
                returns a numpy array of shape ``(N, D)`` representing
                the new sub-population.
            param_name (str): The name of the parameter to change. I.e.,
                the name of the attribute of this class to change.
            param_values (list[str]): A list of values to assign to the
                attribute called ``param_name`` for each sub-population.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the new population.
        """
        # Mixes the population by changing the value of a parameter
        sub_populations = np.array_split(population, len(param_values))
        mixed_population = []

        for param_value, sub_population in zip(param_values, sub_populations):
            # Set the parameter value and call the callback function
            setattr(self, param_name, param_value)
            mixed_population.append(callback(sub_population))

        # Reset the parameter value
        setattr(self, param_name, "mixed")

        return np.vstack(mixed_population)

    @submethod
    def roulette_wheel_sampling(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Selects individuals from a population based on fitness.

        This function uses proportionate selection to select the best
        individuals (a *bag of solutions*) based on their fitness values
        for the next phase of GA: crossover and mutation.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            fitness (numpy.ndarray): A numpy array of shape (N,)
                representing fitness values for each individual. The
                values may be negative.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the selected individuals.
        """
        # Use fitness as distribution and sample based on it
        p = softmax(fitness) if len(set(fitness)) > 1 else None
        population = self.rng.choice(population, len(population), p=p)

        return population

    @submethod
    def tournament_sampling(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Selects individuals from a population based on fitness.

        This function uses tournament selection to select the best
        individuals (a *bag of solutions*) based on their fitness values
        for the next phase of GA: crossover and mutation. Unlike in
        roulette wheel sampling, this method does not use the fitness
        values as a distribution. Instead, it selects random groups of
        individuals and selects the best individuals from each group.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            fitness (numpy.ndarray): A numpy array of shape (N,)
                representing fitness values for each individual. The
                values may be negative.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the selected individuals.
        """
        # Population size
        N = len(population)

        # Select random groups and determine the best individuals
        idx = self.rng.integers(N, size=(N, self.tournament_size))
        population = population[idx][range(N), np.argmax(fitness[idx], 1)]

        return population

    @submethod
    def segment_swap(
        self,
        parents1: np.ndarray,
        parents2: np.ndarray,
    ) -> np.ndarray:
        """N-point crossover between 2 sets of parents.

        This is a general method for performing n-point crossover (n+1
        segment swap) between parents. In other words, each parent is
        split to ``n+1`` segments and every other segment is swapped.

        Note:
            The generated segment lengths are random, i.e., the choice
            of crossover points are random for every parent pair.

        Args:
            parents1 (numpy.ndarray): A numpy array of shape (N, D)
                representing the first set of parents.
            parents2 (numpy.ndarray): A numpy array of shape (N, D)
                representing the second set of parents.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the children generated from the parents.
        """

        if len(parents1) == 0 or len(parents2) == 0:
            return np.empty((0, self.D))

        def reflatten_half(population, is_random=False):
            # Re-flatten half of the population to colum-based fashion
            if is_random:
                row_based, col_based = self.split(population, 0.5)
            else:
                row_based = population[: len(population) // 2]
                col_based = population[len(population) // 2 :]

            # Reshape from (N, rows, cols) to (N, D)
            matrix = col_based.reshape(-1, self.K, self.K)
            col_based = matrix.transpose((0, 2, 1)).reshape(-1, self.D)

            return np.vstack([row_based, col_based])

        # Re-flatten half of the parents to column-based fashion
        parents1 = reflatten_half(parents1, is_random=True)
        parents2 = reflatten_half(parents2, is_random=True)
        children1, children2 = parents1.copy(), parents2.copy()

        # Generate n random cross points for each pair of parents
        points = np.sort(
            np.array(
                [
                    self.rng.choice(
                        range(1, parents1.shape[1]),
                        self.num_cross_points,
                        replace=False,
                    )
                    for _ in range(parents1.shape[0])
                ]
            ),
            axis=1,
        ).astype(np.int64)

        if points.shape[1] % 2 != 0:
            # Add last column of the final indices in case it's missing
            last_column = np.full((points.shape[0], 1), parents1.shape[1])
            points = np.hstack([points, last_column])

        # True values fall into the ranges specified by idx0 and idx1
        idx0, idx1 = points[:, 0::2, None], points[:, 1::2, None]
        mask = ((idx0 <= range(self.D)) & (range(self.D) <= idx1)).any(1)

        # Perform crossover using the mask
        children1[mask] = parents2[mask]
        children2[mask] = parents1[mask]

        # Re-flatten half of the children back to row-based fashion
        children1 = reflatten_half(children1, is_random=False)
        children2 = reflatten_half(children2, is_random=False)

        return np.vstack([children1, children2])

    @submethod
    def row_col_swap(
        self,
        parents1: np.ndarray,
        parents2: np.ndarray,
    ) -> np.ndarray:
        """Performs a crossover between 2 sets of parents.

        This function performs a crossover between 2 sets of parents by
        swapping rows and columns between them. The number of rows and
        columns to swap is determined by the ``frac_cross_axes``
        parameter.

        Note:
            Half of the total swapping fraction goes to rows and the
            other half goes to columns.

        Args:
            parents1 (numpy.ndarray): A numpy array of shape (N, D)
                representing the first set of parents.
            parents2 (numpy.ndarray): A numpy array of shape (N, D)
                representing the second set of parents.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the children generated from the parents.
        """
        # Split the parents into 2 groups: for rows and columns
        children1, children2 = parents1.copy(), parents2.copy()
        K = np.sqrt(self.D).astype(np.int64)
        N = children1.shape[0]

        # Reshape from (N, D) to (N, rows, cols)
        children1 = children1.reshape(-1, K, K)
        children2 = children2.reshape(-1, K, K)

        # Choose indices for rows/cols that will be swapped
        indices = self.rng.permuted(
            np.tile(range(K), (N, 1)),
            axis=1,
        )[:, : self.num_cross_axes]

        # Choose which tuples will be rows and which will be columns
        mask = self.rng.random(N) < 0.5
        rows = indices[mask]
        cols = indices[~mask]

        # Generate indices for rows and columns to swap
        R = np.arange(N)[mask, None]
        C = np.arange(N)[~mask, None]

        # Perform the actual swap of columns
        children1[C, :, cols], children2[C, :, cols] = (
            children2[C, :, cols],
            children1[C, :, cols],
        )

        # Perform the actual swap of rows
        children1[R, rows, :], children2[R, rows, :] = (
            children2[R, rows, :],
            children1[R, rows, :],
        )

        # Reshape from (N, rows, cols) to (N, D)
        children1 = children1.reshape(-1, self.D)
        children2 = children2.reshape(-1, self.D)

        return np.vstack([children1, children2])

    @submethod
    def bit_flip(
        self,
        individuals: np.ndarray,
    ) -> np.ndarray:
        """Performs a mutation by flipping bits in binary genomes.

        Loops through each genotype in a binary genome and flips it with
        probability ``p_flip``.

        Args:
            individuals (numpy.ndarray): The binary genomes of shape
                (N, D).

        Returns:
            numpy.ndarray: Binary genomes of shape (N, D) with mutated
            genotypes.
        """
        # Create a copy of individuals
        individuals = individuals.copy()

        # Check which genotypes to mutate and mutate them
        is_mutable = self.rng.random(individuals.shape) < self.p_flip
        individuals[is_mutable] = 1 - individuals[is_mutable]

        return individuals

    @submethod
    def axis_shuffle(
        self,
        individuals: np.ndarray,
    ) -> np.ndarray:
        """Performs a mutation by shuffling rows and columns.

        This function performs a mutation by shuffling the values within
        rows and columns of a binary genome. The number of rows and
        columns to shuffle is determined by the ``frac_mutate_axes``
        parameter.

        Args:
            individuals (numpy.ndarray): The binary genomes of shape
                (N, D).

        Returns:
            numpy.ndarray: Binary genomes of shape (N, D) with mutated
            genotypes.
        """
        # Split individuals to mutate, reshape to (rows, cols)
        individuals, rest = self.split(individuals, self.p_m)
        individuals = individuals.reshape(-1, self.K, self.K)

        # Choose indices for rows/cols that will be shuffled
        matrix_idx = np.tile(np.arange(self.K), (len(individuals), 1))
        random_idx = self.rng.permuted(matrix_idx, axis=1)
        chosen_idx = random_idx[:, : self.num_mutate_axes]

        # Choose which rows and columns to shuffle
        mask = self.rng.random(len(chosen_idx)) < 0.5
        rows, R = chosen_idx[mask], np.arange(len(individuals))[mask, None]
        cols, C = chosen_idx[~mask], np.arange(len(individuals))[~mask, None]

        # Regenerate new values for teh chosen rows and columns
        individuals[R, rows, :] = self.rng.integers(
            2, size=(len(R), self.num_mutate_axes, self.K)
        )
        individuals[C, :, cols] = self.rng.integers(
            2, size=(len(C), self.num_mutate_axes, self.K)
        )

        return np.vstack([rest, individuals.reshape(-1, self.D)])

    @override
    def initialize_population(self) -> np.ndarray:
        """Generates an initial population of individuals.

        Generates a population of N individuals, each of D dimensions.
        Each individual is strictly a binary genome, i.e., each genotype
        can only be 0 or 1: :math:`\mathbf{x} \in \{0, 1\}^D`.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the initial population of individuals.
        """
        return self.rng.integers(2, size=(self.N, self.D))

    @override
    def evolve(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs a genetic algorithm optimization.

        This function performs a genetic algorithm optimization for a
        given fitness function. It uses roulette wheel sampling for
        selecting individuals for crossover and mutation.

        Args:
            population (numpy.ndarray): The population of individuals
                to be evolved. It should be a numpy array of shape
                ``(N, D)`` where ``N`` is the number of individuals and
                ``D`` is the number of dimensions for each individual.
            fitnesses (numpy.ndarray): The fitness scores of the
                individuals in the population. The fitness scores must
                be in the same order as the individuals in the
                population.
            *cache: Not used here.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple[typing.Any]]: A
            tuple containing the new population and its corresponding
            fitness scores (final value of a tuple is cache but it is
            not used).
        """
        # Select parents, perform crossover and mutation
        population, fitness = self.refit(population, fitness)
        population, elites = self.selection(population, fitness)

        population = self.crossover(population)
        population = self.mutation(population)

        # Add the elites back to the population and escape local optima
        population = np.vstack([elites, population])
        population = self.escape(population)

        # Compute fitness for each individual
        fitness = self.evaluate(population)

        return population, fitness, cache

    def refit(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refits the fitness of the local best solutions.

        This function refits the fitness of the local best solutions
        (i.e., the individuals that are the best in their local
        neighborhood) to avoid stagnation. This is done by multiplying
        the fitness of the local best solutions by a decay factor
        (``decay_factor``) to reduce their fitness over time. This
        function also adds new local best solutions to the memory.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            fitness (numpy.ndarray): A numpy array of shape (N,)
                representing fitness values for each individual. All the
                values must be positive.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple of numpy arrays
            of shapes ``(N, D)`` and ``(N,)`` where the first array
            represent the population and the second array represents the
            fitness values of the population.
        """
        if not self.use_refit or (self.stop_at_1 and (fitness >= 1).any()):
            # No refit is needed
            return population, fitness

        # Determine which genomes (if any) hit the local best solutions
        is_local = (population[:, None, :] == self.local_bests).all(axis=2)
        local_bests_idx = np.where(is_local & (fitness < 1)[:, None])[1]

        # Increment the stagnation counters
        self.counters[local_bests_idx] += 1
        mult = self.decay_factor ** self.counters[local_bests_idx]

        # Don't use if not all local_bests are unique
        # counters_index = np.argmax(is_local[refit_mask], axis=1)

        # Refit the fitness of the local best solutions
        refit_mask = is_local.any(axis=1) & (fitness < 1)
        fitness[refit_mask] = np.round(fitness[refit_mask] * mult, 8)

        # Determine if any new local best solutions were found
        better_mask = fitness[~refit_mask] > self.best_fitness
        local_bests = np.unique(population[~refit_mask][better_mask], axis=0)

        if not better_mask.any():
            return population, fitness

        # Add local best solutions to the memory (pruning possible btw)
        self.counters = np.append(self.counters, np.zeros(len(local_bests)), 0)
        self.local_bests = np.append(self.local_bests, local_bests, 0)
        self.best_fitness = fitness[~refit_mask][better_mask].max()

        return population, fitness

    def selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Selects individuals for crossover and mutation.

        Performs a selection based on ``selection_type``. It can select
        individuals for crossover and mutation based on their fitness
        values (roulette wheel sampling) or randomly (tournament
        selection). It can also select a fraction of elite members
        (individuals with best fitness values) that should not be used
        for crossover/mutation and would be passed immediately to the
        next population instead.

        Note:
            Some of the selected non-elite individuals can be repeated.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.
            fitness (numpy.ndarray): A numpy array of shape (N,)
                representing fitness values for each individual. All the
                values must be positive.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple of numpy arrays
            of shapes ``(N-num_elites, D)`` and ``(num_elites, D)``where
            the first array represent the non-elite members and the
            second array represents the elite members of the population.
        """
        # Split the population into 2 groups: elites and non-elites
        splits = self.split(population, self.elite_frac, fitness)
        (elites, _), (non_elites, fitness) = splits

        match self.selection_type:
            case "roulette_wheel":
                # Select non-elites using roulette wheel sampling
                non_elites = self.roulette_wheel_sampling(non_elites, fitness)
            case "tournament":
                # Select non-elites using tournament selection
                non_elites = self.tournament_sampling(non_elites, fitness)

        return non_elites, elites

    def crossover(self, population: np.ndarray) -> np.ndarray:
        """Performs a crossover between individuals.

        Performs a crossover between individuals based on
        ``crossover_type``. It can perform n-point crossover (n+1
        segment swap) or row/column swap.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.

        Raises:
            ValueError: If ``crossover_type`` is not one of the
                following: "n_point", "row_col_swap", or "mixed".

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the new population.
        """
        if self.crossover_type == "mixed":
            # Perform mixed crossover
            return self.mix(
                population,
                self.crossover,
                "crossover_type",
                self.VALID_CROSSOVER_TYPES,
            )

        # Split the population into 2 groups: for mating and remaining
        split = self.split(population, self.p_c, ensure_even=True)
        parents_mating, parents_remain = split

        # Split the parents into 2 groups, initialize children as copies
        [parents1, parents2] = parents_mating.reshape(2, -1, self.D)

        match self.crossover_type:
            case "n_point":
                # Perform n-point crossover (segment swap)
                children = self.segment_swap(parents1, parents2)
            case "row_col_swap":
                # Perform full row/column swap
                children = self.row_col_swap(parents1, parents2)
            case _:
                raise ValueError(f"Invalid crossover type: {self.crossover_type}")

        return np.vstack([parents_remain, children])

    def mutation(self, population: np.ndarray) -> np.ndarray:
        """Performs a mutation on individuals.

        Performs a mutation on individuals based on ``mutation_type``.
        It can perform bit-flip or row/column shuffle.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.


        Raises:
            ValueError: If ``mutation_type`` is not one of the
                following: "bit_flip", "row_col_shuffle", or "mixed".

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the new population.
        """
        if self.mutation_type == "mixed":
            # Perform mixed mutation
            return self.mix(
                population,
                self.mutation,
                "mutation_type",
                self.VALID_MUTATION_TYPES,
            )

        # Split the population into 2 groups: mutants and regular
        mutants, regular = self.split(population, self.p_m)

        match self.mutation_type:
            case "bit_flip":
                mutants = self.bit_flip(mutants)
            case "row_col_shuffle":
                mutants = self.axis_shuffle(mutants)
            case _:
                raise ValueError(f"Invalid mutation type: {self.mutation_type}")

        return np.vstack([regular, mutants])

    def escape(self, population: np.ndarray) -> np.ndarray:
        """Escapes local optima by replacing stagnated individuals.

        This function escapes local optima by replacing stagnated
        individuals with random individuals. An individual is considered
        stagnated if it matches a local best solution for at least 80%
        of its row or column values. The individual is also considered
        stagnated if it matches the mean of the population for at least
        80% of its rows or column values. The mean of the population is
        computed by rounding the mean of each row/column to the nearest
        integer.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as
                D-dimensional vectors of type :attr:`numpy.float32`.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the new population.
        """
        if not self.use_escape or (self.stop_at_1 and self.best_fitness >= 1):
            # No escape is needed
            return population

        # Extract the top local best solutions and compute mean
        top_local_bests = self.local_bests[self.counters >= 30]
        mean = np.round(population.mean(axis=0, keepdims=True))

        if (self.mean == mean).all():
            # Same mean as before
            self.mean_counter += 1
        else:
            # The mean has changed
            self.mean_counter = 0
            self.mean = mean

        if self.mean_counter >= 50:
            # The mean has not changed so assume it is a local optima
            top_local_bests = np.append(top_local_bests, mean, axis=0)

        if len(top_local_bests) == 0:
            # No local best solutions
            return population

        # Check how well each individual matches each local best
        eq_masks = population.reshape(-1, self.K, self.K)[
            :, None, :, :
        ] == top_local_bests.reshape(-1, self.K, self.K)

        # Assume that an individual matches a local best if at least
        # 80% of its row or column values match those of any local best
        col_eq = (eq_masks.sum(axis=3) / self.K) > 0.8
        row_eq = (eq_masks.sum(axis=2) / self.K) > 0.8

        # Check if at least 30% of the rows or columns match
        eq_sums = (col_eq + row_eq).sum(axis=-1).max(axis=-1)
        eq_frac = eq_sums / (self.K * 2)
        mask = (eq_frac > 0.3) & (eq_sums.mean() > (0.3 * self.K * 2))

        # Replace stagnated individuals with random individuals
        population[mask] = self.rng.integers(2, size=(mask.sum(), self.D))
        mean = np.round(population.mean(axis=0, keepdims=True))

        return population
