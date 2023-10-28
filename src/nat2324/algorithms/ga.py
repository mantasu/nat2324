from typing import Callable, Collection

import numpy as np
from scipy.special import softmax

from .base_runner import BaseRunner


class BinaryGeneticAlgorithm(BaseRunner):
    """Optimizer based on *binary* genetic algorithm.

    A binary genetic algorithm (GA) implementation for solving
    optimization problems. This implementation uses roulette wheel
    sampling for selecting individuals for crossover and mutation.

    Note:
        A better implementation would be to generalize this to a general
        Genetic Algorithm that can deal variable discrete (an even
        continuos) cases, with different crossover and mutation types.
        However, for this assignment, it is sufficient to use a binary
        GA.

    Args:
        N (int, optional): The number of individuals. Defaults to
            ``100``.
        D (int, optional): The number of dimensions for each individual.
            Defaults to ``9``.
        p_c (float, optional): The crossover probability for each
            non-elite parent pair (which is also teh fraction of pairs
            taken for crossover). Defaults to ``0.7``.
        p_m (float, optional): The probability of a bit being flipped.
            Defaults to ``0.001``.
        elite_frac (float, optional): The fraction of the top
            individuals to select from the population as elites (not
            used for crossover). Defaults to ``0.0``.
        num_cross_points (int, optional): The number of points to use
            for crossover (n+1 segments will be created and every second
            one will be swapped to create 2 children). Defaults to
            ``1``, meaning single-point crossover.
        parallelize_fitness (bool, optional): Whether to parallelize the
            computation of the fitness scores for each individual.
            Defaults to ``False``, which should be set if the fitness
            function is not that complex, in which case running on a
            single process would be more efficient.
        seed (int | None, optional): The seed for the random number
            generator. Defaults to ``None``, which means on every run,
            the results will be random.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        N: int = 100,
        D: int = 9,
        p_c: float = 0.7,
        p_m: float = 0.001,
        elite_frac: float = 0.0,
        num_cross_points: int = 1,
        mutation_type: str = "bit_flip",
        parallelize_fitness: bool = False,
        seed: int | None = None,
    ):
        super().__init__(fitness_fn, N, parallelize_fitness, seed=seed)

        # Initialize variables
        self.D = D
        self.p_c = p_c
        self.p_m = p_m
        self.num_cross_points = num_cross_points
        self.seed = seed

        self.K = np.sqrt(self.D).astype(np.int64)  # If x represents matrix
        self.row_col_frac = 0.2
        self.num_row_cols = max(1, round(self.K * self.row_col_frac))

        self.mutation_type = mutation_type
        self.mean_counter = 0
        self.mean = np.zeros(self.D)

        # Calculate/initialize other variables
        self.num_elites = round(elite_frac * N)

        self.decay_factor = 0.9995
        self.local_bests = np.zeros((1, self.K**2))
        self.stagnation_counters = np.zeros(1)
        self.local_best_fitness = 0

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

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the
            new population and its corresponding fitness scores.
        """
        # Select parents, perform crossover and mutation
        population, fitness = self.refit(population, fitness)
        parents = self.roulette_wheel_sampling(population, fitness)
        children = self.n_point_crossover(parents)
        # population = self.bit_flip(children)
        population = self.mutation(children)
        population = self.escape(population)

        # population, elites = self.sample(population, fitness)
        # population = self.mutate(self.crossover(population))
        # population = np.vstack([elites, population])

        # Compute fitness for each individual (parallelize if large N)
        fitness = self.evaluate(population)
        # fitness = (
        #     np.array(self.parallel_apply(self.fitness_fn, population))
        #     if self.parallelize_fitness
        #     else np.apply_along_axis(self.fitness_fn, 1, population)
        # )

        return population, fitness, cache

    def split(
        self,
        population: np.ndarray,
        frac: float = 0.0,
        fitness: Collection[float] | None = None,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    ):
        # Splits the population randomly or based on fitness
        # frac: fraction of population to select in the first group

        # Number of individuals to select
        N = population.shape[0]
        size = round(N * frac)

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

    def refit(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fitness = np.array(fitness)

        # if (fitness >= 1).any():
        #     return population, fitness

        non_global = fitness < 1
        is_local = (population[:, None, :] == self.local_bests).all(axis=2)
        index = np.where(is_local & non_global[:, None])[
            1
        ]  # no need for list(set(...))
        self.stagnation_counters[index] += 1

        refit_mask = is_local.any(axis=1) & non_global

        # print("OI")
        # print(non_global.sum())
        # print(self.stagnation_counters.shape, self.stagnation_counters)
        # print(is_local)
        # print(refit_mask)
        # print(np.where(is_local))
        # print(list(set(np.where(is_local)[0])), list(set(np.where(is_local)[1])))

        # stagnation_counters_index = np.argmax(is_local[refit_mask], axis=1) // don't use if not all local_bests are unique

        fitness[refit_mask] = np.round(
            fitness[refit_mask]
            * (self.decay_factor ** self.stagnation_counters[index]),
            8,
        )
        better_mask = fitness[~refit_mask] > self.local_best_fitness
        new_local_bests = np.unique(population[~refit_mask][better_mask], axis=0)

        # print("OOF")
        # print(population[~refit_mask][better_mask])
        # print(new_local_bests)

        self.local_bests = np.append(self.local_bests, new_local_bests, axis=0)
        self.stagnation_counters = np.append(
            self.stagnation_counters, np.zeros(len(new_local_bests)), axis=0
        )
        self.local_best_fitness = (
            fitness[~refit_mask][better_mask].max()
            if better_mask.any()
            else self.local_best_fitness
        )

        # Punish local neighbors
        top_local_bests = self.local_bests[self.stagnation_counters >= 2]

        if True or len(top_local_bests) == 0:
            return population, fitness

        eq_masks = population.reshape(-1, self.K, self.K)[
            :, None, :, :
        ] == top_local_bests.reshape(-1, self.K, self.K)
        col_eq = (eq_masks.sum(axis=3) / self.K) > 0.8
        row_eq = (eq_masks.sum(axis=2) / self.K) > 0.8
        same_row_col_sums = (col_eq + row_eq).sum(axis=-1)
        stag_idx = same_row_col_sums.argmax(axis=-1)

        is_damped = same_row_col_sums[range(len(same_row_col_sums)), stag_idx] > 0
        # print(is_damped.shape, same_row_col_sums.shape, stag_idx.shape, same_row_col_sums.max(axis=-1).shape, len(top_local_bests))
        stag_idx = stag_idx[is_damped]

        # print(is_damped.sum(), fitness[is_damped].shape)

        num_same = same_row_col_sums[is_damped][
            range(len(same_row_col_sums[is_damped])), stag_idx
        ]
        # msk = num_same >= self.K # num_same.mean() * 0.5

        damp_frac = (
            same_row_col_sums[is_damped][
                range(len(same_row_col_sums[is_damped])), stag_idx
            ]
            + 0 * self.K
        ) / (self.K * 2)
        # damp_frac *= 0.5
        damped = (
            damp_frac
            * fitness[is_damped]
            * (self.decay_factor ** self.stagnation_counters[stag_idx])
        )
        # print(damped.shape)
        fitness[is_damped] = np.round(fitness[is_damped] * (1 - damp_frac) + damped, 8)

        return population, fitness

    def escape(self, population: np.ndarray) -> np.ndarray:
        if self.local_best_fitness >= 1:
            return population

        top_local_bests = self.local_bests[self.stagnation_counters >= 30]
        mean = np.round(population.mean(axis=0, keepdims=True))

        if (self.mean == mean).all():
            self.mean_counter += 1
        else:
            self.mean_counter = 0
            self.mean = mean

        if self.mean_counter >= 50:
            top_local_bests = np.append(top_local_bests, mean, axis=0)  # [-1:, :]
            # print(self.mean_counter, top_local_bests.shape)

        # if self.mean_counter == 30:
        #     print("added")

        if len(top_local_bests) == 0:
            return population

        eq_masks = population.reshape(-1, self.K, self.K)[
            :, None, :, :
        ] == top_local_bests.reshape(-1, self.K, self.K)
        # col_eq = eq_masks.all(axis=3)
        # row_eq = eq_masks.all(axis=2)
        col_eq = (eq_masks.sum(axis=3) / self.K) > 0.8
        row_eq = (eq_masks.sum(axis=2) / self.K) > 0.8

        eq_sums = (col_eq + row_eq).sum(axis=-1).max(axis=-1)
        eq_frac = eq_sums / (self.K * 2)

        mask = (eq_frac > 0.3) & (eq_sums.mean() > (0.3 * self.K * 2))
        # print(eq_sums.mean(), mask.sum())

        # , '\n', eq_sums, '\n', mask)

        # if mask.sum() / self.N > 0.3:
        #     population = self.initialize_population()

        population[mask] = self.rng.integers(2, size=(mask.sum(), self.D))

        # p_m = self.p_m
        # n = self.num_row_cols
        # self.p_m = 0.7
        # self.num_row_cols = int(self.K * 0.8)
        # population[mask] = self.bit_flip(population[mask])
        # self.p_m = p_m
        # self.num_row_cols = n

        mean = np.round(population.mean(axis=0, keepdims=True))
        # print("New mean", (mean == self.mean).all())

        return population

    def roulette_wheel_sampling(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Selects individuals from a population based on fitness.

        This function uses proportionate selection to select the best
        individuals (a *bag of solutions*) based on their fitness values
        for the next phase of GA: crossover and mutation. It can also
        select a fraction of elite members (individuals with best
        fitness values) that should not be used for crossover and be
        passed to the next population instead.

        Note:
            Some of the selected non-elite individuals can be repeated.

        Args:
            population (numpy.ndarray): A numpy array of shape (N, D)
                representing a population of N individuals as D-dimensional
                vectors of type :attr:`numpy.float32`.
            fitness (numpy.ndarray): A numpy array of shape (N,)
                representing fitness values for each individual. All the
                values must be positive.

        Returns:
            numpy.ndarray: A numpy arrays of shapes ``(N, D)`` where the
            first ``num_elites`` rows  represent the elite members and
            the remaining rows represent sampled individuals from
            non-elite members of the population.
        """
        # Get mask for selecting only the elite members, apply to population
        elites_mask = np.argsort(fitness) > (self.N - self.num_elites)
        fitness = np.array(fitness)[~elites_mask]
        elites = population[elites_mask]

        if True:  # self.rng.random() < 0.5:
            parents = self.tournament_selection(population[~elites_mask], fitness)
        else:
            # T = 2.0

            if len(set(fitness)) > 1:
                # p = softmax(fitness)
                p = fitness / fitness.sum()
            else:
                p = None

            # Get the non-elite members based on proportionate sampling
            parents = self.rng.choice(
                population[~elites_mask],
                size=len(population) - len(elites),
                p=p,
            )

        return np.vstack([elites, parents])

    def tournament_selection(self, population, fitness, tournament_size=2):
        # Get the size of the population
        pop_size = population.shape[0]

        # Initialize an array to hold the selected individuals
        selected = np.empty_like(population)

        for i in range(pop_size):
            # Randomly select tournament_size individuals from the population
            tournament_indices = self.rng.integers(0, pop_size, tournament_size)
            tournament_individuals = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            # Select the best individual from the tournament
            winner_index = np.argmax(tournament_fitness)
            selected[i] = tournament_individuals[winner_index]

        return selected

    def n_point_crossover(
        self,
        parents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """N-point crossover between 2 sets of parents.

        This is a general method for performing n-point crossover
        between N parents. In other words, each parent is split to
        ``n+1`` segments and every other segment is swapped.

        Note:
            The generated segment lengths are random, i.e., the choice
            of crossover points are random for every parent pair.

        Args:
            parents (numpy.ndarray): An ordered list of parents from
                best to worst. The first ``num_elites`` parents are
                considered elite and are not used for crossover.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the children. The first ``num_elites`` rows
            are the elite parents, the next ``(N-num_elites)*(1-p_c)``
            rows are the non-elite parents not chosen for crossover and
            the last ``(N-num_elites)*(1-p_c)`` rows are the children
            generated by crossover.
        """
        # Get the elite parents, remove them from the list
        parents_elite = parents[: self.num_elites]
        parents = parents[self.num_elites :]

        # Calculate the number of mating parents to sample
        num_rows = round(parents.shape[0] * self.p_c)
        n = self.num_cross_points

        if num_rows % 2 != 0:
            # Ensure even
            num_rows -= 1

        # Generate a permutation of row indices, select mating parents
        indices = self.rng.permutation(parents.shape[0])
        parents_mating = parents[indices][:num_rows]
        parents_remain = parents[indices][num_rows:]

        # Split the parents into 2 groups, initialize children as copies
        [parents1, parents2] = parents_mating.reshape(2, -1, parents.shape[1])
        children1, children2 = parents1.copy(), parents2.copy()

        if False:  # self.rng.random() < 0.5:
            K = np.sqrt(self.D).astype(np.int64)
            N = children1.shape[0]
            swap_size = K // 2

            children1 = children1.reshape(-1, K, K)
            children2 = children2.reshape(-1, K, K)

            indices = self.rng.permuted(np.tile(range(K), (N, 1)), axis=1)[
                :, :swap_size
            ]
            mask = self.rng.random(N) < 0.5
            rows = indices[mask]
            cols = indices[~mask]

            R = np.arange(N)[mask, None]
            C = np.arange(N)[~mask, None]

            children1[C, :, cols], children2[C, :, cols] = (
                children2[C, :, cols],
                children1[C, :, cols],
            )
            children1[R, rows, :], children2[R, rows, :] = (
                children2[R, rows, :],
                children1[R, rows, :],
            )

            children1 = children1.reshape(-1, self.D)
            children2 = children2.reshape(-1, self.D)
        else:
            # Generate n random cross points for each pair of parents
            points = np.sort(
                np.array(
                    [
                        self.rng.choice(range(1, parents1.shape[1]), n, replace=False)
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
            mask = ((idx0 <= range(self.D)) & (range(self.D) <= idx1)).any(axis=1)

            # Perform crossover using the mask
            children1[mask] = parents2[mask]
            children2[mask] = parents1[mask]

        return np.vstack([parents_elite, parents_remain, children1, children2])

    def bit_flip(
        self,
        individuals: np.ndarray,
    ) -> np.ndarray:
        """Performs a mutation by flipping bits in binary genomes.

        Loops through each genotype in a binary genome and flips it with
        probability ``p_m``.

        Note:
            Mutation is also applied to elite members as well!

        Args:
            individual (numpy.ndarray): The binary genomes of shape
                (N, D).

        Returns:
            numpy.ndarray: Binary genomes of shape (N, D) with mutated
            genotypes.
        """
        # Create a copy of individuals
        individuals = individuals.copy()

        # K = np.sqrt(self.D).astype(np.int64)
        # individuals = individuals.reshape(-1, K, K)

        # Check which genotypes to mutate and mutate them
        is_mutable = self.rng.random(individuals.shape) < self.p_m
        individuals[is_mutable] = 1 - individuals[is_mutable]

        return individuals

    def row_col_gen(
        self,
        individuals: np.ndarray,
    ) -> np.ndarray:
        # Split individuals to mutate, reshape to (rows, cols)
        individuals, rest = self.split(individuals, self.p_m)
        individuals = individuals.reshape(-1, self.K, self.K)

        # print(individuals[:2])

        # Choose indices for rows/cols that will be shuffled
        matrix_idx = np.tile(np.arange(self.K), (len(individuals), 1))
        random_idx = self.rng.permuted(matrix_idx, axis=1)
        chosen_idx = random_idx[:, : self.num_row_cols]

        # Choose which rows and columns to shuffle
        mask = self.rng.random(len(chosen_idx)) < 0.5
        rows, R = chosen_idx[mask], np.arange(len(individuals))[mask, None]
        cols, C = chosen_idx[~mask], np.arange(len(individuals))[~mask, None]

        # print(mask[:2].tolist(), '\n', rows[:2].tolist(), '\n', cols[:2].tolist())

        # Regenerate new values for teh chosen rows and columns
        individuals[R, rows, :] = self.rng.integers(
            2, size=(len(R), self.num_row_cols, self.K)
        )
        individuals[C, :, cols] = self.rng.integers(
            2, size=(len(C), self.num_row_cols, self.K)
        )

        # print(individuals[:2])

        return np.vstack([rest, individuals.reshape(-1, self.D)])

    def mutation(self, population: np.ndarray) -> np.ndarray:
        match self.mutation_type:
            case "bit_flip":
                return self.bit_flip(population)
            case "row_col_gen":
                return self.row_col_gen(population)
            case "mixed":
                # Split the population into 2 groups
                population1, population2 = self.split(population, 0.5)

                # Perform mutation on each group
                population1 = self.bit_flip(population1)
                population2 = self.row_col_gen(population2)

                return np.vstack([population1, population2])
            case _:
                raise ValueError(f"Invalid mutation type: {self.mutation_type}")
