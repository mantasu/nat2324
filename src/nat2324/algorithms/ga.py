import numpy as np
from typing import Callable
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
        parallelize_fitness: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        # Initialize variables
        self.fitness_fn = fitness_fn
        self.N = N
        self.D = D
        self.p_c = p_c
        self.p_m = p_m
        self.num_cross_points = num_cross_points
        self.parallelize_fitness = parallelize_fitness
        self.seed = seed

        # Calculate/initialize other variables
        self.num_elites = round(elite_frac * N)
        self.rng = np.random.default_rng(seed=seed)

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
        parents = self.roulette_wheel_sampling(population, fitness)
        children = self.n_point_crossover(parents)
        population = self.bit_flip_mutation(children)

        # Compute fitness for each individual (parallelize if large N)
        fitness = np.array(self.parallel_apply(self.fitness_fn, population)) \
                  if self.parallelize_fitness else \
                  np.apply_along_axis(self.fitness_fn, 1, population)

        return population, fitness

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
        elites = population[elites_mask]

        # Get the non-elite members based on proportionate sampling
        parents = self.rng.choice(
            population[~elites_mask],
            size=len(population) - len(elites),
            p=fitness / fitness.sum() if len(set(fitness)) > 1 else None,
        )

        return np.vstack([elites, parents])
    
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
        parents_elite = parents[:self.num_elites]
        parents = parents[self.num_elites:]

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

        # Generate n random cross points for each pair of parents
        points = np.sort(np.array([
            self.rng.choice(range(1, parents1.shape[1]), n, replace=False)
            for _ in range(parents1.shape[0])
        ]), axis=1).astype(np.int64)

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
    
    def bit_flip_mutation(
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

        # Check which genotypes to mutate and mutate them
        is_mutable = self.rng.random(individuals.shape) < self.p_m
        individuals[is_mutable] = 1 - individuals[is_mutable]

        return individuals
