import numpy as np
from tqdm import tqdm
from typing import Callable


class BinaryGeneticAlgorithm:
    """
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
        seed (int, optional): The seed for the random number generator.
            Defaults to ``None``.
    """
    def __init__(
        self,
        N: int = 100,
        D: int = 9,
        p_m: float = 0.001,
        p_c: float = 0.7,
        elite_frac: float = 0.0,
        num_cross_points: int = 1,
        seed: int = None,
    ):  
        # Initialize variables
        self.N = N
        self.D = D
        self.p_c = p_c
        self.p_m = p_m
        self.num_elites = round(elite_frac * N)
        self.num_cross_points = num_cross_points
        self.seed = seed

    def generate_individuals(self) -> np.ndarray:
        """Generates an initial population of individuals.

        Generates a population of N individuals, each of D dimensions.
        Each individual is strictly a binary genome, i.e., each genotype
        can only be 0 or 1: :math:`\mathbf{x} \in \{0, 1\}^D`.

        Returns:
            numpy.ndarray: A numpy array of shape ``(N, D)``
            representing the initial population of individuals.
        """
        np.random.seed(self.seed)
        return np.random.randint(2, size=(self.N, self.D))

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

        # Calculate the probability of selecting each non-elite member
        p = fitness[~elites_mask] / fitness[~elites_mask].sum()

        # Get the non-elite members based on proportionate sampling
        parents = np.random.default_rng().choice(
            population[~elites_mask],
            size=len(population) - len(elites),
            p=p,
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
        indices = np.random.permutation(parents.shape[0])
        parents_mating = parents[indices][:num_rows]
        parents_remain = parents[indices][num_rows:]

        # Split the parents into 2 groups, initialize children as copies
        [parents1, parents2] = parents_mating.reshape(2, -1, parents.shape[1])
        children1, children2 = parents1.copy(), parents2.copy()

        # Generate n random cross points for each pair of parents
        points = np.sort(np.array([
            np.random.choice(range(1, parents1.shape[1]), n, replace=False)
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
        p_m: float = 0.001,
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

        # Check which genotypes to mutate and mutate
        is_mutable = np.random.rand(*individuals.shape) < p_m
        individuals[is_mutable] = 1 - individuals[is_mutable]

        return individuals
    
    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        max_generations: int = 100,
        patience: int = 50,
    ) -> np.ndarray:
        """Performs a genetic algorithm optimization.

        This function performs a genetic algorithm optimization for a given
        fitness function. It uses roulette wheel sampling for selecting
        individuals for crossover and mutation. It also supports early
        stopping based on the number of generations without improvement.

        Args:
            fitness_fn (Callable[[numpy.ndarray], float]): A fitness 
                function that should return a value to be optimized
                (maximized) given a solution represented as a numpy
                array of shape (D,).
            max_generations (int, optional): The maximum number of
                generations. Defaults to ``100``.
            patience (int, optional): The number of generations without
                improvement before early stopping. Set to a value below
                ``0`` to run without early stopping. Defaults to ``50``.

        Returns:
            numpy.ndarray: A numpy array of shape ``(D,)`` representing
            the best solution found.
        """
        # Initialize a bag of random solutions
        population = self.generate_individuals()

        # Init early stopping
        last_best = -np.inf
        _patience = 0

        # Initialize progress bar
        pbar = tqdm(range(max_generations), desc="Current best: N/A")

        for _ in pbar:
            # Compute fitness for each individual, select the best
            fitness = np.apply_along_axis(fitness_fn, 1, population)
            parents = self.roulette_wheel_sampling(population, fitness)

            # Perform crossover and mutation to get the next generation
            children = self.n_point_crossover(parents)
            population = self.bit_flip_mutation(children)
            
            # Retrieve the best value from population
            best_solution = fitness.argmax()
            best_fitness = fitness.max()

            # Update progress bar with current best
            pbar.set_description(f"Current best {best_fitness:.8f}")

            # Check for termination
            if patience > 0 and _patience >= patience:
                break
            elif best_fitness == last_best:
                _patience += 1
            else:
                _patience = 0
            
            # Update last best
            last_best = best_fitness
        
        return best_solution


if __name__ == "__main__":
    fitness_fn = lambda x: np.sum(x)
    ga = BinaryGeneticAlgorithm(N=100, D=20, p_c=0.7, p_m=0.1)
    solution = ga.optimize(fitness_fn, max_generations=1000, patience=60)
    print(solution)