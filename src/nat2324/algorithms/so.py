from math import gamma
from typing import Any, Callable, Collection

import numpy as np

from ..utils.decorators import override, private, utilmethod
from .base_runner import BaseRunner


class SwarmOptimization(BaseRunner):
    """Swarm Optimization algorithm.

    This class implements various Swarm Optimization (SO) algorithms. In
    particular, it implements the following algorithms:

        * Particle Swarm Optimization (PSO)
        * Differential Evolution (DE)
        * Cuckoo Search (CS)
        * Bat Algorithm (BA)

    Args:
        fitness_fn (typing.Callable[[numpy.ndarray], float]): The
            fitness function to optimize.
        bounds (tuple, optional): The bounds of the search space.
            Defaults to (-100, 100).
        N (int, optional): The number of particles. Defaults to 25.
        D (int, optional): The dimensionality of the search space.
            Defaults to 10.
        algorithm_type (str, optional): The type of SO algorithm to use.
            It must be one of the following: "particle_swarm", "ps",
            "differential_evolution", "de", "cuckoo_search", "cs", or
            "bat_algorithm", "ba". Defaults to "ps".
        parallelize_fitness (bool, optional): Whether to parallelize the
            fitness function evaluation. Defaults to ``False``.
        seed (int, optional): The seed to use for the random number
            generator. Defaults to ``None``.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: tuple = (-100, 100),
        N: int = 25,
        D: int = 10,
        algorithm_type: str = "ps",
        parallelize_fitness: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(fitness_fn, N, parallelize_fitness, seed)

        # Initialize parameters
        self.bounds = bounds
        self.D = D
        self.algorithm_type = algorithm_type

        # Initialize algorithm-specific parameters
        self._init_params(**kwargs)

    @private
    def _init_params(self, **kwargs):
        """Initializes algorithm-specific parameters.

        If teh algorithm type is *Particle Swarm Optimization (PSO)*,
        the following parameters are used:

            * w (float): The inertia weight. Defaults to 0.5.
            * alpha1 (float): The local force weight. Defaults to 2.5.
            * alpha2 (float): The global force weight. Defaults to 1.5.
            * alpha3 (float, optional): The repulsion force weight.
              Defaults to ``None``.
            * max_vel_frac (float, optional): The maximum velocity
                fraction in terms of space range. Defaults to 0.1.

        If the algorithm type is *Differential Evolution (DE)*, the
        following parameters are used:

            * F (float): The differential weight. Defaults to 0.1.
            * p (float): The crossover probability. Defaults to 0.1.

        If the algorithm type is *Cuckoo Search (CS)*, the following
        parameters are used:

            * beta (float): The Lévy exponent. Defaults to 1.0.
            * alpha (float): The step size. Defaults to 0.01.
            * pa (float): The abandonment probability. Defaults to 0.1.

        If the algorithm type is *Bat Algorithm (BA)*, the following
        parameters are used:

            * A (float): The loudness. Defaults to 2.0.
            * r (float): The pulse rate. Defaults to 0.5.
            * f_min (float): The minimum frequency. Defaults to 0.0.
            * f_max (float): The maximum frequency. Defaults to 0.25.
            * epsilon (float): The local random walk. Defaults to 0.01.

        Args:
            **kwargs: The keyword arguments to use for initializing the
                algorithm-specific parameters.

        Raises:
            ValueError: If the algorithm type is not supported.
        """
        match self.algorithm_type:
            case "particle_swarm" | "ps":
                # Particle Swarm Optimization (PSO)
                self.w = kwargs.get("w", 0.5)
                self.alpha1 = kwargs.get("alpha1", 2.5)
                self.alpha2 = kwargs.get("alpha2", 1.5)
                self.alpha3 = kwargs.get("alpha3", None)
                self.max_vel = kwargs.get("max_vel_frac", 0.1) * (
                    self.bounds[1] - self.bounds[0]
                )
            case "differential_evolution" | "de":
                # Differential Evolution (DE)
                self.F = kwargs.get("F", 0.1)
                self.p = kwargs.get("p", 0.1)
            case "cuckoo_search" | "cs":
                # Cuckoo Search (CS)
                self.beta = (beta := kwargs.get("beta", 1.0))  # Lévy exponent
                num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
                den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
                self.sigma = (num / den) ** (1 / beta)
                self.alpha = kwargs.get("alpha", 0.01)  # step size
                self.pa = kwargs.get("pa", 0.1)  # abandonment probability
            case "bat_algorithm" | "ba":
                # Bat Algorithm (BA)
                self.A = kwargs.get("A", 2.0)  # loudness
                self.r = kwargs.get("r", 0.5)  # pulse rate
                self.f_min = kwargs.get("f_min", 0.0)  # minimum frequency
                self.f_max = kwargs.get("f_max", 0.25)  # maximum frequency
                self.epsilon = kwargs.get("epsilon", 0.01)  # local random walk
            case _:
                raise ValueError(f"SO type not supported: {self.algorithm_type}")

        # Most of the algorithms make use of ``is_best_ever``
        self.is_best_ever = kwargs.get("is_best_ever", True)

    @utilmethod
    def update(
        self,
        candidates: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        global_best: np.ndarray | None = None,
        additional_conditions: np.ndarray = np.array(True),
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Updates the population and fitness based on the candidates.

        Simply checks which candidate positions are better than the
        current population and updates the population and fitness
        accordingly. If ``global_best`` is given, it is also updated
        based on the best candidate.

        Args:
            candidates (numpy.ndarray): The candidate solutions.
            population (numpy.ndarray): The current population.
            fitness (numpy.ndarray): The current fitness.
            global_best (numpy.ndarray | None, optional): The global
                best position. Defaults to ``None``.
            additional_conditions (numpy.ndarray, optional): Additional
                conditions to check for updating the population and
                fitness. Defaults to numpy.array(True).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]
            | tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        # Clip candidates (each dimension) within bounds
        candidates = np.clip(candidates, *self.bounds)
        population = population.copy()
        fitness = fitness.copy()

        # Compute fitness of candidates
        new_fitness = self.evaluate(candidates)

        # Update population and fitness where candidates are better
        mask = (new_fitness > fitness) & additional_conditions
        population[mask] = candidates[mask]
        fitness[mask] = new_fitness[mask]

        if (gb := global_best) is None:
            # New population + fitness
            return population, fitness

        if not self.is_best_ever or new_fitness.max() > self.fitness_fn(gb):
            # Update global best position based on new best
            global_best = candidates[new_fitness.argmax()]

        return population, fitness, global_best

    @override
    def initialize_population(
        self,
    ) -> np.ndarray:
        """Initializes the population.

        Simply generates random positions within the given bounds.

        Returns:
            numpy.ndarray: The initial population of shape (N, D).
        """
        # Initialize particle random positions and constant velocities
        return self.rng.uniform(*self.bounds, (self.N, self.D))

    @override
    def evolve(
        self,
        population: np.ndarray,
        fitness: Collection[float],
        *cache,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any]]:
        """Evolves the population.

        Runs the Swarm Optimization algorithm (update rule) based on
        :attr:`algorithm_type`. The ``cache`` argument can be used to
        store the velocities, local best positions, and global best
        position.

        Args:
            population (numpy.ndarray): The current population.
            fitness (typing.Collection[float]): The fitness of the
                current population in the same order.
            *cache: The cache to use for the algorithm.

        Raises:
            ValueError: If the algorithm type is not supported.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple[typing.Any]]:
            The new population, fitness, and cache.
        """
        # Convert to numpy array
        fitness = np.array(fitness)

        match self.algorithm_type:
            case "particle_swarm" | "ps":
                # Run Particle Swarm Optimization (PSO)
                return self.particle_swarm(population, fitness, *cache)
            case "differential_evolution" | "de":
                # Run Differential Evolution (DE)
                return self.differential_evolution(population, fitness, *cache)
            case "cuckoo_search" | "cs":
                # Run Cuckoo Search (CS)
                return self.cuckoo_search(population, fitness, *cache)
            case "bat_algorithm" | "ba":
                # Run Bat Algorithm (BA)
                return self.bat_algorithm(population, fitness, *cache)
            case _:
                raise ValueError(f"SO type not supported: {self.algorithm_type}")

    def particle_swarm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        """Particle Swarm Optimization (PSO) update rule.

        Computes the new velocities and positions based on the current
        population, fitness, and cache. The cache is used to store the
        velocities, local best positions, and global best position.

        Args:
            population (numpy.ndarray): The current population of
                particle positions.
            fitness (numpy.ndarray): The fitness of the current
                population in the same order.
            *cache: The cache to use for the algorithm.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple[numpy.ndarray, numpy.ndarray, float]]:
            The new population, fitness, and cache.
        """
        if len(cache) == 0:
            # Create cache if it does not exist with default values
            cache = np.zeros_like(population), population[:], population[0]

        # Unpack population: bests refer to positions, not fitnesses
        positions, (velocities, local_best, global_best) = population, cache

        if self.alpha3 is not None:
            # Compute repulsion term based on distance differences
            diff = positions[:, None, :] - positions[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            z = -np.sum(diff / (dist[..., None] + 1e-9), axis=1)
            z *= self.alpha3 * self.rng.random((self.N, self.D))
        else:
            z = 0

        # Generate 2 random vectors for every particle
        r1, r2 = self.rng.random((2, self.N, self.D))

        # Update velocities based on local and global forces
        velocities = (
            self.w * velocities
            + z
            + self.alpha1 * r1 * (local_best - positions)
            + self.alpha2 * r2 * (global_best - positions)
        )
        # Clip velocities, update positions with them and clip as well
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)
        positions = positions + velocities

        # Update local and global best positions and fitnesses
        update = self.update(positions, local_best, fitness, global_best)
        local_best, fitness, global_best = update

        return positions, fitness, (velocities, local_best, global_best)

    def differential_evolution(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        """Differential Evolution (DE) update rule.

        Computes the new positions based on the current population,
        fitness, and cache. Unlike in PSO, instead of local and global
        best positions, updates are done with respect to randomly chosen
        individuals. The cache in not used at all.

        Args:
            population (numpy.ndarray): The current population of
                particle positions.
            fitness (numpy.ndarray): The fitness of the current
                population in the same order.
            *cache: The cache to use for the algorithm (empty).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple]: The new
            population, fitness, and cache (empty).
        """
        # Unpack population: bests refer to positions, not fitnesses
        positions = population

        # Generate 3 random indices for every particle (mostly non-self)
        abc = [self.rng.choice(self.N, 3, self.N < 3) for _ in range(self.N)]
        a, b, c = np.array(abc).T

        # Compute mutant vector
        mutant = positions[a] + self.F * (positions[b] - positions[c])

        # Generate a random vector for every particle
        r = self.rng.random((self.N, self.D))

        # Compute trial vector based on crossover
        trial = np.where(r < self.p, mutant, positions)
        positions, fitness = self.update(trial, positions, fitness)

        return positions, fitness, cache

    def cuckoo_search(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        """Cuckoo Search (CS) update rule.

        Computes the new positions based on the current population,
        fitness, and cache. Cuckoo birds that represent the new
        positions are generated by Lévy flights and random walks. They
        can also abandon their nests with a certain probability. The
        cache is used to store the global best position, which is used
        to guide the Lévy flights.

        Args:
            population (numpy.ndarray): The current population of
                individual positions.
            fitness (numpy.ndarray): The fitness of the current
                population in the same order.
            *cache: The cache to use for the algorithm. It stores the
                global best position.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple[numpy.ndarray, numpy.ndarray, float]]:
            The new population, fitness, and cache.
        """
        if len(cache) == 0:
            # Create cache if it does not exist with default values
            cache = (population[0],)

        # Unpack population: bests refer to positions, not fitnesses
        positions, (global_best,) = population, cache

        # Generate new solutions by Lévy flights
        u = self.rng.normal(0, self.sigma, size=(self.N, self.D))
        v = self.rng.normal(0, 1, size=(self.N, self.D))
        steps = u / abs(v) ** (1 / self.beta)
        levy_flights = self.alpha * steps * (positions - global_best)

        # Apply Lévy flights to generate new positions
        new_positions = positions + levy_flights
        positions, fitness = self.update(new_positions, positions, fitness)

        # Generate new solutions by random walk
        d1, d2 = self.rng.integers(self.N, size=(2, self.N))
        random_walk = self.rng.random(size=(self.N, self.D)) * (
            positions[d1] - positions[d2]
        )

        # Apply random walk to generate new positions
        mask = self.rng.random((self.N, self.D)) < self.pa
        new_positions = positions.copy()
        new_positions[mask] += random_walk[mask]

        # Update positions and fitness where improvement is found
        update = self.update(new_positions, positions, fitness, global_best)
        positions, fitness, global_best = update

        return positions, fitness, (global_best,)

    def bat_algorithm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        """Bat Algorithm (BA) update rule.

        Computes the new positions based on the current population,
        fitness, and cache. Bats look for prey by flying around and
        emitting ultrasonic pulses. The pulses are emitted with a
        certain frequency and loudness. The cache is used to store the
        velocities and global best position.


        Args:
            population (numpy.ndarray): The current population of
                bat positions.
            fitness (numpy.ndarray): The fitness of the current
                population in the same order.
            *cache: The cache to use for the algorithm. It stores the
                velocities and global best position.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, tuple[numpy.ndarray, numpy.ndarray, float]]:
            The new population, fitness, and cache.
        """
        if len(cache) == 0:
            # Create cache if it does not exist with default values
            cache = (np.zeros_like(population), population[0])

        # Unpack population: bests refer to positions, not fitnesses
        positions, (velocities, global_best) = population, cache

        # Generate new frequencies
        beta = self.rng.random(self.N)
        frequencies = self.f_min + (self.f_max - self.f_min) * beta

        # Update velocities and positions
        velocities += (positions - global_best) * frequencies[:, None]
        new_positions = positions + velocities

        # Generate random numbers for each bat
        random_nums = self.rng.random(self.N)

        # Apply local random walk if random number is greater than pulse rate
        mask = random_nums > self.r
        new_positions[mask] = global_best + self.epsilon * self.rng.normal(
            size=(np.sum(mask), self.D)
        )

        # Update positions and fitness where improvement is found
        positions, fitness, global_best = self.update(
            candidates=new_positions,
            population=positions,
            fitness=fitness,
            global_best=global_best,
            additional_conditions=(self.rng.random(self.N) < self.A),
        )

        return positions, fitness, (velocities, global_best)
