import numpy as np
import scipy.special
from typing import Any, Callable, Collection
from .base_runner import BaseRunner
from math import gamma


class SwarmOptimization(BaseRunner):
    def __init__(
        self, 
        fitness_fn: Callable[[np.ndarray], float],
        bounds: tuple = (-100, 100),
        N: int = 25,
        D: int = 10,
        so_type: str = "pso",
        parallelize_fitness: bool = False,
        seed: int | None = None,
        **kwargs,
    ):  
        super().__init__(N=N, seed=seed)

        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.D = D
        self.so_type = so_type
        self.parallelize_fitness = parallelize_fitness
        self._init_params(**kwargs)
    
    def _init_params(self, **kwargs):
        match self.so_type:
            case "particle_swarm_optimization" | "pso":
                # W, alpha1, alpha2
                self.w = kwargs.get("w", 0.5)
                self.alpha1 = kwargs.get("alpha1", 2)
                self.alpha2 = kwargs.get("alpha2", 2)
                self.alpha3 = kwargs.get("alpha3", None)
                self.max_vel = kwargs.get("max_vel_frac", 0.2) \
                               * (self.bounds[1] - self.bounds[0])
            case "differential_evolution" | "de":
                self.F = kwargs.get("F", 0.5)
                self.p = kwargs.get("p", 0.9)
            case "cuckoo_search" | "cs":
                self.beta = (beta := kwargs.get("beta", 1.6))  # Lévy exponent
                num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
                den = gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
                self.sigma = (num / den) ** (1 / beta)
                self.alpha = kwargs.get("alpha", 0.05)  # step size
                self.pa = kwargs.get("pa", 0.25)  # abandonment probability
            case "bat_algorithm" | "ba":
                self.A = kwargs.get('A', 0.5) # loudness
                self.r = kwargs.get('r', 0.5)  # pulse rate
                self.f_min = kwargs.get("f_min", 0)  # minimum frequency
                self.f_max = kwargs.get("f_max", 1)  # maximum frequency
                self.epsilon = kwargs.get('epsilon', 0.005)  # local random walk
            case _:
                raise ValueError(f"SO type not supported: {self.so_type}")
        
        # Most of the algorithms make use of ``is_best_ever``
        self.is_best_ever = kwargs.get("is_best_ever", True)
    
    def initialize_population(
        self,
    ) -> tuple[np.ndarray]:
        # Initialize particle random positions and constant velocities
        return self.rng.uniform(*self.bounds, (self.N, self.D))

    def evolve(
        self,
        population: np.ndarray,
        fitness: Collection[float],
        *cache,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any]]:
        # Convert to numpy array
        fitness = np.array(fitness)

        match self.so_type:
            case "particle_swarm_optimization" | "pso":
                return self.particle_swarm(population, fitness, *cache)
            case "differential_evolution" | "de":
                return self.differential_evolution(population, fitness, *cache)
            case "cuckoo_search" | "cs":
                return self.cuckoo_search(population, fitness, *cache)
            case "bat_algorithm" | "ba":
                return self.bat_algorithm(population, fitness, *cache)
            case _:
                raise ValueError(f"SO type not supported: {self.so_type}")
    
    def update(
        self,
        candidates: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        global_best: np.ndarray | None = None,
        additional_conditions: np.ndarray = np.array(True),
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Clip candidates (each dimension) within bounds
        candidates = np.clip(candidates, *self.bounds)
        population = population.copy()
        fitness = fitness.copy()

        if self.parallelize_fitness:
            # Compute fitness in parallel fashion if N is large
            new_fitness = self.parallel_apply(self.fitness_fn, candidates)
            new_fitness = np.array(new_fitness)
        else:
            # Compute fitness in serial fashion if N is relatively small
            new_fitness = np.apply_along_axis(self.fitness_fn, 1, candidates)

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

    def particle_swarm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        if len(cache) == 0:
            # Create cache if it does not exist with default values
            cache = np.zeros_like(population), population[:], population[0]

        # Unpack population: bests refer to positions, not fitnesses
        positions, (velocities, local_best, global_best) = population, cache

        if self.alpha3 is not None:
            # Compute repulsion term based on distance differences
            diff = positions[:, None, :] - positions[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            z = np.sum(diff / (dist[..., None] + 1e-6), axis=1)
            z *= self.alpha3 * self.rng.random((self.N, self.D))
        else:
            z = 0

        # Generate 2 random vectors for every particle
        r1, r2 = self.rng.random((2, self.N, self.D))

        # Update velocities based on local and global forces
        velocities = self.w * velocities + z \
                     + self.alpha1 * r1 * (local_best - positions) \
                     + self.alpha2 * r2 * (global_best - positions) \

        # Clip velocities, update positions with them and clip as well
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)
        positions = positions + velocities

        # Update local and global best positions and fitnesses
        update = self.update(positions, local_best, fitness, global_best)
        local_best, fitness, global_best = update

        # Clip candidates (each dimension) within bounds
        # candidates = np.clip(positions, *self.bounds)
        # local_best = local_best.copy()
        # local_best = local_best.copy()
        # fitness = fitness.copy()

        # if self.parallelize_fitness:
        #     # Compute fitness in parallel fashion if N is large
        #     new_fitness = self.parallel_apply(self.fitness_fn, positions)
        #     new_fitness = np.array(new_fitness)
        # else:
        #     # Compute fitness in serial fashion if N is relatively small
        #     new_fitness = np.apply_along_axis(self.fitness_fn, 1, positions)

        # # Update local_best and fitness where positions are better
        # mask = (new_fitness > fitness)
        # local_best[mask] = positions[mask]
        # fitness[mask] = new_fitness[mask]
        
        # if (gb := global_best) is None:
        #     # New local_best + fitness
        #     return local_best, fitness
        
        # if not self.is_best_ever or new_fitness.max() > self.fitness_fn(gb):
        #     # Update global best position based on new best
        #     global_best = positions[new_fitness.argmax()]

        return positions, fitness, (velocities, local_best, global_best)

    def differential_evolution(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        # Unpack population: bests refer to positions, not fitnesses
        positions = population

        # indices = np.arange(self.N)
        # idx = np.empty((self.N, 3), dtype=int)
        # for i in range(self.N):
        #     idx[i] = self.rng.choice(indices[indices!=i], size=3, replace=False)

        # a, b, c = idx.T

        # Generate 3 random indices for every particle (mostly non-self)
        abc = [self.rng.choice(self.N, 3, self.N < 3) for _ in range(self.N)]
        # abc = [self.rng.choice(np.arange(self.N)[np.arange(self.N)!=i], size=3, replace=self.N < 4) for i in range(self.N)]
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

        # Apply Lévy flights and random walk to generate new positions
        new_positions = positions + levy_flights
        positions, fitness = self.update(new_positions, positions, fitness)

        # Generate new solutions by random walk
        d1, d2 = self.rng.integers(self.N, size=(2, self.N))
        random_walk = self.rng.random(size=(self.N, self.D)) * (positions[d1] - positions[d2])
        mask = (self.rng.random((self.N, self.D)) < self.pa)
        new_positions = positions.copy()
        new_positions[mask] += random_walk[mask]

        update = self.update(new_positions, positions, fitness, global_best)
        positions, fitness, global_best = update

        return positions, fitness, (global_best,)

    def bat_algorithm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
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
        new_positions[mask] = global_best + self.epsilon * self.rng.normal(size=(np.sum(mask), self.D))

        # Update positions and fitness where improvement is found
        positions, fitness, global_best = self.update(
            candidates=new_positions,
            population=positions,
            fitness=fitness,
            global_best=global_best,
            additional_conditions=(self.rng.random(self.N) < self.A),
        )

        return positions, fitness, (velocities, global_best)
