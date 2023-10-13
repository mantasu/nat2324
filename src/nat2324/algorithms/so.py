import numpy as np
from typing import Any, Callable, Collection
from .base_runner import BaseRunner


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
        super().__init__()

        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.N = N
        self.D = D
        self.so_type = so_type
        self.parallelize_fitness = parallelize_fitness
        self.rng = np.random.default_rng(seed=seed)
        self._init_params(**kwargs)
    
    def _init_params(self, **kwargs):
        match self.so_type:
            case "plain":
                # W, alpha1, alpha2
                self.w = kwargs.get("w", 0.7)
                self.alpha1 = kwargs.get("alpha1", 2)
                self.alpha2 = kwargs.get("alpha2", 2)
                self.alpha3 = kwargs.get("alpha3", 1)
                self.max_vel = kwargs.get("max_vel_frac", 0.2) \
                               * (self.bounds[1] - self.bounds[0])
                self.is_best_ever = kwargs.get("is_best_ever", True)
            case _:
                raise ValueError(f"SO type not supported: {self.so_type}")
    
    def initialize_population(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        # Initialize particle random positions and constant velocities
        positions = self.rng.uniform(*self.bounds, (self.N, self.D))
        velocities = np.zeros_like(positions)

        # Init local and global bests
        local_best = np.copy(positions)
        global_best = np.copy(positions[0])

        return positions, (velocities, local_best, global_best)

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
            case "artificial_bee_colony" | "abc":
                return self.artificial_bee_colony(population, fitness, *cache)
            case "cuckoo_search" | "cs":
                return self.cuckoo_search(population, fitness, *cache)
            case "bat_algorithm" | "ba":
                return self.bat_algorithm(population, fitness, *cache)
            case _:
                raise ValueError(f"SO type not supported: {self.so_type}")

    def particle_swarm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        # Unpack population: bests refer to positions, not fitnesses
        positions, (velocities, local_best, global_best) = population, cache

        if self.alpha3 is not None:
            # Compute repulsion term based on distance differences
            diff = positions[:, None, :] - positions[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            z = np.sum(diff / (dist[..., None] + 1e-9), axis=1)
            z *= self.alpha3 * self.rng.random((self.N, self.D))
        else:
            z = 0

        # Generate a random vector for every particle
        r1 = self.rng.random((self.N, self.D))
        r2 = self.rng.random((self.N, self.D))

        # Update velocities based on local and global forces
        velocities = self.w * velocities + z \
                     + self.alpha1 * r1 * (local_best - positions) \
                     + self.alpha2 * r2 * (global_best - positions) \

        # Clip velocities, update positions with them and clip as well
        velocities = np.clip(velocities, -self.max_vel, self.max_vel)
        positions = np.clip(positions + velocities, *self.bounds)
        
        if self.parallelize_fitness:
            # Compute fitness in parallel fashion if N is large
            new_fitness = self.parallel_apply(self.fitness_fn, positions)
            new_fitness = np.array(new_fitness)
        else:
            # Compute fitness in serial fashion if N is relatively small
            new_fitness = np.apply_along_axis(self.fitness_fn, 1, positions)

        # Update local best positions and fitnesses
        mask = new_fitness > fitness
        local_best[mask] = positions[mask]
        fitness[mask] = new_fitness[mask]
        
        if not self.is_best_ever \
           or new_fitness.max() > self.fitness_fn(global_best):
            # Update global best position and fitness
            global_best = positions[new_fitness.argmax()]

        return positions, fitness, (velocities, local_best, global_best)

    def differential_evolution(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        positions, (velocities, local_best, global_best) = population, cache
        # Differential Evolution parameters
        F = 0.5  # mutation factor
        CR = 0.9  # crossover probability

        # Mutation
        a, b, c = self.rng.choice(self.N, (3,self.N), replace=False)
        mutant = positions[a] + F * (positions[b] - positions[c])
        mutant = np.clip(mutant, *self.bounds)

        # Crossover
        cross_points = self.rng.random((self.N, self.D)) < CR
        cross_points[self.rng.integers(0, self.D)] = True
        trial = np.where(cross_points[:, None], mutant, positions)

        # Selection
        trial_fitness = np.apply_along_axis(self.fitness_fn, 1, trial)
        
        mask = trial_fitness > fitness
        
        positions[mask] = trial[mask]
        
        fitness[mask] = trial_fitness[mask]

        return positions, fitness, (velocities ,local_best ,global_best)


    def artificial_bee_colony(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        positions, (velocities, local_best, global_best) = population, cache

        # Artificial Bee Colony parameters
        limit = 100  # abandonment limit parameter
        trials = np.zeros(self.N)  # reset trials counter

        # Employed Bee Phase
        j = self.rng.choice(self.D, self.N)
        k = self.rng.choice(self.N, self.N)
        phi = self.rng.uniform(-1, 1, self.N)

        new_positions = positions.copy()
        new_positions[np.arange(self.N), j] = positions[np.arange(self.N), j] + phi * (positions[np.arange(self.N), j] - positions[k, j])
        new_positions = np.clip(new_positions, *self.bounds)

        new_fitness = np.apply_along_axis(self.fitness_fn, 1, new_positions)

        # Update positions and fitness where improvement is found
        mask = new_fitness > fitness
        positions[mask] = new_positions[mask]
        fitness[mask] = new_fitness[mask]
        trials[~mask] += 1

        # Onlooker Bee Phase
        prob = (0.9 * fitness / np.max(fitness)) + 0.1
        i = self.rng.choice(self.N, p=prob/np.sum(prob))

        j = self.rng.choice(self.D)
        k = self.rng.choice(self.N)
        while k == i:
            k = self.rng.choice(self.N)

        phi = self.rng.uniform(-1, 1)
        new_position = positions[i]
        new_position[j] = positions[i][j] + phi * (positions[i][j] - positions[k][j])
        new_position = np.clip(new_position, *self.bounds)

        new_fitness = self.fitness_fn(new_position)
        if new_fitness > fitness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness
            trials[i] = 0  # reset trials counter if improvement is found
        else:
            trials[i] += 1

        # Scout Bee Phase
        i_max_trial = np.argmax(trials)
        if trials[i_max_trial] > limit:
            positions[i_max_trial] = self.rng.uniform(*self.bounds, self.D)  # re-initialize position
            fitness[i_max_trial] = self.fitness_fn(positions[i_max_trial])  # re-calculate fitness
            trials[i_max_trial] = 0  # reset trials counter

        return positions, fitness, (velocities, local_best, global_best)

    def cuckoo_search(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        positions, (velocities, local_best, global_best) = population, cache

        # Cuckoo Search parameters
        alpha = 0.01  # step size
        pa = 0.25  # probability of replacement

        # Generate new solutions by Levy flight
        levy = np.power(self.rng.normal(0, 1, self.N), -1/1.5)
        new_positions = positions + alpha * levy[:, None] * (positions - global_best)
        new_positions = np.clip(new_positions, *self.bounds)

        new_fitness = np.apply_along_axis(self.fitness_fn, 1, new_positions)

        # Update positions and fitness where improvement is found
        mask = new_fitness > fitness
        positions[mask] = new_positions[mask]
        fitness[mask] = new_fitness[mask]

        # Randomly replace solutions with new solutions
        mask = self.rng.random(self.N) < pa
        positions[mask] = self.rng.uniform(*self.bounds, (np.sum(mask), self.D))
        fitness[mask] = np.apply_along_axis(self.fitness_fn, 1, positions[mask])

        return positions, fitness, (velocities, local_best, global_best)

    def bat_algorithm(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        *cache: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, float]]:
        positions, (velocities, local_best, global_best) = population, cache
        # Bat Algorithm parameters
        A = 0.5  # loudness
        r = 0.5  # pulse rate

        # Velocity update
        velocities += (positions - global_best) * self.rng.uniform(-1, 1, (self.N,self.D))

        # Position update
        new_positions = positions + velocities
        new_positions = np.clip(new_positions, *self.bounds)

        mask = self.rng.random((self.N,self.D)) > r
        new_positions[mask] += self.rng.uniform(-1, 1, (mask.sum(),)) * global_best

        new_fitness = np.apply_along_axis(self.fitness_fn, 1,new_positions)
        
        mask = (new_fitness > fitness) & (self.rng.random((self.N,)) < A)
        positions[mask] = new_positions[mask]
        fitness[mask] = new_fitness[mask]

        
        return positions, fitness,(velocities ,local_best ,global_best)
