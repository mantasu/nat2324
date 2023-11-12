import copy
from typing import Callable

import numpy as np

from ..utils import GPTree, NonTerminal, Terminal
from .base_runner import BaseRunner


class GeneticProgrammingAlgorithm(BaseRunner):
    def __init__(
        self,
        fitness_fn: Callable[..., float],
        N: int = 10000,
        min_depth: int = 2,
        max_depth: int = 5,
        p_c: float = 0.7,
        p_m: float = 0.1,
        tournament_size: int = 5,
        mutation_type: str = "mixed",
        terminals: set[Terminal] = {},  # Terminal.get_default(),
        non_terminals: set[NonTerminal] = {},  # NonTerminal.get_default(),
        parallelize_fitness: bool = False,
        seed: int | None = None,
    ):
        super().__init__(fitness_fn, N, parallelize_fitness, seed=seed)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.p_c = p_c
        self.p_m = p_m
        self.tournament_size = tournament_size
        self.mutation_type = mutation_type
        self.terminals = terminals
        self.non_terminals = non_terminals

    def generate_individual(self, **kwargs) -> GPTree:
        # Set default values in case they are not in kwargs
        kwargs.setdefault("min_depth", self.min_depth)
        kwargs.setdefault("max_depth", self.max_depth)
        kwargs.setdefault("terminals", self.terminals)
        kwargs.setdefault("non_terminals", self.non_terminals)
        kwargs.setdefault("rng", self.rng)

        return GPTree.generate_tree(**kwargs)

    def initialize_population(self) -> list[GPTree]:
        return [self.generate_individual() for _ in range(self.N)]

    def evolve(
        self,
        population: list[GPTree],
        fitness: list[float],
    ) -> tuple[list[GPTree], list[float]]:
        """Performs a genetic algorithm optimization.

        This function performs a genetic algorithm optimization for a
        given fitness function. It uses roulette wheel sampling for
        selecting individuals for crossover and mutation.

        Args:
            population (list[GPTree]): The population of individuals
                (program trees) to be evolved.
            fitnesses (list[float]): The fitness scores of the trees in
                the population. They must be in the same order as the
                individuals in the population.

        Returns:
            tuple[list[GPTree], list[float]]: A tuple containing the
            new population and its corresponding fitness scores.
        """
        # Select parents, perform crossover and mutation
        # print("SEL", len(population))
        parents = self.tournament_selection(population, fitness)
        # print("CROSS", len(population))
        children = self.crossover(parents)
        # print("MUTA", len(population))
        population = self.mutation(children)
        # print("VALI", len(population))
        population = self.validation(population)

        # Compute fitness for each individual (parallelize if large N)

        # print("EV", len(population))
        fitness = self.evaluate(population)
        # print("EV Done")

        # (
        #     self.parallel_apply(self.fitness_fn, population)
        #     if self.parallelize_fitness
        #     else [self.fitness_fn(tree) for tree in population]
        # )

        return population, fitness

    def hoist(self, tree: GPTree) -> GPTree:
        # Select a random subtree
        new_tree = self.rng.choice(tree.descendants)
        new_tree.parent = None

        tree = self.generate_individual(
            max_depth=max(2, self.max_depth - new_tree.height)
        )
        subtree = self.rng.choice(tree.descendants)

        children = tuple(
            child if child != subtree else new_tree for child in subtree.parent.children
        )
        new_tree.parent, subtree.parent = subtree.parent, None
        new_tree.parent.children = children

        return tree

    def shrink(self, tree: GPTree) -> GPTree:
        # Select a random subtree
        subtree = self.rng.choice(tree.descendants)
        terminal = self.rng.choice(list(self.terminals))

        # Replace the subtree with a terminal node
        terminal_node = GPTree(str(terminal), terminal)

        children = tuple(
            child if child != subtree else terminal_node
            for child in subtree.parent.children
        )
        terminal_node.parent, subtree.parent = subtree.parent, None
        terminal_node.parent.children = children

        # subtree.parent = terminal_node
        return tree

    def renew(self, tree: GPTree) -> GPTree:
        # Select a random subtree
        subtree = self.rng.choice(tree.descendants)

        if subtree.name in self.terminals:
            symbol = self.rng.choice(list(self.terminals))
            new_node = GPTree(str(symbol), symbol)
        else:
            symbol = self.rng.choice(list(self.non_terminals))
            new_node = GPTree(str(symbol), symbol)

            for child in subtree.children[: symbol.arity]:
                child.parent = new_node

            while len(new_node.children) < symbol.arity:
                terminal = self.rng.choice(list(self.terminals))
                child_node = GPTree(str(terminal), terminal)
                child_node.parent = new_node

        children = tuple(
            child if child != subtree else new_node for child in subtree.parent.children
        )

        new_node.parent, subtree.parent = subtree.parent, None
        new_node.parent.children = children

        return tree

    def grow(self, tree: GPTree):
        # Select a random subtree
        subtree = self.rng.choice(tree.descendants)

        # Calculate the new maximum depth
        max_depth = max(1, self.max_depth - (tree.height - subtree.height))
        # Replace the subtree with a new randomly generated tree of maximum depth new_max_depth
        new_tree = self.generate_individual(max_depth=max_depth)

        children = tuple(
            child if child != subtree else new_tree for child in subtree.parent.children
        )
        new_tree.parent, subtree.parent = subtree.parent, None
        new_tree.parent.children = children

        return tree

    def mutate(self, tree):
        mutation_type = self.mutation_type

        if mutation_type == "mixed":
            mutation_type = self.rng.choice(
                ["hoist", "shrink", "grow", "renew", "regen"]
            )
            # mutation_type = self.rng.choice(["shrink", "renew", "regen"])

        if mutation_type == "hoist":
            return self.hoist(tree)
        elif mutation_type == "shrink":
            return self.shrink(tree)
        elif mutation_type == "renew":
            return self.renew(tree)
        elif mutation_type == "grow":
            return self.grow(tree)
        elif mutation_type == "regen":
            return self.generate_individual()

    def cross(self, parent1: GPTree, parent2: GPTree) -> tuple[GPTree, GPTree]:
        node1 = self.rng.choice(parent1.descendants)
        node2 = self.rng.choice(parent2.descendants)

        # Create new children, however, note that swapped children still have references to their old parents
        children1 = tuple(
            child if child != node1 else node2 for child in node1.parent.children
        )
        children2 = tuple(
            child if child != node2 else node1 for child in node2.parent.children
        )

        # Swap the parent references and swap the orders
        node1.parent, node2.parent = node2.parent, node1.parent
        node1.parent.children = children2
        node2.parent.children = children1

        # return copy.deepcopy(parent1), copy.deepcopy(parent2)
        return parent1, parent2

    def trim(self, tree: GPTree, curr_depth: int = 1) -> GPTree:
        # Check if the current depth is greater than the maximum depth
        if curr_depth > self.max_depth:
            # Replace the tree with a random terminal
            terminal = self.rng.choice(list(self.terminals))
            terminal_node = GPTree(str(terminal), terminal)

            children = tuple(
                child if child != tree else terminal_node
                for child in tree.parent.children
            )
            terminal.parent, tree.parent = tree.parent, None
            terminal.parent.children = children
        else:
            # Iterate over each child
            for child in tree.children:
                # Trim the child
                self.trim(child, curr_depth + 1)

    def crossover(self, population: list[GPTree]) -> list[GPTree]:
        # Perform crossover on each pair of individuals
        return [
            child
            for parent1, parent2 in zip(population[::2], population[1::2])
            for child in (
                self.cross(parent1, parent2)
                if self.rng.random() < self.p_c
                else (parent1, parent2)
            )
        ]

    def mutation(self, population: list[GPTree]) -> list[GPTree]:
        # Perform mutation on each individual
        return [
            self.mutate(tree) if self.rng.random() < self.p_m else tree
            for tree in population
        ]

    def validation(self, population: list[GPTree]) -> list[GPTree]:
        # Iterate over each individual in the population
        for i, individual in enumerate(population):
            # Check if the individual's height is within the specified range
            if individual.height > self.max_depth:
                # If not, replace the individual with a new one (could do while but it's slow)
                individual = self.rng.choice(individual.descendants)
                population[i] = individual
                individual.parent = None
                # individual = individual.descendants[0]
                # individual.parent = None

            if individual.height < self.min_depth:
                # If not, replace the individual with a new one
                population[i] = self.generate_individual()

            # population[i] = individual

            # population[i] = self.trim(individual)
            # population[i] = self.generate_individual()

        return population

    # def tournament_selection(
    #     self, population: list[GPTree], fitness: list[float]
    # ) -> GPTree:
    #     # Select a random sample of individuals from the population
    #     sample = self.rng.choice(population, size=self.tournament_size)
    #     # Calculate the fitnesses of the individuals in the sample
    #     sample_fitnesses = [fitness[i] for i in sample]
    #     # Return the individual with the highest fitness
    #     return sample[np.argmax(sample_fitnesses)]

    def tournament_selection(
        self,
        population: list[GPTree],
        fitness: np.ndarray,
    ) -> list[GPTree]:
        # print("\tNumpys")
        # Population size
        N = len(population)
        population = np.array(population)

        # print("\tSELEMAS")

        # Select random groups and determine the best individuals
        idx = self.rng.integers(N, size=(N, self.tournament_size))
        # print(idx.shape, idx)
        population = population[idx][range(N), np.argmax(fitness[idx], 1)]
        # print(fitness[idx].shape)
        # print(np.argmax(fitness[idx], 1).shape, np.argmax(fitness[idx], 1))

        # print("\tCOPIES")
        num = 0
        pop_set = set()
        # pop = [ind if copy.deepcopy(ind) for ind in population]

        for i, ind in enumerate(population):
            if ind in pop_set:
                num += 1
                population[i] = copy.deepcopy(ind)
            else:
                pop_set.add(ind)

        # print("\tCOPIES", num)

        return population
