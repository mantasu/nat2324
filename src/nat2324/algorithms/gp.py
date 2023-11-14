import copy
from typing import Callable

import numpy as np

from ..utils import GPTree, NonTerminal, Terminal
from ..utils.decorators import override, utilmethod
from .base_runner import BaseRunner


class GeneticProgrammingAlgorithm(BaseRunner):
    def __init__(
        self,
        fitness_fn: Callable[..., float],
        terminals: set[Terminal],
        non_terminals: set[NonTerminal],
        N: int = 1000,
        min_depth: int = 2,
        max_depth: int = 6,
        p_c: float = 0.7,
        p_m: float = 0.5,
        tournament_size: int = 20,
        parallelize_fitness: bool = False,
        seed: int | None = None,
    ):
        super().__init__(fitness_fn, N, parallelize_fitness, seed=seed)

        # Set the parameters
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.p_c = p_c
        self.p_m = p_m
        self.tournament_size = tournament_size

    @utilmethod
    def generate_individual(self, **kwargs) -> GPTree:
        """Generates a random genetic program tree.

        This function generates a random genetic program tree as defined
        in :meth:`.GPTree.generate_tree` using the specified parameters.
        If no parameters are specified, then the default values are
        used.

        Args:
            **kwargs: Keyword arguments to be passed to
                :meth:`.GPTree.generate_tree`.

        Returns:
            GPTree: A randomly generated genetic program tree.
        """
        # Set default values in case they are not in kwargs
        kwargs.setdefault("min_depth", self.min_depth)
        kwargs.setdefault("max_depth", self.max_depth)
        kwargs.setdefault("terminals", self.terminals)
        kwargs.setdefault("non_terminals", self.non_terminals)
        kwargs.setdefault("rng", self.rng)

        return GPTree.generate_tree(**kwargs)

    @utilmethod
    def swap_nodes(
        self,
        node1: GPTree,
        node2: GPTree,
    ) -> tuple[GPTree, GPTree]:
        """Swaps two nodes of 2 different trees.

        This function swaps two nodes of two different trees. It
        preserves the parent-child relationships of the nodes and
        returns the two trees with the swapped nodes (the children order
        is preserved).

        Note:
            If one of the nodes is the root node, i.e., it has no
            parent, then the other node that is detached from its parent
            (if it has one) effectively becomes the new root node.

        Args:
            node1 (GPTree): The first node to be swapped.
            node2 (GPTree): The second node to be swapped.

        Returns:
            tuple[GPTree, GPTree]: A tuple containing the two trees with
            the swapped nodes.
        """
        # Get children of node 1 parent and swap node 1 with node 2
        kids1 = (node1,) if node1.parent is None else node1.parent.children
        kids1 = tuple(child if child != node1 else node2 for child in kids1)

        # Get children of node 2 parent and swap node 2 with node 1
        kids2 = (node2,) if node2.parent is None else node2.parent.children
        kids2 = tuple(child if child != node2 else node1 for child in kids2)

        # Swap the parent references (children order becomes arbitrary)
        node1.parent, node2.parent = node2.parent, node1.parent

        if node1.parent is not None:
            # Change the order of children
            node1.parent.children = kids2

        if node2.parent is not None:
            # Change the order of children
            node2.parent.children = kids1

        return node1, node2

    @override
    def initialize_population(self) -> list[GPTree]:
        """Initializes the population.

        This function initializes the population by generating
        :attr:`.N` random individuals (genetic program trees).

        Returns:
            list[GPTree]: A list of randomly generated genetic program
            trees.
        """
        return [self.generate_individual() for _ in range(self.N)]

    @override
    def evolve(
        self,
        population: list[GPTree],
        fitness: np.ndarray,
    ) -> tuple[list[GPTree], list[float]]:
        """Evolves the population of genetic programs.

        This function evolves the population of genetic programs by
        performing selection, crossover, mutation, and validation. The
        new fitness of the individuals in the population is computed
        after the performed evolution operations.

        Args:
            population (list[GPTree]): The population of individuals
                (genetic program trees) to be evolved.
            fitnesses (numpy.ndarray): The fitness scores of the trees
                in the population. They must be in the same order as the
                individuals in the population.

        Returns:
            tuple[list[GPTree], numpy.ndarray]: A tuple containing the
            new population and its corresponding fitness scores.
        """
        # Selection, crossover, mutation, validation, and evaluation
        population = self.selection(population, fitness)
        population = self.crossover(population)
        population = self.mutation(population)
        population = self.validation(population)
        fitness = self.evaluate(population)

        return population, fitness

    def selection(
        self,
        population: list[GPTree],
        fitness: np.ndarray,
    ) -> list[GPTree]:
        """Selects the individuals to be used in the next generation.

        This function selects the individuals to be used in the next
        generation using tournament selection. The number of individuals
        to be selected is equal to the size of the population.

        Args:
            population (list[GPTree]): The population of individuals
                (genetic program trees) to be selected from.
            fitness (numpy.ndarray): The fitness scores of the trees in
                the population. They must be in the same order as the
                individuals in the population.

        Returns:
            list[GPTree]: A list of selected individuals.
        """
        # Select tournament groups and winners in each group
        N, population = len(population), np.array(population)
        idx = self.rng.integers(N, size=(N, self.tournament_size))
        population = population[idx][range(N), np.argmax(fitness[idx], 1)]

        # Make a copies of individuals (this is very expensive)
        # population = [copy.deepcopy(ind) for ind in population]
        population = [ind.copy() for ind in population]

        return population

    def crossover(self, population: list[GPTree]) -> list[GPTree]:
        """Performs crossover on the population (in-place).

        This function performs crossover on the population by swapping
        random nodes of two individuals with a probability of
        :attr:`.p_c` (crossover probability).

        Note:
            The number of individuals in the population must be even.

        Args:
            population (list[GPTree]): The population of individuals
                (genetic program trees) to be crossed over.

        Returns:
            list[GPTree]: A list of crossed over individuals.
        """
        for ind1, ind2 in zip(population[::2], population[1::2]):
            if self.rng.random() < self.p_c:
                # Swap a random node from each individual
                node1 = self.rng.choice(ind1.descendants)
                node2 = self.rng.choice(ind2.descendants)
                self.swap_nodes(node1, node2)

        return population

    def mutation(self, population: list[GPTree]) -> list[GPTree]:
        """Performs mutation on the population (in-place).

        This function performs mutation on the population by mutating at
        random nodes of each individual with a probability of
        :attr:`.p_m` (mutation probability).

        Args:
            population (list[GPTree]): The population of individuals
                (genetic program trees) to be mutated.

        Returns:
            list[GPTree]: A list of mutated individuals.
        """
        # Define all possible mutation types (note: re-gen is there too)
        TYPES = [
            "grow_up",
            "grow_down",
            "shrink",
            "hoist",
            "renew",
            "regen",
        ]

        for i, individual in enumerate(population):
            if self.rng.random() < self.p_m:
                # Select a random mutation type
                mutation_type = self.rng.choice(TYPES)

                match mutation_type:
                    case "grow_up":
                        # Grow an ancestor subtree from a random node
                        population[i] = self.grow_up(individual)
                    case "grow_down":
                        # Grow a descendant subtree from a random node
                        population[i] = self.grow_down(individual)
                    case "shrink":
                        # Shrink a random subtree to a terminal
                        population[i] = self.shrink(individual)
                    case "hoist":
                        # Hoist a random subtree to the root
                        population[i] = self.hoist(individual)
                    case "renew":
                        # Replace a random symbol with a new one
                        population[i] = self.renew(individual)
                    case "regen":
                        # Replace the individual with a new one
                        population[i] = self.generate_individual()

        return population

    def validation(self, population: list[GPTree]) -> list[GPTree]:
        """Performs validation on the population (in-place).

        This function performs validation on the population by
        validating each individual. If an individual is invalid, then it
        is replaced with a valid one. An individual is invalid if its
        height is less than :attr:`.min_depth` or greater than
        :attr:`.max_depth`.

        Warning:
            Individuals that have a height greater than
            :attr:`.max_depth` are replaced with a valid individual
            using the :meth:`.hoist` method. This could still result in
            an individual being too tall but it's much faster than
            performing full trimming.

        Args:
            population (list[GPTree]): The population of individuals
                (genetic program trees) to be validated.

        Returns:
            list[GPTree]: A list of validated individuals.
        """
        for i in range(len(population)):
            if population[i].height > self.max_depth:
                # Could do while but it's too slow
                population[i] = self.hoist(population[i])

            if population[i].height < self.min_depth:
                # Replace the individual with a valid one
                population[i] = self.generate_individual()

        return population

    def grow_up(self, tree: GPTree) -> GPTree:
        # Select a random subtree to grow up from
        subtree = self.rng.choice(tree.descendants)

        # Generate an ancestor subtree to attach on top of subtree
        max_depth = max(2, self.max_depth - subtree.height)
        ancestor = self.generate_individual(max_depth=max_depth)

        # Select a random ancestor-descendant node to swap out
        descendent = self.rng.choice(ancestor.descendants)
        self.swap_nodes(subtree, descendent)

        return ancestor

    def grow_down(self, tree: GPTree) -> GPTree:
        # Select a random subtree to grow down from
        subtree = self.rng.choice(tree.descendants)

        # Generate a descendant and attach it instead of the subtree
        max_depth = max(1, self.max_depth - (tree.height - subtree.height))
        descendant = self.generate_individual(max_depth=max_depth)
        self.swap_nodes(subtree, descendant)

        return tree

    def shrink(self, tree: GPTree) -> GPTree:
        # Select a random subtree and a random terminal
        subtree = self.rng.choice(tree.descendants)
        terminal = self.rng.choice(list(self.terminals))
        terminal_node = GPTree(str(terminal), terminal)

        # Replace the subtree with a terminal node
        self.swap_nodes(subtree, terminal_node)

        return tree

    def hoist(self, tree: GPTree) -> GPTree:
        # Select a random subtree and make it root
        subtree = self.rng.choice(tree.descendants)
        subtree.parent = None

        return subtree

    def renew(self, tree: GPTree) -> GPTree:
        # Select a random subtree to renew based on symbol
        subtree = self.rng.choice(tree.descendants)

        if subtree.name in self.terminals:
            # Select a random terminal to replace the leaf with
            terminal = self.rng.choice(list(self.terminals))
            new_node = GPTree(str(terminal), terminal)
        else:
            # Select a random non-terminal to replace the subtree with
            non_terminal = self.rng.choice(list(self.non_terminals))
            new_node = GPTree(str(non_terminal), non_terminal)

            for child in subtree.children[: non_terminal.arity]:
                # Attach the subtree children to the new node
                child.parent = new_node

            while len(new_node.children) < non_terminal.arity:
                # Attach random terminals to new node until full
                terminal = self.rng.choice(list(self.terminals))
                child_node = GPTree(str(terminal), terminal)
                child_node.parent = new_node

        # Replace the subtree with the new node
        self.swap_nodes(subtree, new_node)

        return tree
