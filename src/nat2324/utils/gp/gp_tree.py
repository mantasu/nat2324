import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

from .symbol import NonTerminal, Terminal


class GPTree(Node):
    def __init__(
        self,
        name: str = None,
        symbol: Terminal | NonTerminal | None = None,
        children: list["GPTree"] = [],
    ):
        super().__init__(name, children=children)
        self.symbol = symbol

    @classmethod
    def generate_tree(
        cls,
        terminals: set[Terminal],
        non_terminals: set[NonTerminal],
        min_depth: int = 2,
        max_depth: int = 5,
        rng: np.random.Generator | None = None,
    ) -> "GPTree":
        if rng is None:
            # Default random number generator
            rng = np.random.default_rng()

        # Use shorter naming for readability and to allow random choice
        terms, non_terms = list(terminals), list(non_terminals)

        if max_depth <= 0 or (min_depth <= 0 and rng.random() < 0.5):
            # Set node as terminal (20% chance to become a leaf node)
            terminal = rng.choice(terms, p=[t.p for t in terms])
            node = cls(name=str(terminal), symbol=terminal)
        else:
            # Choose a non-terminal, generate children, and init node
            non_terminal = rng.choice(non_terms, p=[nt.p for nt in non_terms])
            children = [
                cls.generate_tree(
                    min_depth=min_depth - 1,
                    max_depth=max_depth - 1,
                    terminals=terminals,
                    non_terminals=non_terminals,
                    rng=rng,
                )
                for _ in range(non_terminal.arity)
            ]
            node = cls(str(non_terminal), non_terminal, children=children)

        return node

    def compute(self, **kwargs) -> Terminal.TYPE:
        if isinstance(self.symbol, Terminal) and self.symbol.is_variable:
            return kwargs[str(self.symbol)]
        elif isinstance(self.symbol, Terminal):
            return self.symbol.value
        elif isinstance(self.symbol, NonTerminal) and self.symbol.is_flow:
            return self.symbol(*self.children, **kwargs)
        elif isinstance(self.symbol, NonTerminal):
            return self.symbol(*[child(**kwargs) for child in self.children])

    def show(self):
        for pre, _, node in RenderTree(self):
            # Print connector + node's name
            print("%s%s" % (pre, node.name))

    def export(self, filepath: str):
        # Export tree as image using Graphviz
        DotExporter(self).to_picture(filepath)

    def __call__(self, **kwargs) -> Terminal.TYPE:
        return self.compute(**kwargs)
