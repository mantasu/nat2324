import numpy as np
from anytree import Node, RenderTree

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
        min_depth: int = 2,
        max_depth: int = 5,
        terminals: set[Terminal] = set(),
        non_terminals: set[NonTerminal] = set(),
        rng: np.random.Generator | None = None,
    ) -> "GPTree":
        if rng is None:
            # Default random number generator
            rng = np.random.default_rng()

        if terminals == set():
            # Get the default terminals
            terminals = Terminal.get_default()
            terminals = Terminal.validate_ps(terminals)

        if non_terminals == set():
            # Get the default non-terminals
            non_terminals = NonTerminal.get_default()
            non_terminals = NonTerminal.validate_ps(non_terminals)

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
                if i != 0 and non_terminal.name not in {"push", "get"}
                else cls(name="s", symbol=Terminal("s"))
                for i in range(non_terminal.arity)
            ]
            node = cls(str(non_terminal), non_terminal, children=children)

        return node

    def compute(self, **kwargs):
        if isinstance(self.symbol, Terminal) and self.symbol.is_variable:
            res = kwargs[str(self.symbol)]
            return res
        elif isinstance(self.symbol, Terminal):
            res = self.symbol.value
            return res
        elif isinstance(self.symbol, NonTerminal) and self.symbol.is_flow:
            res = self.symbol(*self.children, **kwargs)

            return res
        elif isinstance(self.symbol, NonTerminal):
            res = self.symbol(*[child(**kwargs) for child in self.children])
            return res

    def show(self):
        for pre, _, node in RenderTree(self):
            # Print connector + node's name
            print("%s%s" % (pre, node.name))

    def __call__(self, **kwargs) -> list[float | int | bool]:
        keys = list(kwargs.keys())

        # self.show()

        if len(keys) == 0:
            return self.compute()

        if isinstance(kwargs[keys[0]], Terminal.TYPE):
            return self.compute(**kwargs)

        results = []

        for i in range(len(kwargs[keys[0]])):
            kwargs_i = {k: v[i] for k, v in kwargs.items()}
            results.append(self.compute(**kwargs_i))

        return results
