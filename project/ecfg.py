from typing import Set, Dict

from pyformlang.cfg import Variable, CFG
from pyformlang.regular_expression import Regex


class ECFG:
    def __init__(
        self,
        variables: Set[Variable],
        start: Variable,
        productions: Dict[Variable, Regex],
    ):
        self.variables = variables
        self.start = start
        self.productions = productions

    def to_text(self) -> str:
        return "\n".join(
            str(p) + " -> " + str(self.productions[p]) for p in self.productions
        )

    @classmethod
    def from_cfg(cls, cfg: CFG):
        variables = set(cfg.variables)
        start_symbol = (
            cfg.start_symbol if cfg.start_symbol is not None else Variable("S")
        )
        variables.add(start_symbol)

        productions: dict[Variable, Regex] = {}
        for p in cfg.productions:
            body = Regex(" ".join(o.value for o in p.body) if len(p.body) > 0 else "$")
            if p.head in productions:
                productions[p.head] = productions[p.head].union(body)
            else:
                productions[p.head] = body

        return cls(variables, start_symbol, productions)

    @classmethod
    def from_text(cls, text: str, start_symbol: str = Variable("S")) -> "ECFG":
        variables = set()
        productions = dict()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            production_objects = line.split("->")
            if len(production_objects) != 2:
                raise Exception("There should be only one production per line.")

            head_text, body_text = production_objects
            head = Variable(head_text.strip())

            if head in variables:
                raise Exception(
                    "There should be only one production for each variable."
                )

            variables.add(head)
            body = Regex(body_text.strip())
            productions[head] = body

        return cls(
            variables=variables,
            start=Variable(start_symbol),
            productions=productions,
        )
