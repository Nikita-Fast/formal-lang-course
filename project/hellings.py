from typing import Set, Tuple, Union

import networkx as nx
from pyformlang.cfg import CFG, Variable, Terminal
from project.cfg import cfg_to_wcnf


def hellings(
    graph: nx.Graph,
    cfg: CFG,
) -> Set[Tuple]:
    # grammar_in_file = kwargs.get("grammar_in_file", False)
    # start_symbol = kwargs.get("start_symbol", "S")

    # # transform graph and grammar
    # if grammar_in_file:
    #     cfg = read_grammar_to_str(cfg)
    #
    # if isinstance(cfg, str):
    #     cfg = read_cfg(cfg, start_symbol)
    #
    # if isinstance(graph, str):
    #     graph = get_graph(graph)

    cfg = cfg_to_wcnf(cfg)

    # split productions in 3 groups
    eps_prods = set()  # A -> epsilon
    term_prods = {}  # A -> a
    var_prods = {}  # A -> B C
    for p in cfg.productions:
        if not p.body:
            eps_prods.add(p.head)
        elif len(p.body) == 1:
            t = p.body[0]
            term_prods.setdefault(p.head, set()).add(t)
        elif len(p.body) == 2:
            v1, v2 = p.body
            var_prods.setdefault(p.head, set()).add((v1, v2))

    # prepare result
    result = set()

    for v, u, data in graph.edges(data=True):
        label = data["label"]
        for var in term_prods:
            if Terminal(label) in term_prods[var]:
                result.add((v, var, u))

    for node in graph.nodes:
        for var in eps_prods:
            result.add((node, var, node))

    # helling
    queue = result.copy()
    while len(queue) > 0:
        s, var, f = queue.pop()

        temp = set()

        for triple in result:
            if triple[-1] == s:
                for curr_var in var_prods:
                    if (triple[1], var) in var_prods[curr_var] and (
                        triple[0],
                        curr_var,
                        f,
                    ) not in result:
                        queue.add((triple[0], curr_var, f))
                        temp.add((triple[0], curr_var, f))
            if triple[0] == f:
                for curr_var in var_prods:
                    if (var, triple[1]) in var_prods[curr_var] and (
                        s,
                        curr_var,
                        triple[-1],
                    ) not in result:
                        queue.add((s, curr_var, triple[-1]))
                        temp.add((s, curr_var, triple[-1]))

        result = result.union(temp)

    return result


# def hellings(graph: nx.Graph, cfg: CFG) -> Set[Tuple[int, str, int]]:
#     wcnf = cfg_to_wcnf(cfg)
#
#     r = {
#         (N, v, v)
#         for v in graph.nodes
#         for N in {p.head.value for p in wcnf.productions if not p.body}
#     }.union(
#         {
#             (N, v, u)
#             for (v, u, data) in graph.edges(data=True)
#             for N in {
#                 p.head.value
#                 for p in wcnf.productions
#                 if p.body[0].value == data["label"]
#             }
#         }
#     )
#     m = r.copy()
#
#     var_productions = {p for p in wcnf.productions if len(p.body) == 2}
#     while m != set():
#         # N_i -> v--u
#         N_i, v, u = m.pop()
#         new_triplets = set()
#
#         # пробуем пристроить путь слева к v--u
#         # N_j -> v1--v2
#         # новый путь v1--v2==v--u (в итоге v1--u)
#         for (N_j, v1, v2) in r:
#             if v2 == v:
#                 for N_k in {
#                     p.head.value for p in var_productions if p.body == [N_j, N_i]
#                 }:
#                     if (N_k, v1, u) not in r:
#                         new_triplets.add((N_k, v1, u))
#         m.update(new_triplets)
#         r.update(new_triplets)
#         new_triplets.clear()
#
#         # пробуем пристроить путь справа к v--u
#         # N_j -> v1--v2
#         # новый путь v--u==v1--v2 (в итоге v--v2)
#         for (N_j, v1, v2) in r:
#             if v1 == u:
#                 for N_k in {
#                     p.head.value for p in var_productions if p.body == [N_i, N_j]
#                 }:
#                     if (N_k, v, v2) not in r:
#                         new_triplets.add((N_k, v, v2))
#         m.update(new_triplets)
#         r.update(new_triplets)
#         new_triplets.clear()
#
#     # переупорядочиваем, чтобы соответствовать требованию в домашке
#     return {(v1, N, v2) for (N, v1, v2) in r}
