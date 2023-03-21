from typing import Set, Dict, Tuple

import networkx as nx
import numpy as np
from pyformlang.cfg import CFG, Variable, Production, Terminal

from project.cfg import cfg_to_wcnf
from scipy.sparse import csr_matrix


# def matrix(graph: nx.Graph, cfg: CFG) -> Set[Tuple]:
#     # grammar_in_file = kwargs.get("grammar_in_file", False)
#     # start_symbol = kwargs.get("start_symbol", "S")
#     #
#     # # transform graph and grammar
#     # if grammar_in_file:
#     #     cfg = read_grammar_to_str(cfg)
#     #
#     # if isinstance(cfg, str):
#     #     cfg = read_cfg(cfg, start_symbol)
#     #
#     # if isinstance(graph, str):
#     #     graph = get_graph(graph)
#
#     cfg = cfg_to_wcnf(cfg)
#
#     # split productions in 3 groups
#     eps_prods = set()  # A -> epsilon
#     term_prods = {}  # A -> a
#     var_prods = {}  # A -> B C
#     for p in cfg.productions:
#         if not p.body:
#             eps_prods.add(p.head)
#         elif len(p.body) == 1:
#             t = p.body[0]
#             term_prods.setdefault(p.head, set()).add(t)
#         elif len(p.body) == 2:
#             v1, v2 = p.body
#             var_prods.setdefault(p.head, set()).add((v1, v2))
#
#     # prepare adjacency matrix
#     nodes_num = graph.number_of_nodes()
#     nodes = {vertex: i for i, vertex in enumerate(graph.nodes)}
#     nodes_reversed = {i: vertex for i, vertex in enumerate(graph.nodes)}
#     matrices = {
#         v: csr_matrix((nodes_num, nodes_num), dtype=bool) for v in cfg.variables
#     }
#
#     # A -> terminal
#     for v, u, data in graph.edges(data=True):
#         label = data["label"]
#         i = nodes[v]
#         j = nodes[u]
#         for var in term_prods:
#             if Terminal(label) in term_prods[var]:
#                 matrices[var][i, j] = True
#
#     # A -> espilon loops
#     for var in eps_prods:
#         for i in range(nodes_num):
#             matrices[var][i, i] = True
#
#     # A -> B C
#     changed = True
#     while changed:
#         changed = False
#         for head in var_prods:
#             for body_b, body_c in var_prods[head]:
#                 old_nnz = matrices[head].nnz
#                 matrices[head] += matrices[body_b] @ matrices[body_c]
#                 new_nnz = matrices[head].nnz
#                 changed = old_nnz != new_nnz
#
#     return {
#         (nodes_reversed[v], var, nodes_reversed[u])
#         for var, matrix in matrices.items()
#         for v, u in zip(*matrix.nonzero())
#     }


def matrix(graph: nx.Graph, cfg: CFG) -> Set[Tuple[int, Variable, int]]:
    wcnf = cfg_to_wcnf(cfg)
    n = len(graph.nodes)
    bool_matrices = {v: csr_matrix((n, n), dtype=bool) for v in wcnf.variables}

    terminal_productions: Set[Production] = {
        p for p in wcnf.productions if len(p.body) == 1
    }
    # словарь node_number нужен, чтобы алгоритм правильно обрабатывал графы, где номера узлов начинаются не с нуля
    node_number = {node: i for (i, node) in enumerate(graph.nodes)}
    for (i, j, data) in graph.edges(data=True):
        production_heads: Set[Variable] = {
            p.head for p in terminal_productions if p.body[0].value == data["label"]
        }
        for v in production_heads:
            bool_matrices[v][node_number[i], node_number[j]] = True

    # добавляем петли за счет эпсилон продукций
    epsilon_productions_heads = {p.head for p in wcnf.productions if not p.body}
    for v in epsilon_productions_heads:
        for i in range(n):
            bool_matrices[v][i, i] = True

    variables_productions = {p for p in wcnf.productions if len(p.body) == 2}
    is_changed = True
    while is_changed:
        is_changed = False
        # A_i -> A_j A_k
        for p in variables_productions:
            old_nnz = bool_matrices[p.head].nnz
            bool_matrices[p.head] += bool_matrices[p.body[0]] @ bool_matrices[p.body[1]]
            is_changed |= old_nnz != bool_matrices[p.head].nnz

    # redefine node_number dict
    node_number = {number: node for (node, number) in node_number.items()}

    return {
        (node_number[i], v, node_number[j])
        for v, mat in bool_matrices.items()
        for (i, j) in zip(*mat.nonzero())
    }
