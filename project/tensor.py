from typing import Set, Tuple

import networkx as nx
from pyformlang.cfg import CFG, Variable

from project.automata_tools import create_nfa_from_graph
from project.boolean_matrices import BooleanMatrices
from scipy.sparse import dok_matrix
from project.rsm import RSM
from project.ecfg import ECFG


def tensor(graph: nx.Graph, cfg: CFG) -> Set[Tuple[int, Variable, int]]:
    # матрицы смежности
    graph_nfa = create_nfa_from_graph(graph)
    graph_bm = BooleanMatrices.from_automaton(graph_nfa, dok_matrix)

    ecfg = ECFG.from_cfg(cfg)
    rsm = RSM.from_ecfg(ecfg).minimize()
    rsm_bm = BooleanMatrices.from_rsm(rsm, dok_matrix)

    rsm_variables = set(rsm.boxes.keys())

    # Добавим петли для эпсилон порождающих нетерминалов
    for v in rsm_variables:
        if v in cfg.get_nullable_symbols():
            for i in range(graph_bm.number_of_states):
                graph_bm.bool_matrices[v][i, i] = True

    matrix_changed = True
    old_nnz = 0
    while matrix_changed:
        tc = rsm_bm.intersect(graph_bm).transitive_closure()

        new_nnz = tc.nnz
        if new_nnz == old_nnz:
            break
        old_nnz = new_nnz

        for (i, j) in zip(*tc.nonzero()):
            rsm_i = i // graph_bm.number_of_states
            rsm_j = j // graph_bm.number_of_states
            s1 = rsm_bm.index_to_state[rsm_i]
            s2 = rsm_bm.index_to_state[rsm_j]
            if (
                rsm_i in rsm_bm.start_state_indexes
                and rsm_j in rsm_bm.final_state_indexes
            ):
                graph_i = i % graph_bm.number_of_states
                graph_j = j % graph_bm.number_of_states

                v, _ = s1.value

                if v not in graph_bm.bool_matrices.keys():
                    # todo надо ли уметь выбирать type_of_matrix?
                    graph_bm.bool_matrices[v] = dok_matrix(
                        (graph_bm.number_of_states, graph_bm.number_of_states),
                        dtype=bool,
                    )
                graph_bm.bool_matrices[v][graph_i, graph_j] = True

    return {
        (graph_bm.index_to_state[i], v, graph_bm.index_to_state[j])
        for v, bm in graph_bm.bool_matrices.items()
        for (i, j) in zip(*bm.nonzero())
    }
