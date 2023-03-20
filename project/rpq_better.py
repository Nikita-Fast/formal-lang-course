from __future__ import annotations

from typing import List
import networkx as nx

from project.boolean_matrices import BooleanMatrices
from project.automata_tools import (
    regex_to_minimal_dfa,
    graph_to_nfa,
    create_nfa_from_graph,
)

from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from scipy import sparse
from pyformlang.regular_expression import Regex

__all__ = ["rpq", "bfs_rpq"]


def rpq(
    graph: nx.MultiDiGraph,
    regex: Regex,
    start_vertices: set = None,
    final_vertices: set = None,
) -> set:
    """
    Get set of reachable pairs of graph vertices
    Parameters
    ----------
    graph
        Input Graph
    regex
        Input regular expression
    start_vertices
        Start vertices for graph
    final_vertices
        Final vertices for graph
    Returns
    -------
    set
        Set of reachable pairs of graph vertices
    """
    regex_automaton_matrix = BooleanMatrices.from_automaton(regex_to_minimal_dfa(regex))
    graph_automaton_matrix = BooleanMatrices.from_automaton(
        create_nfa_from_graph(graph, start_vertices, final_vertices)
    )
    intersected_automaton = graph_automaton_matrix.intersect(regex_automaton_matrix)
    tc_matrix = intersected_automaton.get_transitive_closure()
    res = set()

    for s_from, s_to in zip(*tc_matrix.nonzero()):
        if (
            s_from in intersected_automaton.start_states
            and s_to in intersected_automaton.final_states
        ):
            res.add(
                (
                    s_from // regex_automaton_matrix.num_states,
                    s_to // regex_automaton_matrix.num_states,
                )
            )

    return res


def _build_adj_empty_matrix(g: nx.MultiDiGraph) -> sparse.csr_matrix:
    """
    Build empty adjacency matrix for passed graph
    Parameters
    ----------
    g: nx.MultiDiGraph
        Input graph
    Returns
    -------
    adj_m: sparse.csr_matrix
        Adjacency matrix
    """

    return sparse.csr_matrix(
        (len(g.nodes), len(g.nodes)),
        dtype=bool,
    )


def _build_direct_sum(
    r: DeterministicFiniteAutomaton, g: nx.MultiDiGraph
) -> dict[sparse.csr_matrix]:
    """
    Build direct sum of boolean matrix decomposition dfa and graph
    Parameters
    ----------
    r: DeterministicFiniteAutomaton
        Input dfa
    g: nx.MultiDiGraph
        Input graph
    Returns
    -------
    d: dict[sparse.csr_matrix]
        Result of direct sum
    """

    d = {}

    r_matrix = BooleanMatrices.from_automaton(r)
    g_matrix = BooleanMatrices.from_automaton(graph_to_nfa(g))

    r_labels = set(r_matrix.bool_matrices.keys())
    g_labels = set(g_matrix.bool_matrices.keys())
    labels = r_labels.intersection(g_labels)

    r_size = r_matrix.num_states
    g_size = g_matrix.num_states
    for label in labels:
        left_up_matrix = r_matrix.bool_matrices[label]
        right_up_matrix = sparse.csr_matrix(
            (r_size, g_size),
            dtype=bool,
        )
        left_down_matrix = sparse.csr_matrix(
            (g_size, r_size),
            dtype=bool,
        )
        right_down_matrix = g_matrix.bool_matrices[label]
        d[label] = sparse.vstack(
            [
                sparse.hstack(
                    [left_up_matrix, right_up_matrix], dtype=bool, format="csr"
                ),
                sparse.hstack(
                    [left_down_matrix, right_down_matrix], dtype=bool, format="csr"
                ),
            ],
            dtype=bool,
            format="csr",
        )

    return d


def _create_masks(
    r: DeterministicFiniteAutomaton, g: nx.MultiDiGraph
) -> sparse.csr_matrix:
    """
    Create M matrix
    Parameters
    ----------
    r: DeterministicFiniteAutomaton
        Input dfa
    g: nx.MultiDiGraph
        Input graph
    Returns
    -------
    m: sparse.csr_matrix
    """

    r_size = len(r.states)
    g_size = len(g.nodes)

    id = sparse.csr_matrix((r_size, r_size), dtype=bool)
    for i, state in enumerate(r.states):
        if state in r.start_states:
            id[i, i] = 1
    front = sparse.csr_matrix(
        (r_size, g_size),
        dtype=bool,
    )
    m = sparse.hstack([id, front], dtype=bool, format="csr")

    return m


def _set_start_verts(m: sparse.csr_matrix, v_src: set) -> sparse.csr_matrix:
    """
    Add start vertices to right part of M matrix
    Parameters
    ----------
    m: sparse.csr_matrix
        M matrix
    v_src: set
        Start vertices set
    Returns
    -------
    m_new: sparse.csr_matrix
        Updated M matrix
    """
    r_size = m.get_shape()[0]
    for i in range(r_size):
        for start_v in v_src:
            m[i, r_size + start_v] = 1

    return m


def _get_graph_labels(g: nx.MultiDiGraph) -> set[str]:
    """
    Extract graph labels from edges
    Parameters
    ----------
    g: nx.MultiDiGraph
        Input graph
    Returns
    -------
    labels: set[str]
        Extracted labels
    """

    labels = set()
    for node_from, node_to in g.edges():
        labels.add(g.get_edge_data(node_from, node_to)[0]["label"])

    return labels


def _extract_left_submatrix(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Extract left part of M matrix --- identity matrix
    Parameters
    ----------
    m: sparse.csr_matrix
        M matrix
    Returns
    -------
    m: sparse.csr_matrix
        Identity matrix
    """

    extr_size = m.shape[0]
    return m[:, :extr_size]


def _extract_right_submatrix(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Extract right part of M matrix --- front
    Parameters
    ----------
    m: sparse.csr_matrix
        M matrix
    Returns
    -------
    m: sparse.csr_matrix
        Front
    """

    extr_size = m.shape[0]
    return m[:, extr_size:]


def _transform_front_part(front_part: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Transform another front to right form of M matrix. Left submatrix is identity matrix
    Parameters
    ----------
    front_part: sparse.csr_matrix
        Input matrix
    Returns
    -------
    m: sparse.csr_matrix
        Transformed matrix
    """

    t = _extract_left_submatrix(front_part)
    m_new = sparse.csr_matrix(
        front_part.shape,
        dtype=bool,
    )
    nnz_row, nnz_col = t.nonzero()
    for i in range(len(nnz_col)):
        row = front_part.getrow(nnz_row[i])
        m_new[nnz_col[i], :] = row
    return m_new


def _reduce_to_vector(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Reduce matrix to vector
    Parameters
    ----------
    m: sparse.csr_matrix
        Input matrix
    Returns
    -------
    v: sparse.csr_matrix
        Reduced vector
    """
    shape = m.shape
    v = sparse.csr_matrix(
        (1, shape[1]),
        dtype=bool,
    )
    for row in range(shape[0]):
        v += m.getrow(row)

    return v


def _bfs_based_rpq(
    r: DeterministicFiniteAutomaton, g: nx.MultiDiGraph, v_src: set, separated=False
) -> List[sparse.csr_matrix]:

    """
    Parameters
    ----------
    r: DeterministicFiniteAutomaton
        Input dfa
    g: nx.MultiDiGraph
        Input graph
    v_src: set
        Start vertices set
    separated: bool
        Process for each start vertex or for set of start vertices
    Returns
    -------
    visited: List[sparse.csr_matrix]
        List of matrices in two parts. In the first part of the square matrix with the state of the automata,
        in the second part of the matrix with ones in the columns,
        the vertices of which can be found from the state of the automaton of this row
    """
    d = _build_direct_sum(r, g)
    if separated:
        init_m = [
            _set_start_verts(_create_masks(r, g), {start_vertex})
            for start_vertex in v_src
        ]
    else:
        init_m = _create_masks(r, g)
        init_m = [_set_start_verts(init_m, v_src)]

    labels = r.symbols.intersection(_get_graph_labels(g))

    old_nnz = []
    for front_matrix in init_m:
        old_nnz.append(0)
    is_continue = True
    visited = []
    for front_matrix in init_m:
        visited.append(front_matrix.copy())
    not_updated_matrix = set()  # matrix numbers in which nnz counts do not change

    while is_continue:
        is_continue = False
        for num_front_matrix in range(len(init_m)):
            # not processing unchanged matrix
            if num_front_matrix in not_updated_matrix:
                continue
            new_front = sparse.csr_matrix(
                init_m[num_front_matrix].shape,
                dtype=bool,
            )
            for label in labels:
                # multiply D matrix and current front
                temp = init_m[num_front_matrix].dot(d[label])
                # transform to right form
                new_front += _transform_front_part(temp)
                visited[num_front_matrix] += new_front  # update visited vertices

            # change front to new
            init_m[num_front_matrix] = new_front

            # count nonzero values
            if old_nnz[num_front_matrix] == visited[num_front_matrix].nnz:
                not_updated_matrix.add(num_front_matrix)
            else:
                old_nnz[num_front_matrix] = visited[num_front_matrix].nnz
                is_continue = True
    return visited


def bfs_rpq(
    graph: nx.MultiDiGraph,
    regex: Regex,
    start_vertices: set = None,
    final_vertices: set = None,
    separated: bool = False,
) -> set[tuple[int, frozenset] | frozenset]:
    """
    Get set of reachable pairs of graph vertices
    Parameters
    ----------
    graph
        Input Graph
    regex
        Input regular expression
    start_vertices
        Start vertices for graph
    final_vertices
        Final vertices for graph
    separated
        Process for each start vertex or for set of start vertices
    Returns
    -------
    set
        Set of reachable pairs of graph vertices
    """

    if start_vertices is None:
        start_vertices = set()
        for node in graph.nodes:
            start_vertices.add(node)

    if final_vertices is None:
        final_vertices = set()
        for node in graph.nodes:
            final_vertices.add(node)

    regex_automaton = regex_to_minimal_dfa(regex)
    rpq_result = _bfs_based_rpq(
        regex_automaton, graph, v_src=start_vertices, separated=separated
    )

    res = set()
    if separated:
        for s_v in start_vertices:
            visited_per_start = _extract_right_submatrix(rpq_result[s_v])
            temp = list()
            for i, automaton_state in enumerate(regex_automaton.states):
                if not (automaton_state in regex_automaton.final_states):
                    continue
                row = visited_per_start.getrow(i)
                for vertex in row.indices:
                    if vertex in final_vertices:
                        temp.append(vertex)
            res.add((s_v, frozenset(temp)))
    else:
        reachable_vertices = list()
        visited_per_start = _extract_right_submatrix(rpq_result[0])
        for i, automaton_state in enumerate(regex_automaton.states):
            if not (automaton_state in regex_automaton.final_states):
                continue
            row = visited_per_start.getrow(i)

            for vertex in row.indices:
                if vertex in final_vertices:
                    reachable_vertices.append(vertex)
        res.add(frozenset(reachable_vertices))

    return res
