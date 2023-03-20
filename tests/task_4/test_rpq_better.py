import networkx as nx
import pytest
from numpy import interp
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    DeterministicFiniteAutomaton,
)
from scipy import sparse
from pyformlang.regular_expression import Regex

from project.boolean_matrices import BooleanMatrices
from project.graph_tools import create_two_cycle_graph
from project.rpq_better import rpq


@pytest.fixture
def graph():
    return create_two_cycle_graph(3, 2, ("x", "y"))


@pytest.fixture
def empty_graph():
    return nx.empty_graph(create_using=nx.MultiDiGraph)


@pytest.fixture
def acyclic_graph():
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [(0, 1, {"label": "x"}), (1, 2, {"label": "y"}), (2, 3, {"label": "y"})]
    )
    return graph


def test_transitive_closure():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(
        [
            (0, "a", 1),
            (0, "a", 2),
            (2, "d", 3),
            (1, "c", 1),
            (1, "b", 2),
            (3, "d", 0),
        ]
    )
    matrix_automaton = BooleanMatrices.from_automaton(nfa)
    tc_matrix = matrix_automaton.get_transitive_closure()

    assert tc_matrix.sum() == tc_matrix.size


def test_intersection():
    first_nfa = NondeterministicFiniteAutomaton()
    first_nfa.add_transitions(
        [(0, "a", 1), (0, "c", 1), (0, "c", 0), (1, "b", 1), (1, "c", 2), (2, "d", 0)]
    )
    first_nfa.add_start_state(State(0))
    first_nfa.add_final_state(State(0))
    first_nfa.add_final_state(State(1))
    first_nfa.add_final_state(State(2))

    second_nfa = NondeterministicFiniteAutomaton()
    second_nfa.add_transitions([(0, "a", 1), (0, "a", 0), (1, "b", 1), (1, "e", 2)])
    second_nfa.add_start_state(State(0))
    second_nfa.add_final_state(State(1))

    expected_nfa = first_nfa.get_intersection(second_nfa)

    first_matrix_automaton = BooleanMatrices.from_automaton(first_nfa)
    second_matrix_automaton = BooleanMatrices.from_automaton(second_nfa)
    intersected_fa = first_matrix_automaton.intersect(second_matrix_automaton)

    actual_nfa = intersected_fa.to_automaton()

    assert actual_nfa.is_equivalent_to(expected_nfa)


@pytest.mark.parametrize(
    "regex,start_nodes,final_nodes,expected_rpq",
    [
        ("x* | y", {0}, {1, 2, 3, 4}, {(0, 1), (0, 2), (0, 3), (0, 4)}),
        ("x* | y", {4}, {4, 5}, {(4, 5)}),
        ("x x", {0, 1, 2, 3}, {0, 1, 2, 3}, {(0, 2), (1, 3), (2, 0), (3, 1)}),
        ("y", {0}, {0, 1, 2, 3}, set()),
        ("y*", {0}, {5, 4}, {(0, 5), (0, 4)}),
        ("x* | z", {0}, None, {(0, 1), (0, 2), (0, 3), (0, 0)}),
        ("z* | w", None, None, set()),
    ],
)
def test_two_cycles(graph, regex, start_nodes, final_nodes, expected_rpq):
    regex = Regex(regex)
    actual_rpq = rpq(graph, regex, start_nodes, final_nodes)

    assert actual_rpq == expected_rpq


@pytest.mark.parametrize(
    "regex,start_nodes,final_nodes,expected_rpq",
    [
        ("x* | y", None, None, set()),
        ("x* | y", None, None, set()),
        ("x x", None, None, set()),
        ("y", None, None, set()),
        ("", None, None, set()),
    ],
)
def test_empty(empty_graph, regex, start_nodes, final_nodes, expected_rpq):
    regex = Regex(regex)
    actual_rpq = rpq(empty_graph, regex, start_nodes, final_nodes)

    assert actual_rpq == expected_rpq


@pytest.mark.parametrize(
    "regex,start_nodes,final_nodes,expected_rpq",
    [
        ("x* | y", {0}, {3}, set()),
        ("x* | y", {1}, {2, 3}, {(1, 2)}),
        ("x y y", None, None, {(0, 3)}),
    ],
)
def test_acyclic(acyclic_graph, regex, start_nodes, final_nodes, expected_rpq):
    regex = Regex(regex)
    actual_rpq = rpq(acyclic_graph, regex, start_nodes, final_nodes)

    assert actual_rpq == expected_rpq
