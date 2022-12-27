import pytest
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    DeterministicFiniteAutomaton,
    State,
)
from scipy.sparse import dok_matrix

from project.boolean_matrices import BooleanMatrices


@pytest.fixture
def get_input_nfa():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(
        [
            (0, "a", 0),
            (0, "a", 1),
            (0, "b", 2),
            (1, "c", 3),
            (2, "a", 3),
            (3, "b", 1),
            (3, "c", 0),
        ]
    )

    nfa.add_start_state(State(0))
    nfa.add_final_state(State(3))
    return nfa


def test_labels(get_input_nfa):
    bm = BooleanMatrices(get_input_nfa)
    actual_labels = bm.bool_matrices.keys()
    expected_labels = get_input_nfa.symbols

    assert actual_labels == expected_labels


@pytest.mark.parametrize(
    "label, transitions",
    [("a", [(0, 0), (0, 1), (2, 3)]), ("b", [(3, 1)]), ("c", [(1, 3), (3, 0)])],
)
def test_create_boolean_matrices(get_input_nfa, label, transitions):
    bm = BooleanMatrices(get_input_nfa)
    assert all(bm.bool_matrices[label][transition] for transition in transitions)


def test_to_automaton(get_input_nfa):
    expected = get_input_nfa
    bm = BooleanMatrices(get_input_nfa)
    actual = bm.to_automaton()

    assert actual == expected


def test_transitive_closure(get_input_nfa):
    bm = BooleanMatrices(get_input_nfa)
    transitive_closure = bm.transitive_closure()
    assert transitive_closure.sum() == transitive_closure.size


def test_intersect():
    fa1 = NondeterministicFiniteAutomaton()
    fa1.add_transitions(
        [(0, "e", 0), (0, "a", 1), (0, "d", 1), (1, "b", 1), (1, "c", 2), (2, "e", 0)]
    )
    fa1.add_start_state(State(0))
    fa1.add_final_state(State(1))
    fa1.add_final_state(State(2))

    bm1 = BooleanMatrices(fa1)

    fa2 = NondeterministicFiniteAutomaton()
    fa2.add_transitions([(0, "a", 1), (0, "a", 0), (1, "b", 1), (1, "e", 2)])
    fa2.add_start_state(State(0))
    fa2.add_final_state(State(1))

    bm2 = BooleanMatrices(fa2)

    expected_fa = DeterministicFiniteAutomaton()
    expected_fa.add_transitions([(0, "a", 3), (3, "b", 3)])
    expected_fa.add_start_state(State(0))
    expected_fa.add_final_state(State(3))

    # expected_fa = NondeterministicFiniteAutomaton()
    # expected_fa.add_transitions([(0, "a", 3), (0, 'a', 4), (4, "b", 4)])
    # expected_fa.add_start_state(State(0))
    # expected_fa.add_final_state(State(4))

    actual_fa = bm1.intersect(bm2).to_automaton()

    assert actual_fa.is_equivalent_to(expected_fa)
