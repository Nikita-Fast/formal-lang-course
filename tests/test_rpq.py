import numpy as np
import pytest
from tests.utils import read_data_from_json, dot_to_graph
from project.automata_tools import create_nfa_from_graph
from project.boolean_matrices import BooleanMatrices


@pytest.mark.parametrize(
    "input_bm, expected_bm, start_states, final_states",
    read_data_from_json(
        "test_bm_init",
        lambda data: (
            BooleanMatrices(create_nfa_from_graph(dot_to_graph(data["graph"]))),
            data["expected_bm"],
            set(data["start_states"]),
            set(data["final_states"]),
        ),
    ),
)
def test_bm_init(input_bm: BooleanMatrices, expected_bm, start_states, final_states):
    is_correct = True
    for symbol, matrix in input_bm.bool_matrices.items():
        expected = expected_bm[symbol]
        is_correct = np.array_equal(matrix.toarray(), expected)
        if not is_correct:
            break
    assert (
        is_correct
        and input_bm.get_start_states() == start_states
        and input_bm.get_final_states() == final_states
    )


@pytest.mark.parametrize(
    "input_bm, expected_bm",
    read_data_from_json(
        "test_transitive_closure",
        lambda data: (
            BooleanMatrices(create_nfa_from_graph(dot_to_graph(data["graph"]))),
            data["matrix"],
        ),
    ),
)
def test_transitive_closure(input_bm: BooleanMatrices, expected_bm):
    tc = input_bm.transitive_closure()
    assert np.array_equal(tc.toarray(), expected_bm)
