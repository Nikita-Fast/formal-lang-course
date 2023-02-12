import pytest
from project.interpreter.interpreter import GQLInterpreter
from tests.utils import read_data_from_json


@pytest.mark.parametrize(
    "input_query, expected_output",
    read_data_from_json(
        "test_atomic_functions", lambda d: (d["input_query"], d["expected_output"])
    ),
)
def test_atomic_functions(input_query, expected_output):
    test_interp = GQLInterpreter()
    test_interp.run_query(input_query)
    answer = test_interp.visitor.output_logger

    result = False
    for out in expected_output:
        result = answer == out
        if result:
            break

    assert result


@pytest.mark.parametrize(
    "input_query, expected_output",
    read_data_from_json(
        "test_multiple_functions", lambda d: (d["input_query"], d["expected_output"])
    ),
)
def test_multiple_functions(input_query, expected_output):
    for i in range(4):
        test_interp = GQLInterpreter()
        test_interp.run_query(input_query)
        answer = test_interp.visitor.output_logger

        result = False
        for out in expected_output:
            result = answer == out
            if result:
                break
        if result:
            break

    assert result


@pytest.mark.parametrize(
    "input_, expect_output",
    read_data_from_json(
        "test_errors", lambda d: (d["input_query"], d["expected_output"])
    ),
)
def test_errors(input_, expect_output):
    test_interp = GQLInterpreter()
    test_interp.run_query(input_)
    answer = test_interp.out_log_list

    assert answer == expect_output


@pytest.mark.parametrize(
    "input_, expect_output",
    read_data_from_json(
        "test_multi_single_command", lambda d: (d["input_query"], d["expected_output"])
    ),
)
def test_multi_single_command(input_, expect_output):
    test_interp = GQLInterpreter()
    test_interp.run_query(input_)
    answer = test_interp.visitor.output_logger

    result = False
    for out in expect_output:
        result = answer == out
        if result:
            break
    assert result
