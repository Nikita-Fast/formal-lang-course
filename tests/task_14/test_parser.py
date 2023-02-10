import pytest
from project.parser.parser_invoker import *
from tests.utils import read_data_from_json


@pytest.mark.parametrize(
    "input_string, expected",
    read_data_from_json(
        "test_parse_to_string",
        lambda d: (
            d["input_string"],
            d["expected"],
        ),
    ),
)
def test_parse_to_string(input_string, expected):
    s = parse_to_string(input_string)
    assert s == expected


@pytest.mark.parametrize(
    "input_string, expected",
    read_data_from_json(
        "test_is_in_grammar",
        lambda d: (
            d["input_string"],
            d["expected"],
        ),
    ),
)
def test_is_in_grammar(input_string, expected):
    s = is_in_grammar(input_string)
    assert s == expected
