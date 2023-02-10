import pytest
import sys
import os
from project.parser.parser_invoker import write_to_dot
from tests.utils import read_data_from_json

dir_path = os.path.dirname(os.path.realpath(__file__))
if sys.platform.startswith("win"):
    path = dir_path + "\\parsed.dot"
else:
    path = dir_path + "/parsed.dot"


@pytest.mark.parametrize(
    "line, expected",
    read_data_from_json("test_write_to_dot", lambda d: (d["line"], d["expected"])),
)
def test_write_to_dot(line: str, expected: str):
    status = write_to_dot(line, path)
    obtained = open(path, "r")

    assert (expected == obtained.read()) and status
    os.remove(path)


@pytest.mark.parametrize(
    "line, expected_status",
    read_data_from_json(
        "test_write_to_dot_with_wrong_text", lambda d: (d["line"], d["expected_status"])
    ),
)
def test_write_to_dot_with_wrong_text(line: str, expected_status: bool):
    status = write_to_dot(line, path)

    assert status == expected_status
