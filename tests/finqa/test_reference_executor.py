"""Tests for the port of the official FinQA executor."""

import pytest

from finqa import get_finqa_dataset, result_matches
from finqa.reference_executor import (
    INVALID,
    eval_program,
    process_row,
    program_tokenization,
    str_to_num,
)

TABLE = [
    ["", "2018", "2017", "2016"],
    ["intrinsic value", "$ 9", "$ 10", "$ 18"],
    ["income tax benefit realized", "2", "3", "6"],
    ["provision", "-32 ( 32 )", "-35 ( 35 )", "-14 ( 14 )"],
    ["notes", "n/a", "1", "2"],
    ["net income ( loss )", "1", "2", "3"],
    ["name only"],
]


def execute(program: str) -> tuple[int, float | str]:
    return eval_program(program_tokenization(program), TABLE)


def test_program_tokenization() -> None:
    assert program_tokenization("subtract(275, 317), divide(#0, 317)") == [
        "subtract(",
        "275",
        "317",
        ")",
        "divide(",
        "#0",
        "317",
        ")",
        "EOF",
    ]
    assert program_tokenization("table_average(income tax benefit realized, none)") == [
        "table_average(",
        "income tax benefit realized",
        "none",
        ")",
        "EOF",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("94", 94.0),
        ("1,234", 1234.0),
        ("5%", 0.05),
        ("const_100", 100.0),
        ("const_m1", -1.0),
        ("$5", INVALID),
        ("abc", INVALID),
    ],
)
def test_str_to_num(text: str, expected: float | str) -> None:
    assert str_to_num(text) == expected


def test_process_row() -> None:
    assert process_row(["$ 9", "-32 ( 32 )", "1,000"]) == [9.0, -32.0, 1000.0]
    assert process_row(["1", "n/a"]) == INVALID


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("add(1, 2)", 3.0),
        ("subtract(5829, 5735)", 94.0),
        ("multiply(2, 7)", 14.0),
        ("divide(1, 3)", 0.33333),
        ("exp(2, 3)", 8.0),
        ("greater(23542, 3485)", "yes"),
        ("greater(1, 2)", "no"),
        ("subtract(275, 317), divide(#0, 317)", -0.13249),
        ("subtract(148.18, const_100), divide(#0, const_100)", 0.4818),
        ("multiply(5, const_m1)", -5.0),
        ("add(5%, 1)", 1.05),
        ("add(1,234, 1)", 1235.0),
        ("table_average(income tax benefit realized, none)", 3.66667),
        ("table_max(intrinsic value, none)", 18.0),
        ("table_min(provision, none)", -35.0),
        ("table_sum(provision, none)", -81.0),
        ("table_sum(provision, none), divide(#0, 3)", -27.0),
        # "#n" selects no row, so the previous table step's row is reused.
        ("table_sum(income tax benefit realized, none), table_max(#0, none)", 6.0),
        # As in the original, yes/no results can be added (concatenated) or
        # compared as strings, and a later numeric step makes the program valid.
        ("greater(2, 1), add(#0, #0), subtract(101, 114), divide(#2, 114)", -0.11404),
        ("greater(2, 1), greater(1, 2), greater(#0, #1)", "yes"),
    ],
)
def test_eval_program_valid(program: str, expected: float | str) -> None:
    assert execute(program) == (0, expected)


@pytest.mark.parametrize(
    "program",
    [
        "foo(1, 2)",
        "add(1)",
        "divide(1, 0)",
        "add(#1, 2)",
        "add($5, 1)",
        "exp(-8, 0.5)",
        "table_max(unknown row, none)",
        "table_max(notes, none)",
        "table_max(#0, none)",
        "greater(1, 2), add(#0, 1)",
        "greater(2, 1), add(#0, #0)",
        "add(const_abc, 1)",
        "subtract(101,114)",
        "subtract (101, 114)",
        "table_sum(net income ( loss ), none)",
    ],
)
def test_eval_program_invalid(program: str) -> None:
    invalid_flag, _ = execute(program)
    assert invalid_flag == 1


def test_table_operation_uses_the_last_row_with_a_name() -> None:
    table = [["revenue", "1", "2"], ["revenue", "10", "20"]]
    assert eval_program(program_tokenization("table_sum(revenue, none)"), table) == (
        0,
        30.0,
    )


def test_table_sum_of_a_row_without_values_is_zero() -> None:
    invalid_flag, result = execute("table_sum(name only, none)")
    assert (invalid_flag, result) == (0, 0)
    assert result_matches(result, "0.0")


def test_eval_program_without_a_complete_step_has_no_result() -> None:
    # The official executor does not flag these; the scorer treats them as invalid.
    assert execute("add(1, 2") == (0, INVALID)
    assert execute("") == (0, INVALID)


@pytest.mark.dataset_download
def test_gold_programs_reproduce_gold_results() -> None:
    dataset = get_finqa_dataset()
    assert len(dataset) == 1147

    mismatches = []
    for sample in dataset:
        assert sample.metadata is not None
        invalid_flag, result = eval_program(
            program_tokenization(sample.metadata["program"]), sample.metadata["table"]
        )
        assert isinstance(sample.target, str)
        if invalid_flag or not result_matches(result, sample.target):
            mismatches.append(sample.id)

    assert mismatches == []
