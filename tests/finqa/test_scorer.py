"""Tests for FinQA program extraction and the execution-accuracy scorer."""

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target
from inspect_ai.solver import TaskState

from finqa import extract_program, finqa_scorer, normalize_program, result_matches

TABLE = [
    ["", "2018", "2017", "2016"],
    ["income tax benefit realized", "2", "3", "6"],
]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (
            "PROGRAM: subtract(101, 114), divide(#0, 114)",
            "subtract(101, 114), divide(#0, 114)",
        ),
        ("The change is 101 - 114.\nPROGRAM: subtract(101, 114)", "subtract(101, 114)"),
        ("program: add(1, 2)", "add(1, 2)"),
        ("**PROGRAM:** add(1, 2)", "add(1, 2)"),
        ("PROGRAM: `add(1, 2)`", "add(1, 2)"),
        ("PROGRAM: add(1, 2).", "add(1, 2)"),
        ("- PROGRAM: add(1, 2)", "add(1, 2)"),
        ("PROGRAM: add(1, 2)\nPROGRAM: add(3, 4)", "add(3, 4)"),
        ("PROGRAM: add(1, 2)\nThat is the answer.", "add(1, 2)"),
        ("PROGRAM: table_average(u.k ., none)", "table_average(u.k ., none)"),
        ("PROGRAM:\nsubtract(101, 114)", "subtract(101, 114)"),
        ("PROGRAM:\n```\nsubtract(101, 114)\n```", "subtract(101, 114)"),
        ("**PROGRAM:**\n\n```text\nadd(1, 2)\n```", "add(1, 2)"),
        ("PROGRAM: PROGRAM: add(1, 2)", "add(1, 2)"),
        ("**PROGRAM:** PROGRAM: add(1, 2)", "add(1, 2)"),
        (
            "subtract(35.3, 33.2), divide(#0, 33.2)",
            "subtract(35.3, 33.2), divide(#0, 33.2)",
        ),
        ("The change is 101 - 114.\n\n`subtract(101, 114)`\n", "subtract(101, 114)"),
    ],
)
def test_extract_program(completion: str, expected: str) -> None:
    assert extract_program(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "The answer is 94.",
        "PROGRAM:",
        "PROGRAM: **",
        "PROGRAM:\n```\n```",
        "",
        "add(1, 2)\nThe answer is 3.",
        "addition(1, 2)",
    ],
)
def test_extract_program_missing(completion: str) -> None:
    assert extract_program(completion) is None


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("subtract (101, 114)", "subtract(101, 114)"),
        ("table_sum  (revenue, none)", "table_sum(revenue, none)"),
        ("add(1,  2),  divide(#0, 3)", "add(1, 2), divide(#0, 3)"),
        ("add(1, 2),\ndivide(#0, 3)", "add(1, 2), divide(#0, 3)"),
        # Unchanged: a comma with no space may be a thousands separator.
        ("subtract(101,114)", "subtract(101,114)"),
        ("add(1,234, 1)", "add(1,234, 1)"),
        ("subtract(101, 114), divide(#0, 114)", "subtract(101, 114), divide(#0, 114)"),
    ],
)
def test_normalize_program(program: str, expected: str) -> None:
    assert normalize_program(program) == expected


@pytest.mark.parametrize(
    ("result", "target", "expected"),
    [
        (94.0, "94.0", True),
        (-0.11404, "-0.11404", True),
        (-0.11403, "-0.11404", False),
        (11.404, "-0.11404", False),
        ("yes", "yes", True),
        ("no", "yes", False),
        ("yes", "1.0", False),
        (1.0, "yes", False),
    ],
)
def test_result_matches(result: float | str, target: str, expected: bool) -> None:
    assert result_matches(result, target) is expected


async def score_completion(completion: str, target: str) -> Score:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="test",
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput.from_content("mockllm/model", completion),
        metadata={"table": TABLE},
    )
    score = await finqa_scorer()(state, Target(target))
    assert score is not None
    return score


def test_finqa_scorer_is_a_scorer() -> None:
    assert isinstance(finqa_scorer(), Scorer)


async def test_scorer_correct_program() -> None:
    score = await score_completion(
        "Average of 2, 3 and 6.\nPROGRAM: table_average(income tax benefit realized, none)",
        "3.66667",
    )
    assert score.value == CORRECT
    assert score.answer == "table_average(income tax benefit realized, none)"
    assert score.metadata == {"valid_program": True, "executed_result": 3.66667}


async def test_scorer_wrong_result() -> None:
    score = await score_completion(
        "PROGRAM: table_max(income tax benefit realized, none)", "3.66667"
    )
    assert score.value == INCORRECT
    assert score.metadata == {"valid_program": True, "executed_result": 6.0}


async def test_scorer_unknown_row_is_invalid() -> None:
    score = await score_completion(
        "PROGRAM: table_average(income tax, none)", "3.66667"
    )
    assert score.value == INCORRECT
    assert score.metadata == {"valid_program": False, "executed_result": None}


async def test_scorer_incomplete_program_is_invalid() -> None:
    score = await score_completion("PROGRAM: add(1, 2", "3.0")
    assert score.value == INCORRECT
    assert score.metadata == {"valid_program": False, "executed_result": None}


async def test_scorer_missing_program_line() -> None:
    score = await score_completion("The answer is 3.66667.", "3.66667")
    assert score.value == INCORRECT
    assert score.answer is None
    assert score.metadata == {"valid_program": False, "executed_result": None}


async def test_scorer_percentage_as_fraction() -> None:
    fraction = await score_completion(
        "PROGRAM: subtract(101, 114), divide(#0, 114)", "-0.11404"
    )
    assert fraction.value == CORRECT
    percent = await score_completion(
        "PROGRAM: subtract(101, 114), divide(#0, 114), multiply(#1, const_100)",
        "-0.11404",
    )
    assert percent.value == INCORRECT


async def test_scorer_normalizes_spacing() -> None:
    score = await score_completion("PROGRAM: subtract (101,  114)", "-13.0")
    assert score.value == CORRECT
    assert score.answer == "subtract(101, 114)"


async def test_scorer_comma_without_space_is_invalid() -> None:
    score = await score_completion("PROGRAM: subtract(101,114)", "-13.0")
    assert score.value == INCORRECT
    assert score.metadata == {"valid_program": False, "executed_result": None}


async def test_scorer_program_on_the_next_line() -> None:
    score = await score_completion("PROGRAM:\n```\nsubtract(101, 114)\n```", "-13.0")
    assert score.value == CORRECT


async def test_scorer_program_without_prefix() -> None:
    score = await score_completion("subtract(101, 114)", "-13.0")
    assert score.value == CORRECT
    assert score.answer == "subtract(101, 114)"


async def test_scorer_yes_no_target() -> None:
    assert (
        await score_completion("PROGRAM: greater(23542, 3485)", "yes")
    ).value == CORRECT
    assert (
        await score_completion("PROGRAM: greater(3485, 23542)", "yes")
    ).value == INCORRECT
