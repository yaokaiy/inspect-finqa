"""Tests for the FinQA task, dataset conversion and prompts."""

import importlib
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pytest_inspect_evals.assertions import assert_eval_success, assert_task_structure

from finqa import (
    DEFAULT_FEWSHOT,
    FEWSHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_system_prompt,
    finqa,
    format_table,
    record_to_sample,
)

# Test record STT/2014/page_54.pdf-2 from the pinned dataset, all fields used.
RECORD: dict[str, Any] = {
    "id": "STT/2014/page_54.pdf-2",
    "filename": "STT/2014/page_54.pdf",
    "pre_text": ["."],
    "table": [
        ["", "2009", "2010", "2011", "2012", "2013", "2014"],
        [
            "state street corporation",
            "$ 100",
            "$ 107",
            "$ 114",
            "$ 101",
            "$ 120",
            "$ 190",
        ],
        ["s&p 500 index", "100", "115", "132", "135", "157", "208"],
        ["s&p financial index", "100", "112", "126", "104", "135", "183"],
        ["kbw bank index", "100", "123", "152", "117", "153", "211"],
    ],
    "post_text": ["."],
    "qa": {
        "question": "what is the roi of an investment in state street corporation from 20011 to 2012?",
        "program": "subtract(101, 114), divide(#0, 114)",
        "exe_ans": -0.11404,
        "answer": "-11.4%",
    },
}
GOLD_PROGRAM = "subtract(101, 114), divide(#0, 114)"


def test_record_to_sample() -> None:
    sample = record_to_sample(RECORD)

    assert sample.id == "STT/2014/page_54.pdf-2"
    assert sample.target == "-0.11404"
    assert isinstance(sample.input, str)
    assert (
        "state street corporation | $ 100 | $ 107 | $ 114 | $ 101 | $ 120 | $ 190"
        in sample.input
    )
    assert (
        "Question: what is the roi of an investment in state street corporation"
        in sample.input
    )
    assert sample.input.endswith(
        'End your response with a line of the form "PROGRAM: <program>".'
    )
    assert GOLD_PROGRAM not in sample.input
    assert sample.metadata == {
        "table": RECORD["table"],
        "program": GOLD_PROGRAM,
        "answer": "-11.4%",
        "filename": "STT/2014/page_54.pdf",
        "num_steps": 2,
        "uses_table_op": False,
    }


def test_record_to_sample_yes_no_target() -> None:
    record = {
        **RECORD,
        "qa": {**RECORD["qa"], "program": "greater(101, 114)", "exe_ans": "no"},
    }
    assert record_to_sample(record).target == "no"


def test_format_table() -> None:
    assert format_table([["", "2018"], ["revenue", "$ 9"]]) == " | 2018\nrevenue | $ 9"


def test_build_system_prompt() -> None:
    assert build_system_prompt(0) == SYSTEM_PROMPT
    assert DEFAULT_FEWSHOT == len(FEWSHOT_EXAMPLES) == 3

    one_shot = build_system_prompt(1)
    assert f"PROGRAM: {FEWSHOT_EXAMPLES[0]['program']}" in one_shot
    assert FEWSHOT_EXAMPLES[1]["program"] not in one_shot

    three_shot = build_system_prompt(3)
    for example in FEWSHOT_EXAMPLES:
        assert f"PROGRAM: {example['program']}" in three_shot


@pytest.mark.parametrize("fewshot", range(len(FEWSHOT_EXAMPLES) + 1))
def test_system_prompt_has_no_template_placeholders(fewshot: int) -> None:
    # system_message() would fill placeholders from sample metadata, which
    # holds the gold program.
    prompt = build_system_prompt(fewshot)
    assert "{" not in prompt
    assert "}" not in prompt


@pytest.mark.parametrize("fewshot", [-1, 4])
def test_invalid_fewshot_raises_before_loading_the_dataset(
    fewshot: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_loaded() -> None:
        raise AssertionError("the dataset was loaded before fewshot was checked")

    # The package exports the task function `finqa`, which shadows the module
    # name, so patch the module object itself.
    task_module = importlib.import_module("finqa.finqa")
    monkeypatch.setattr(task_module, "get_finqa_dataset", fail_if_loaded)
    with pytest.raises(ValueError, match="fewshot must be between 0 and 3"):
        finqa(fewshot=fewshot)


@solver
def gold_program_solver() -> Solver:
    """Answer with the sample's gold program, to check plumbing and scoring."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(
            "mockllm/model", f"PROGRAM: {state.metadata['program']}"
        )
        return state

    return solve


@pytest.mark.dataset_download
def test_finqa_task_structure() -> None:
    task = finqa()
    assert_task_structure(task)
    assert len(task.dataset) == 1147
    assert task.version == "1-A"


@pytest.mark.dataset_download
def test_finqa_end_to_end_with_mock(tmp_path: Path) -> None:
    [log] = eval(tasks=finqa(), model="mockllm/model", limit=2, log_dir=str(tmp_path))
    assert_eval_success(log, expected_metric="accuracy")


@pytest.mark.dataset_download
def test_finqa_runs_by_registry_name(tmp_path: Path) -> None:
    # `inspect eval finqa/finqa` needs the distribution name to match the
    # package name, so that Inspect namespaces the task as "finqa/finqa".
    [log] = eval(
        tasks="finqa/finqa", model="mockllm/model", limit=1, log_dir=str(tmp_path)
    )
    assert_eval_success(log, expected_metric="accuracy")
    assert log.eval.task == "finqa/finqa"


@pytest.mark.dataset_download
def test_finqa_correct_answer_and_no_gold_leak(tmp_path: Path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/model",
                f"ROI is (101 - 114) / 114.\nPROGRAM: {GOLD_PROGRAM}",
            )
        ],
    )
    [log] = eval(
        tasks=finqa(),
        model=model,
        sample_id="STT/2014/page_54.pdf-2",
        log_dir=str(tmp_path),
    )

    assert_eval_success(log, expected_metric="accuracy")
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
    assert log.samples is not None
    [sample] = log.samples
    system_text = sample.messages[0].text
    assert sample.messages[0].role == "system"
    assert GOLD_PROGRAM not in system_text


@pytest.mark.dataset_download
def test_finqa_gold_solver_scores_one(tmp_path: Path) -> None:
    [log] = eval(
        tasks=finqa(),
        solver=gold_program_solver(),
        model="mockllm/model",
        log_dir=str(tmp_path),
    )
    assert_eval_success(log, expected_metric="accuracy", expected_sample_count=1147)
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
