"""Execution-accuracy scorer for FinQA."""

import re

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from finqa.reference_executor import (
    ALL_OPS,
    INVALID,
    eval_program,
    program_tokenization,
)

PROGRAM_PREFIX = "program:"
# Markdown that models put around the answer line, e.g. "**PROGRAM:** `add(1, 2)`".
MARKDOWN_CHARS = "*_`"
CODE_FENCE = "```"
YES_NO_RESULTS = ("yes", "no")

SPACE_BEFORE_PARENTHESIS = re.compile(r"\b(" + "|".join(ALL_OPS) + r")\s+\(")
SEPARATOR_WITH_EXTRA_SPACE = re.compile(r",\s+")
OPERATION_START = re.compile(r"(" + "|".join(ALL_OPS) + r")\s*\(")


def _strip_decoration(text: str) -> str:
    text = text.strip().strip(MARKDOWN_CHARS).strip()
    return text.removesuffix(".").strip(MARKDOWN_CHARS).strip()


def _remove_prefix(text: str) -> str:
    # Repeated, as in "PROGRAM: PROGRAM: add(1, 2)".
    while text.lower().startswith(PROGRAM_PREFIX):
        text = _strip_decoration(text[len(PROGRAM_PREFIX) :])
    return text


def extract_program(completion: str) -> str | None:
    """Return the program from the last line that starts with `PROGRAM:`.

    Surrounding Markdown emphasis or code ticks, a trailing period and a
    repeated prefix are removed. If nothing follows the colon, the next
    non-empty line is used, skipping code fences. If no line starts with
    `PROGRAM:`, the last non-empty line is used when it starts with an
    operation, as in a response that is only `add(1, 2)`.

    Returns:
        The program text, or None if no program is found.
    """
    lines = completion.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip().lstrip(MARKDOWN_CHARS + "#>- ")
        if not stripped.lower().startswith(PROGRAM_PREFIX):
            continue
        program = _remove_prefix(stripped)
        if not program:
            following = (
                _strip_decoration(line)
                for line in lines[index + 1 :]
                if not line.strip().startswith(CODE_FENCE)
            )
            program = next((text for text in following if text), "")
        return program or None
    last_line = next(
        (_strip_decoration(line) for line in reversed(lines) if line.strip()), ""
    )
    return last_line if OPERATION_START.match(last_line) else None


def normalize_program(program: str) -> str:
    """Remove spacing that the official tokenizer cannot split.

    The tokenizer separates steps and arguments on exactly `", "`.
    `"subtract (1, 2)"` becomes `"subtract(1, 2)"` and `",  "` becomes
    `", "`. A comma with no space after it is left alone: in
    `"subtract(101,114)"` it cannot be told apart from a thousands separator
    such as `"1,234"`.
    """
    program = SPACE_BEFORE_PARENTHESIS.sub(r"\1(", program)
    return SEPARATOR_WITH_EXTRA_SPACE.sub(", ", program)


def result_matches(result: float | str, target: str) -> bool:
    """Compare an executed result with the gold result, as the official script does.

    `target` is `str(exe_ans)`: `"yes"`, `"no"`, or a float already rounded
    to 5 decimal places, which `float()` recovers exactly.
    """
    if target in YES_NO_RESULTS:
        return result == target
    if isinstance(result, str):
        return False
    return result == float(target)


@scorer(metrics=[accuracy(), stderr()])
def finqa_scorer() -> Scorer:
    """Score FinQA by execution accuracy.

    The program from `extract_program()`, after `normalize_program()`, is
    executed against the sample's table
    (`metadata["table"]`). The sample is correct if the program is valid and
    its result equals the gold execution result.
    """

    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_program(state.output.completion)
        if extracted is None:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="The response has no 'PROGRAM:' line and does not end with a program.",
                metadata={"valid_program": False, "executed_result": None},
            )

        program = normalize_program(extracted)
        invalid_flag, result = eval_program(
            program_tokenization(program), state.metadata["table"]
        )
        # The executor reports a program with no complete step, such as
        # "add(1, 2", as valid with no result.
        if invalid_flag or result == INVALID:
            return Score(
                value=INCORRECT,
                answer=program,
                explanation=f"The program is malformed or cannot be executed: {program}",
                metadata={"valid_program": False, "executed_result": None},
            )

        correct = result_matches(result, target.text)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=program,
            explanation=f"Executed result {result!r}; gold result {target.text}.",
            metadata={"valid_program": True, "executed_result": result},
        )

    return score
