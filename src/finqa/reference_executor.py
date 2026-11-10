"""FinQA program tokenizer and executor.

Adapted from the official FinQA evaluation script
https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/evaluate/evaluate.py
(Copyright (c) 2021 Zhiyu Chen, MIT License).

The control flow follows the original step by step, so a program is valid and
evaluates to the same result here as in the paper's execution accuracy. The
differences are type annotations, `isinstance` checks where a value can only
be a number or `INVALID` (the original compares with `"n/a"`), and
`except Exception` in place of a bare `except`.
"""

from typing import Any

ALL_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
]
ARITHMETIC_OPS = ("add", "subtract", "multiply", "divide", "exp", "greater")
INVALID = "n/a"


def str_to_num(text: str) -> float | str:
    """Parse a program argument or table cell; return `INVALID` if it is not a number.

    Raises:
        ValueError: For a `const_` token that is not a number, such as
            `const_abc`. `eval_program()` treats this as an invalid program.
    """
    text = text.replace(",", "")
    try:
        num: float | str = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = INVALID
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = INVALID
    return num


def process_row(row_in: list[str]) -> list[float] | str:
    """Parse the value cells of a table row; return `INVALID` if any cell is not a number.

    Raises:
        ValueError: As `str_to_num()` does, for a cell such as `const_abc`.
    """
    row_out: list[float] = []
    invalid_flag = 0
    for raw_cell in row_in:
        cell = raw_cell.replace("$", "").strip()
        cell = cell.split("(")[0].strip()
        num = str_to_num(cell)
        if isinstance(num, str):
            invalid_flag = 1
            break
        row_out.append(num)
    if invalid_flag:
        return INVALID
    return row_out


def _resolve_argument(arg: str, res_dict: dict[int, Any]) -> Any:
    if "#" in arg:
        return res_dict[int(arg.replace("#", ""))]
    return str_to_num(arg)


def eval_program(program: list[str], table: list[list[str]]) -> tuple[int, float | str]:
    """Execute a tokenized program against a table.

    Args:
        program: Tokens from `program_tokenization()`, ending with `EOF`.
        table: The sample's table; the first cell of each row is its name.

    Returns:
        `(invalid_flag, result)`. `invalid_flag` is 1 when the program is
        malformed or cannot be executed, and `result` is then meaningless.
        Otherwise `result` is the last step's value rounded to 5 decimal
        places, or `"yes"`/`"no"` for `greater`.
    """
    invalid_flag = 0
    # Any, as in the original: a yes/no result may feed a later step, where
    # `add` concatenates two of them and `greater` compares them as strings.
    this_res: Any = INVALID
    # A table step whose first argument is "#n" selects no row and reuses the
    # previous table step's row, as in the original script.
    num_row: list[float] | str | None = None

    try:
        program = program[:-1]  # remove EOF
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, INVALID
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, INVALID

        steps = "|".join(program).split(")")[:-1]
        res_dict: dict[int, Any] = {}

        for ind, raw_step in enumerate(steps):
            step = raw_step.strip()
            if step.count("(") > 1:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in ARITHMETIC_OPS:
                value1 = _resolve_argument(arg1, res_dict)
                if value1 == INVALID:
                    invalid_flag = 1
                    break
                value2 = _resolve_argument(arg2, res_dict)
                if value2 == INVALID:
                    invalid_flag = 1
                    break

                if op == "add":
                    this_res = value1 + value2
                elif op == "subtract":
                    this_res = value1 - value2
                elif op == "multiply":
                    this_res = value1 * value2
                elif op == "divide":
                    this_res = value1 / value2
                elif op == "exp":
                    this_res = value1**value2
                elif op == "greater":
                    this_res = "yes" if value1 > value2 else "no"
                res_dict[ind] = this_res

            elif "table" in op:
                table_dict = {row[0]: row[1:] for row in table}
                if "#" in arg1:
                    # Raises for a step that does not exist, making the program invalid.
                    res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    num_row = process_row(table_dict[arg1])
                if num_row is None or isinstance(num_row, str):
                    invalid_flag = 1
                    break

                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                res_dict[ind] = this_res

        if this_res not in ("yes", "no", INVALID):
            this_res = round(this_res, 5)

    except Exception:
        invalid_flag = 1

    return invalid_flag, this_res


def program_tokenization(original_program: str) -> list[str]:
    """Split a program such as `subtract(5829, 5735), divide(#0, 5735)` into tokens.

    Steps are separated by `", "`, and so are the two arguments of a step.
    The returned list ends with `EOF`, which `eval_program()` expects.
    """
    program: list[str] = []
    for tok in original_program.split(", "):
        cur_tok = ""
        for c in tok:
            if c == ")":
                if cur_tok != "":
                    program.append(cur_tok)
                    cur_tok = ""
            cur_tok += c
            if c in ["(", ")"]:
                program.append(cur_tok)
                cur_tok = ""
        if cur_tok != "":
            program.append(cur_tok)
    program.append("EOF")
    return program
