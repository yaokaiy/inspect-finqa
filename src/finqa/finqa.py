"""FinQA: A Dataset of Numerical Reasoning over Financial Data

Chen et al., EMNLP 2021
https://arxiv.org/abs/2109.00122v3
Official implementation: https://github.com/czyssrs/FinQA

The model reads a page of an annual report (text, a table, more text) and
answers a question by writing a program in the FinQA DSL. The program is
executed and scored by the paper's execution accuracy.

# three few-shot exemplars (default)
inspect eval finqa/finqa

# zero-shot
inspect eval finqa/finqa -T fewshot=0
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_ai.solver import generate, system_message

from finqa.scorer import finqa_scorer
from utils.metadata import load_version_from_yaml

DATASET_URL = (
    "https://raw.githubusercontent.com/czyssrs/FinQA/"
    "0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json"
)

EVAL_VERSION = load_version_from_yaml("finqa")

# system_message() fills {placeholders} from sample metadata, which holds the
# gold program, so the system prompt and exemplars must not contain braces.
SYSTEM_PROMPT = """
You answer questions about financial reports. Each question comes with one page of a company's annual report: the text before a table, the table, and the text after it. Table cells are separated by " | ", and the first cell of each row is the row name.

Answer by writing a program in the following language. A program is one or more steps separated by ", ". Each step is an operation with two arguments, also separated by ", ":

- add(a, b), subtract(a, b), multiply(a, b), divide(a, b): arithmetic.
- exp(a, b): a raised to the power b.
- greater(a, b): yes if a is greater than b, otherwise no.
- table_sum(row, none), table_average(row, none), table_max(row, none), table_min(row, none): the sum, average, maximum or minimum of the numbers in the table row with that name. Write the row name exactly as it appears in the table. A row whose name contains a parenthesis or a comma cannot be named this way; use arithmetic on its numbers instead.

An argument is one of:

- a number from the report, without units or currency symbols, such as 5829 or 3.5;
- #n, the result of step n, where the first step is #0;
- a constant: const_1 to const_10, const_100, const_1000, const_10000, const_100000, const_1000000, const_10000000, const_1000000000, or const_m1 for -1.

Operations cannot be nested: to use the result of one operation in another, compute it in an earlier step and refer to it with #n.

The value of the program is the value of its last step. In this dataset a percentage is a fraction: a percentage change is subtract(new, old), divide(#0, old), not multiplied by 100.

You may reason step by step first. The last line of your response must be the program, in the form:
PROGRAM: <program>
""".strip()

FEWSHOT_INTRO = "Here are worked examples."

# Train-split records (none of their report pages is in the test split),
# chosen to show a step reference, a constant and a table operation.
FEWSHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "AON/2007/page_185.pdf-2",
        "pre_text": [
            "notes to consolidated financial statements at december 31 , 2007 , future minimum rental payments required under operating leases for continuing operations that have initial or remaining noncancelable lease terms in excess of one year , net of sublease rental income , most of which pertain to real estate leases , are as follows : ( millions ) ."
        ],
        "table": [
            ["2008", "$ 317"],
            ["2009", "275"],
            ["2010", "236"],
            ["2011", "214"],
            ["2012", "191"],
            ["later years", "597"],
            ["total minimum payments required", "$ 1830"],
        ],
        "post_text": ["aon corporation ."],
        "question": "assuming that actual net rent expense will be the same as presented in the table , what would be the growth rate in the net rent expense from 2008 to 2009?",
        "program": "subtract(275, 317), divide(#0, 317)",
    },
    {
        "id": "BKNG/2018/page_34.pdf-1",
        "pre_text": [
            "measurement point december 31 booking holdings nasdaq composite index s&p 500 rdg internet composite ."
        ],
        "table": [
            [
                "measurement pointdecember 31",
                "booking holdings inc .",
                "nasdaqcomposite index",
                "s&p 500index",
                "rdg internetcomposite",
            ],
            ["2013", "100.00", "100.00", "100.00", "100.00"],
            ["2014", "98.09", "114.62", "113.69", "96.39"],
            ["2015", "109.68", "122.81", "115.26", "133.20"],
            ["2016", "126.12", "133.19", "129.05", "140.23"],
            ["2017", "149.50", "172.11", "157.22", "202.15"],
            ["2018", "148.18", "165.84", "150.33", "201.16"],
        ],
        "post_text": ["."],
        "question": "what was the percentage change in booking holding inc . for the five years ended 2018?",
        "program": "subtract(148.18, const_100), divide(#0, const_100)",
    },
    {
        "id": "AWK/2018/page_150.pdf-3",
        "pre_text": [
            "the following table provides the weighted average assumptions used in the black-scholes option-pricing model for grants and the resulting weighted average grant date fair value per share of stock options granted for the years ended december 31: ."
        ],
        "table": [
            ["", "2018", "2017", "2016"],
            ["intrinsic value", "$ 9", "$ 10", "$ 18"],
            ["exercise proceeds", "7", "11", "15"],
            ["income tax benefit realized", "2", "3", "6"],
        ],
        "post_text": [
            "stock units during 2018 , 2017 and 2016 , the company granted rsus to certain employees under the 2007 plan and 2017 omnibus plan , as applicable .",
            "rsus generally vest based on continued employment with the company over periods ranging from one to three years. .",
        ],
        "question": "what was average income tax benefit realized for the three year period?",
        "program": "table_average(income tax benefit realized, none)",
    },
]

DEFAULT_FEWSHOT = len(FEWSHOT_EXAMPLES)

CONTEXT_TEMPLATE = """
{pre_text}

{table}

{post_text}
""".strip()

PROMPT_TEMPLATE = """
{context}

Question: {question}

End your response with a line of the form "PROGRAM: <program>".
""".strip()

FEWSHOT_EXAMPLE_TEMPLATE = """
{context}

Question: {question}
PROGRAM: {program}
""".strip()


def format_table(table: list[list[str]]) -> str:
    """Render table rows as lines of cells separated by " | "."""
    return "\n".join(" | ".join(row) for row in table)


def format_context(
    pre_text: list[str], table: list[list[str]], post_text: list[str]
) -> str:
    """Render a report page: the text before the table, the table, the text after it."""
    return CONTEXT_TEMPLATE.format(
        pre_text=" ".join(pre_text),
        table=format_table(table),
        post_text=" ".join(post_text),
    )


def build_system_prompt(fewshot: int) -> str:
    """Return the system prompt followed by the first `fewshot` exemplars."""
    if not 0 <= fewshot <= len(FEWSHOT_EXAMPLES):
        raise ValueError(
            f"fewshot must be between 0 and {len(FEWSHOT_EXAMPLES)}, got {fewshot}"
        )
    if fewshot == 0:
        return SYSTEM_PROMPT
    examples = [
        FEWSHOT_EXAMPLE_TEMPLATE.format(
            context=format_context(ex["pre_text"], ex["table"], ex["post_text"]),
            question=ex["question"],
            program=ex["program"],
        )
        for ex in FEWSHOT_EXAMPLES[:fewshot]
    ]
    return "\n\n".join([SYSTEM_PROMPT, FEWSHOT_INTRO, *examples])


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a FinQA record to a Sample.

    The target is `str(exe_ans)`: `"yes"`, `"no"`, or a float rounded to 5
    decimal places. The table is kept in metadata because the scorer
    executes the model's program against it.
    """
    qa = record["qa"]
    program = qa["program"]
    return Sample(
        id=record["id"],
        input=PROMPT_TEMPLATE.format(
            context=format_context(
                record["pre_text"], record["table"], record["post_text"]
            ),
            question=qa["question"],
        ),
        target=str(qa["exe_ans"]),
        metadata={
            "table": record["table"],
            "program": program,
            "answer": qa["answer"],
            "filename": record["filename"],
            "num_steps": program.count("("),
            "uses_table_op": "table_" in program,
        },
    )


def get_finqa_dataset() -> Dataset:
    """Load the FinQA test split (1,147 samples) from the pinned official repository."""
    return json_dataset(DATASET_URL, sample_fields=record_to_sample, name="finqa")


@task
def finqa(fewshot: int = DEFAULT_FEWSHOT) -> Task:
    """FinQA numerical reasoning over financial reports, scored by execution accuracy.

    Args:
        fewshot: Number of worked train-split examples in the system prompt
            (0 to 3).
    """
    system_prompt = build_system_prompt(fewshot)
    return Task(
        dataset=get_finqa_dataset(),
        solver=[system_message(system_prompt), generate()],
        scorer=finqa_scorer(),
        version=EVAL_VERSION,
    )
