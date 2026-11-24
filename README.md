# FinQA for Inspect AI

[![Inspect Evals Register](https://img.shields.io/badge/Inspect_Evals-registered-1f6feb)](https://ukgovernmentbeis.github.io/inspect_evals/evals/finqa/)
[![inspect-evals lint](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ArcadiaImpact/inspect-evals-actions/register-lint/badges/finqa/lint.json)](https://ukgovernmentbeis.github.io/inspect_evals/register-lint.html)
[![Checks](https://github.com/yaokaiy/inspect-finqa/actions/workflows/checks.yml/badge.svg)](https://github.com/yaokaiy/inspect-finqa/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An implementation of [FinQA](https://arxiv.org/abs/2109.00122v3) for
[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), the LLM
evaluation framework from the UK AI Security Institute. The eval is listed in
the [Inspect Evals Register](https://ukgovernmentbeis.github.io/inspect_evals/evals/finqa/).

> Chen et al. *FinQA: A Dataset of Numerical Reasoning over Financial Data.*
> EMNLP 2021. [arXiv:2109.00122](https://arxiv.org/abs/2109.00122v3) ·
> [original code and data](https://github.com/czyssrs/FinQA)

## What it measures

Each of the 1,147 test questions comes with one page of a company's financial
report: the text before a table, the table, and the text after it. The model
answers by writing a program in FinQA's language of arithmetic and table
operations, such as `subtract(101, 114), divide(#0, 114)`. The program is
executed, and the sample is correct when its result, rounded to 5 decimal
places, equals the gold result; on comparison questions the `yes` or `no`
result must equal the gold one. This is the paper's execution accuracy.

- **Official executor.** The scorer reads the program from the response and
  runs it with a port of FinQA's evaluation script. The port gives the same
  result as the original on all 7,398 train and test gold programs and on
  300,000 random programs.
- **Pinned data.** The test split is read from the official repository at
  commit [`0f16e28`](https://github.com/czyssrs/FinQA/tree/0f16e2867befa6840783e58be38c9efb9229d742).
- **Failure diagnosis.** Each score records the extracted program, whether it
  could be executed, and its result, so format failures can be told apart from
  wrong answers.

The [eval README](src/finqa/README.md) covers the prompt, how programs are read
from free text, the differences from the paper and the known limitations of the
dataset.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/). CI tests Python 3.11, 3.12 and 3.13;
the commands below use 3.12.

```bash
git clone https://github.com/yaokaiy/inspect-finqa
cd inspect-finqa
uv sync --python 3.12
uv run inspect eval finqa/finqa --model mockllm/model --limit 3  # downloads the data, needs no API key
```

The model provider's SDK is not a dependency of this package. Both commands
below add the OpenAI Python package with `--with openai`. The first needs
`OPENAI_API_KEY`; the second needs a local Ollama server and
`ollama pull qwen2.5:7b`.

```bash
uv run --with openai inspect eval finqa/finqa --model openai/gpt-5-nano
uv run --with openai inspect eval finqa/finqa --model ollama/qwen2.5:7b
```

The prompt includes three worked examples from the train split by default;
`-T fewshot=0` removes them. `uv run inspect view` opens the logs.

The Register pins commit
[`66be93f`](https://github.com/yaokaiy/inspect-finqa/tree/66be93f24467f50f916df6a2149079214a83ba70).
To run exactly that version, run `git checkout 66be93f` right after
`cd inspect-finqa`, then continue with `uv sync --python 3.12`.

## Results

Full test split (1,147 samples), task version 1-A, Inspect 0.3.277, default
three-shot prompt, one run per model with Ollama 0.40.14:

| Model                          | Accuracy | Stderr |
| ------------------------------ | -------- | ------ |
| `ollama/qwen2.5:7b` (Q4_K_M)   | 0.263    | 0.013  |
| `ollama/qwen2.5:1.5b` (Q4_K_M) | 0.135    | 0.010  |

The FinQA paper reports only fine-tuned systems, so it has no baseline for
prompted models. Its execution accuracy figures: 61.24% for FinQANet
(RoBERTa-large), 50.68% for general crowd workers and 91.16% for financial
experts on 200 sampled questions.

The [evaluation report](src/finqa/README.md#evaluation-report) has the outcome
breakdown, accuracy by program length, error analysis and the exact commands.
Both logs were submitted with the registration
([inspect_evals#2851](https://github.com/UKGovernmentBEIS/inspect_evals/pull/2851)).

## Development

```bash
uv run pytest tests/finqa
make check
```

Changes that can affect scores bump the task version, following
[TASK_VERSIONING.md](TASK_VERSIONING.md), and are listed in the eval README's
[changelog](src/finqa/README.md#changelog). Report bugs and questions in
[issues](https://github.com/yaokaiy/inspect-finqa/issues).

This repository was created from the
[inspect-evals-template](https://github.com/Generality-Labs/inspect-evals-template)
(v1.5.0). [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) and
[MANAGED_FILES.md](MANAGED_FILES.md) describe the template's checks and the
files it keeps in sync.

## Citation

If you use this eval, please cite the FinQA paper:

```bibtex
@inproceedings{chen-etal-2021-finqa,
    title = "{F}in{QA}: A Dataset of Numerical Reasoning over Financial Data",
    author = "Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.300/",
    doi = "10.18653/v1/2021.emnlp-main.300",
    pages = "3697--3711"
}
```

## License

MIT, Copyright (c) 2026 yaokaiy. The FinQA executor port, the few-shot records
and the template scaffolding come from MIT-licensed projects; see
[NOTICE](NOTICE) and `third-party-licenses/`.
