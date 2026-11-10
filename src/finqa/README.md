# FinQA: Numerical Reasoning over Financial Data

[FinQA](https://arxiv.org/abs/2109.00122v3) (Chen et al., EMNLP 2021) tests numerical reasoning over financial reports. Each question comes with one page of a company's report: the text before a table, the table, and the text after it. Financial experts wrote the questions and annotated each one with a program of arithmetic and table operations whose result is the answer.

This implementation asks the model to write that program, executes it with a port of the [official evaluation script](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/evaluate/evaluate.py), and reports the paper's execution accuracy on the 1,147-question test split.

<!-- Contributors: Automatically Generated -->
Contributed by [@yaokaiy](https://github.com/yaokaiy)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval finqa/finqa --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from finqa import finqa
eval(finqa)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval finqa --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval finqa/finqa --limit 10
uv run inspect eval finqa/finqa --max-connections 10
uv run inspect eval finqa/finqa --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `finqa`

- `fewshot` (int): Number of worked train-split examples in the system prompt (0 to 3). (default: `3`)
<!-- /Parameters: Automatically Generated -->

## Dataset

The test split is read from the official repository at commit [`0f16e28`](https://github.com/czyssrs/FinQA/tree/0f16e2867befa6840783e58be38c9efb9229d742) (`dataset/test.json`, MIT License). The paper's splits are 6,251 train, 883 validation and 1,147 test questions. Text and tables are lowercased and tokenized in the source data, and are passed to the model as they are.

Example prompt (sample `STT/2014/page_54.pdf-2`, after processing by Inspect):

```text
.

 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014
state street corporation | $ 100 | $ 107 | $ 114 | $ 101 | $ 120 | $ 190
s&p 500 index | 100 | 115 | 132 | 135 | 157 | 208
s&p financial index | 100 | 112 | 126 | 104 | 135 | 183
kbw bank index | 100 | 123 | 152 | 117 | 153 | 211

.

Question: what is the roi of an investment in state street corporation from 20011 to 2012?

End your response with a line of the form "PROGRAM: <program>".
```

The gold program is `subtract(101, 114), divide(#0, 114)`, with result `-0.11404`.

The system prompt describes the program language and, by default, adds three worked examples from the train split (`AON/2007/page_185.pdf-2`, `BKNG/2018/page_34.pdf-1`, `AWK/2018/page_150.pdf-3`). None of their report pages is in the test split; two of them come from the same company and year as some test pages. `-T fewshot=0` removes the examples.

## Scoring

The scorer takes the last line of the response that starts with `PROGRAM:` (or the next non-empty line, if nothing follows the colon), after removing Markdown emphasis or code ticks around it, a trailing period and a repeated `PROGRAM:` prefix. If no line starts with `PROGRAM:`, it takes the last non-empty line when that line starts with an operation, since some models end with the program alone. Extra spaces after a comma and a space between an operation name and its parenthesis are removed. The program is then split with the official tokenizer and executed against the sample's table with a port of the official executor, which gives the same result as the original for every program (checked on all 7,398 train and test gold programs and on 300,000 random programs). A sample is correct when the program is valid and its result, rounded to 5 decimal places, equals the gold result exactly (`yes` or `no` for comparison questions). This is the paper's execution accuracy, reported as `accuracy` with `stderr`.

`Score.answer` holds the extracted program. `Score.metadata` records whether the program could be executed (`valid_program`) and its result (`executed_result`), which separates format failures from wrong answers.

Reference points:

| System | Execution accuracy |
| ------ | ------------------ |
| Human experts (paper, 200 sampled questions) | 91.16 |
| FinQANet-Gold, RoBERTa-large, gold facts (paper, Table 2) | 70.00 |
| FinQANet, RoBERTa-large (paper, Table 2) | 61.24 |
| General crowd workers (paper) | 50.68 |
| Always answering the most common result, `yes` | 1.13 |

### Differences from the paper

- **No retriever.** The model gets the whole report page. FinQANet first retrieves supporting facts; the closest paper setting is FinQANet-Gold, which is given the gold facts.
- **Prompting instead of fine-tuning.** The paper's systems are fine-tuned models.
- **Execution accuracy only.** Program accuracy, which checks symbolic equivalence with the gold program, is not implemented. The paper notes that it under-counts questions with more than one correct program.
- **Any number may appear in a program.** The paper's program generator only chooses numbers from the report, constants and step references. Here any number is accepted, so a model can compute an answer in its head and wrap it, as in `add(-0.11404, 0)`. Execution accuracy still requires the right value to 5 decimal places; it does not check how the value was reached.
- **Reading programs from free text.** The paper's systems emit program tokens directly, so the official script never parses text. This implementation does, and its rules (the `PROGRAM:` line, the last-line fallback and the spacing fixes above) are its own. Nested operations such as `divide(add(1, 2), 3)` are not part of the language and stay invalid, as in the official executor; the prompt says so.

### Known limitations

- **The attainable score is below 100%.** On about 107 test questions the gold result disagrees with the human-written answer beyond rounding (60 whose answer is a percentage, 47 others). Some of these are slips in the written answer, such as `ETFC/2012/page_24.pdf-1` (gold `0.64961`, written `64.9%`). Others are errors in the gold program, such as `SNA/2012/page_110.pdf-1`, where `subtract(509, 707), subtract(#0, 707)` gives `-905` for a written answer of `-28%`. A model that answers such a question correctly is scored incorrect. These samples are kept so that results stay comparable with the paper.
- **Percentages.** In this dataset a percentage is almost always a fraction (`0.14464` for 14.5%), and the prompt says so. For 19 of the 651 questions whose answer ends with `%`, the gold result is in percent units instead: 11 gold programs end by multiplying by `const_100`, and others are written that way directly, such as `ADBE/2008/page_74.pdf-1`, `divide(const_100, 4)` for 25%. A model that follows the prompt is scored incorrect on these.
- **Argument separators.** The official tokenizer splits on a comma followed by a space. A comma with no space after it is read as part of a number, so `add(1,234, 1)` adds 1234 and 1, and `subtract(101,114)` is invalid. The prompt asks for `", "`.
- **Table rows with a parenthesis or a comma in their name**, such as `net income ( loss )`, cannot be used in a table operation, because the tokenizer splits on those characters; 437 of the 1,147 test tables have such a row. No gold program uses one, and the prompt tells the model to use arithmetic instead. When two rows share a name, a table operation uses the last one, as the official script does.
- **Incomplete programs.** The official executor accepts a program with no complete step, such as `add(1, 2`, and returns no result. The scorer treats it as invalid; the sample is incorrect either way.

## Evaluation Report

Full test split, 1,147 of 1,147 samples, run on 10 November 2026 with eval version 1-A, Inspect 0.3.277 and the default three-shot prompt. Both models ran locally with Ollama 0.40.14 on an Apple M4 Pro with 48 GB of memory. Deviations from the paper and known limitations are listed under [Scoring](#scoring).

| Model | Ollama tag (digest) | Accuracy | Stderr | Time |
| ----- | ------------------- | -------- | ------ | ---- |
| Qwen2.5 7B Instruct, Q4_K_M | `qwen2.5:7b` (`845dbda0ea48`) | 0.263 | 0.013 | 61 min |
| Qwen2.5 1.5B Instruct, Q4_K_M | `qwen2.5:1.5b` (`65ec06548149`) | 0.135 | 0.010 | 27 min |

How the responses were scored:

| Outcome | qwen2.5:7b | qwen2.5:1.5b |
| ------- | ---------- | ------------ |
| Correct | 302 (26.3%) | 155 (13.5%) |
| Valid program, wrong result | 561 (48.9%) | 535 (46.6%) |
| Program cannot be executed | 283 (24.7%) | 454 (39.6%) |
| No program found | 1 (0.1%) | 3 (0.3%) |

Accuracy by the number of steps in the gold program:

| Gold steps | Questions | qwen2.5:7b | qwen2.5:1.5b |
| ---------- | --------- | ---------- | ------------ |
| 1 | 654 | 0.190 | 0.154 |
| 2 | 409 | 0.430 | 0.130 |
| 3 or more | 84 | 0.024 | 0.012 |

The paper reports only fine-tuned systems (see the reference points under [Scoring](#scoring)), so there is no published number for this setting to compare with directly. Both local models score far below FinQANet (61.24) and the general crowd workers (50.68), and well above the constant baseline (1.13). This report uses small local models only; a larger model would show how close prompting comes to the fine-tuned systems.

Observations from the logs:

- **Almost every response contains a program.** With the fallback to a program on the last line, 1 response (qwen2.5:7b) and 3 (qwen2.5:1.5b) have none.
- **Programs that cannot be executed break the language rather than the parser.** For qwen2.5:7b, 136 of the 283 nest one operation inside another although the prompt forbids it, 51 refer to a step that does not exist, 27 give an operation other than two arguments, and 48 have text outside the steps, such as an arithmetic expression or the program repeated after `PROGRAM:` in the middle of the line. Reading these leniently would change little: flattening the nested programs would make 21 correct, and taking the text after a mid-line `PROGRAM:` 2 more, 2.0% of samples together (1 sample for qwen2.5:1.5b). These are the only failures found that come from the form of the answer rather than its value.
- **Wrong results are reasoning errors, not strict matching.** Of qwen2.5:7b's 557 wrong numeric results, 3 are within 0.1% of the gold result and 7 more within 1%. The most common error on one-step questions is the percentage-change pattern (subtract, then divide) applied to a question that asks for a plain ratio, which is why qwen2.5:7b scores lower on one-step questions than on two-step ones.
- **No sample errored and no response was a refusal**; every response ended normally. Ollama sets no output limit by default, and a few times per run a model repeated itself until the client's 10-minute request timeout, after which Inspect retried the request. Including two transient server errors, 3 samples were retried for qwen2.5:7b and 7 for qwen2.5:1.5b. All 10 were scored incorrect, so the retries did not raise either score; they account for part of the run time.

Commands:

```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=4 ollama serve
uv run --with openai inspect eval finqa/finqa --model ollama/qwen2.5:7b --max-connections 4
uv run --with openai inspect eval finqa/finqa --model ollama/qwen2.5:1.5b --max-connections 4
```

- `--max-connections 4` matches `OLLAMA_NUM_PARALLEL=4`. It changes throughput, not results.
- `OLLAMA_CONTEXT_LENGTH=16384` leaves room for the longest prompt (3,720 tokens) and its response, so no prompt is cut.
- Generation settings are the defaults; for these models Ollama samples with temperature 0.8, top-p 0.9 and top-k 40. Each model was run once, and a rerun gives slightly different numbers.

## Changelog

### [1-A] - 2026-11-08

- Initial implementation.
