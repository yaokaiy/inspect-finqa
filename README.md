# FinQA for Inspect

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of
[FinQA](https://arxiv.org/abs/2109.00122v3) (Chen et al., EMNLP 2021):
numerical reasoning over financial reports, scored by the paper's execution
accuracy on the 1,147-question test split.

See [src/finqa/README.md](src/finqa/README.md) for the task, dataset,
scoring, differences from the paper and results.

## Quickstart

```bash
uv sync
uv run inspect eval finqa/finqa --model <provider/model>
```

The model provider's SDK is not a dependency of this package. For
providers that use the OpenAI client, including `ollama/`, add it with
`uv run --with openai inspect eval ...`.

## Development

```bash
uv run pytest tests/finqa
make check
```

This repository was created from the
[inspect-evals-template](https://github.com/Generality-Labs/inspect-evals-template)
(v1.5.0). [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) and
[MANAGED_FILES.md](MANAGED_FILES.md) describe the template's checks and
the files it keeps in sync.

## License

MIT, Copyright (c) 2026 yaokaiy. The FinQA executor port, the few-shot
records and the template scaffolding come from MIT-licensed projects; see
[NOTICE](NOTICE) and `third-party-licenses/`.
