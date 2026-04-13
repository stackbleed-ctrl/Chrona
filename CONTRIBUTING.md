# Contributing to Chrona

Thank you for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/chrona-ai/chrona.git
cd chrona
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

```bash
ruff check src/      # lint
black src/ tests/    # format
```

## Pull Request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Lint passes (`ruff check src/`)
- [ ] Docstrings added for new public functions
- [ ] CHANGELOG.md updated

## Areas we'd love help with

- Real Mamba SSM integration (`mamba-ssm` backend)
- Additional dataset loaders (M4, ETT, Weather benchmarks)
- JS/TypeScript SDK (`sdk/js/`)
- Streaming WebSocket endpoint
- Fine-tuning examples

## Questions?

Open a GitHub Discussion or join Discord.
