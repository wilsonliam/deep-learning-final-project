# Deep Learning Final Project

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you want the notebook stack too:

```bash
python -m pip install -e ".[dev,notebooks]"
```

## Common commands

Run the test suite:

```bash
pytest
```

Run the pseudobulk CLI smoke test:

```bash
pseudobulk-tahoe100 --smoke
```
