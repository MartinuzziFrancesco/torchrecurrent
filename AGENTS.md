# AGENTS.md

Guidance for AI agents working in the **torchrecurrent** repository.

## Project Snapshot

- Package: `torchrecurrent`
- Purpose: PyTorch-compatible recurrent neural network cells and layers from
  research literature, primarily for academic research.
- Python: `>=3.9`; CI covers Python 3.9-3.14 on Linux, Windows, and macOS.
- Runtime dependency policy: `torch` is the only runtime dependency.
- Style: Black, line length 92.

## Commands

Run the narrowest useful command first, then broaden when the change touches
shared behavior.

```bash
pip install -e .[test]
pytest
coverage run -m pytest
black .
flake8
pre-commit run --all-files
```

- `pytest` runs the test suite.
- `coverage run -m pytest` matches the CI test command.
- `black .` formats with the configured 92-character line length.
- `flake8` excludes `docs/`, `benchmarks/`, and `tests/`.
- `pre-commit run --all-files` runs Black and Ruff fixes before committing.

## Repository Map

- `torchrecurrent/base.py`: abstract base classes for cells and layers.
- `torchrecurrent/cells/`: each `*_cell.py` defines both a cell and its layer.
- `torchrecurrent/benchmarks/`: packaged task generators.
- `benchmarks/`: standalone training scripts and saved runs, not packaged.
- `tests/test_cells.py`: per-cell shape, dtype, and state checks.
- `tests/test_layers.py`: per-layer stacking and `batch_first` checks.
- `docs/`: Sphinx docs and the model catalog in `docs/models.rst`.

There is no `layers/` directory. Keep cell and layer implementations together in
the relevant `torchrecurrent/cells/<name>_cell.py` file.

## Architecture Conventions

- `BaseSingleRecurrentCell` uses one hidden state `h`.
- `BaseDoubleRecurrentCell` uses LSTM-style `(h, c)` state.
- `BaseSingleRecurrentLayer` and `BaseDoubleRecurrentLayer` iterate cell stacks
  over the time dimension.
- Weights are concatenated per gate into `weight_ih` and `weight_hh`, then split
  with `.chunk(n, 0)` in `forward`.
- Cells support input shaped `(input_size,)` or `(N, input_size)` via the base
  `_preprocess_*` helpers.
- Bias controls are separate: `bias` for input-side terms and `recurrent_bias`
  for recurrent-side terms.
- Initializers are configurable through `kernel_init`, `recurrent_kernel_init`,
  `bias_init`, and `recurrent_bias_init`; defaults are `xavier_uniform_` for
  weights and `zeros_` for biases.

## Adding A Model

1. Create `torchrecurrent/cells/<name>_cell.py`.
2. Define `<Name>Cell` from the matching single-state or double-state base cell.
3. Define `<Name>` from the matching layer base and call
   `self.initialize_cells(<Name>Cell, **kwargs)`.
4. Use `torchrecurrent/cells/mgu_cell.py` as the implementation and docstring
   template.
5. Re-export both classes from `torchrecurrent/cells/__init__.py` and
   `torchrecurrent/__init__.py`, including each `__all__`.
6. Add the cell to `CELL_CASES` in `tests/test_cells.py`.
7. Add the layer to `tests/test_layers.py`.
8. Add docs under `docs/api/`, generated autosummary coverage, and
   `docs/models.rst`.

## Code Style

- Format Python with Black before finishing changes that touch code.
- Keep comments sparse. Add comments only when they explain non-obvious math,
  paper-specific behavior, numerical stability choices, or API compatibility.
- Do not add comments that merely restate the code.
- Match the existing Google/NumPy-style docstrings with a math block, arXiv link,
  Args, Inputs, Outputs, and Variables sections.
- Keep tests table-driven and update the relevant parametrized cases when adding
  or renaming public models.

## Boundaries

### Always Do

- Preserve native PyTorch-style interfaces that mirror `torch.nn.RNN` and
  `torch.nn.RNNCell` where applicable.
- Keep the three export sites synchronized:
  `torchrecurrent/cells/__init__.py`, `torchrecurrent/__init__.py`, and each
  `__all__`.
- Respect third-party licenses. `NASCell` is an Apache-2.0 reimplementation in an
  MIT-licensed project.

### Ask First

- Adding, removing, or changing runtime dependencies. Do not add dependencies
  just to simplify an implementation.
- Exporting or otherwise wiring up `rhn_cell.py`; it exists but is intentionally
  not part of the public API.
- Large rewrites, API breaks, renamed public classes, or changes to package
  metadata and release configuration.
- Broad documentation regeneration if it would create large generated diffs.

### Never Do

- Do not create a separate `layers/` package.
- Do not commit or edit saved artifacts under `benchmarks/.../runs/`.
- Do not add unnecessary comments.
- Do not skip tests silently; report any tests that could not be run.
- Do not introduce non-`torch` runtime dependencies without explicit approval.

## Done Criteria

Before finishing, check the work against the scope of the change:

- Code is formatted with Black when Python files changed.
- Relevant tests were run, or the reason they were not run is stated.
- New or renamed public models are exported from both package entry points.
- Tests and docs are updated when public behavior changes.
- The final response summarizes changed files, verification, and any remaining
  risk or follow-up.
