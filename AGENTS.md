# AGENTS.md

Guidance for AI agents working in the **torchrecurrent** repository.

## What this project is

`torchrecurrent` is a PyTorch-compatible collection of recurrent neural network
**cells** and **layers** drawn from the research literature. Every model exposes
a native-PyTorch-style interface (mirroring `torch.nn.RNN`/`RNNCell`) while adding
extra knobs for initialization and customization. It is published on
[PyPI](https://pypi.org/project/torchrecurrent/) and conda-forge, and is intended
primarily for academic research.

- Package: `torchrecurrent` (version in `pyproject.toml`)
- Python: `>=3.9` (CI runs 3.9–3.14 on Linux/Windows/macOS)
- Single runtime dependency: `torch`
- Companion projects: [RecurrentLayers.jl](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl)
  (Flux), [LuxRecurrentLayers.jl](https://github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl) (Lux)

## Project structure

Generated with `tree -I '__pycache__|*.egg-info|.venv|.git|.pytest_cache|runs|generated'`
(the `runs/` experiment artifacts and `generated/` autosummary stubs are collapsed):

```
.
├── benchmarks                          # standalone training scripts + saved runs/ (not packaged)
│   ├── adding_problem
│   │   └── adding_problem.py
│   └── copy_memory
│       └── copy_memory.py
├── docs                                # Sphinx documentation
│   ├── api
│   │   ├── benchmarks.rst
│   │   ├── cells.rst
│   │   ├── index.rst
│   │   └── layers.rst
│   ├── _static
│   │   ├── favicon.ico
│   │   ├── logo-long2.png
│   │   └── logo.png
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── models.rst                      # catalog of published models
│   └── requirements.txt
├── tests
│   ├── test_cells.py                   # per-cell shape/dtype/state checks
│   └── test_layers.py                  # per-layer stacking/batch_first checks
├── torchrecurrent                      # the package
│   ├── benchmarks
│   │   ├── adding.py                    # adding_problem task generator
│   │   ├── copymemory.py               # copy_memory task generator
│   │   └── __init__.py
│   ├── cells                           # each file defines BOTH a Cell and its layer
│   │   ├── antisymmetricrnn_cell.py
│   │   ├── atr_cell.py
│   │   ├── br_cell.py
│   │   ├── cfn_cell.py
│   │   ├── cornn_cell.py
│   │   ├── fastrnn_cell.py
│   │   ├── indrnn_cell.py
│   │   ├── __init__.py                 # re-exports every cell + its layer
│   │   ├── janet_cell.py
│   │   ├── lem_cell.py
│   │   ├── lightru_cell.py
│   │   ├── ligru_cell.py
│   │   ├── mgu_cell.py
│   │   ├── multiplicativelstm_cell.py
│   │   ├── mut_cell.py                 # MUT1 / MUT2 / MUT3
│   │   ├── nas_cell.py
│   │   ├── originallstm_cell.py
│   │   ├── peepholelstm_cell.py
│   │   ├── ran_cell.py
│   │   ├── rhn_cell.py                 # present but NOT exported (commented out)
│   │   ├── scrn_cell.py
│   │   ├── sgrn_cell.py
│   │   ├── star_cell.py
│   │   ├── ugrnn_cell.py
│   │   ├── unicornn_cell.py
│   │   └── wmclstm_cell.py
│   ├── base.py                         # abstract base classes (see "Architecture")
│   └── __init__.py                     # top-level public API (alphabetized re-exports)
├── AGENTS.md
├── LICENSE                             # MIT (NASCell re-impl carries Apache-2.0)
├── MANIFEST.in
├── pyproject.toml                      # build, deps, black config, test extras
└── README.md
```

Two things worth internalizing:

- **There is no `layers/` directory.** Each `*_cell.py` file defines *both* the
  cell (e.g. `MGUCell`) and its multi-layer wrapper (e.g. `MGU`). The
  `cells/__init__.py` and top-level `__init__.py` re-export both.
- `torchrecurrent/benchmarks/` (packaged task generators) is distinct from the
  top-level `benchmarks/` (standalone training scripts and saved run artifacts).

## Architecture

All models inherit from base classes in `torchrecurrent/base.py`:

- `BaseRecurrentCell` — common cell machinery: input/state validation, zero-state
  init, parameter/buffer registration (`_register_tensors`,
  `_default_register_tensors`), and `init_weights()` which dispatches on parameter
  name (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`).
  - `BaseSingleRecurrentCell` — single hidden state `h`; `uses_double_state()` → `False`.
  - `BaseDoubleRecurrentCell` — LSTM-style `(h, c)`; `uses_double_state()` → `True`.
- `BaseRecurrentLayer` — stacking, dropout between layers, `batch_first`,
  `initialize_cells(CellClass, **kwargs)`.
  - `BaseSingleRecurrentLayer` / `BaseDoubleRecurrentLayer` — iterate the cell
    stack over the time dimension.

### Conventions every cell follows

- Weights are concatenated per-gate into `weight_ih` / `weight_hh` with shape
  `(n_gates * hidden_size, ...)` and split with `.chunk(n, 0)` in `forward`.
- Separate input-side (`bias`) and recurrent-side (`recurrent_bias`) bias flags.
- Configurable `nonlinearity` / `gate_nonlinearity` and four init callables
  (`kernel_init`, `recurrent_kernel_init`, `bias_init`, `recurrent_bias_init`),
  defaulting to `xavier_uniform_` for weights and `zeros_` for biases.
- A cell `forward` accepts `(input_size,)` or `(N, input_size)` and handles the
  unbatched case internally via the `_preprocess_*` helpers.
- Extensive Google/NumPy-style docstrings with a math block and an arXiv link —
  these feed the Sphinx `generated/` autosummary pages.

## Adding a new model

1. Create `torchrecurrent/cells/<name>_cell.py` defining `<Name>Cell` (subclass a
   `BaseSingle*`/`BaseDouble*` cell) and `<Name>` (subclass the matching layer,
   calling `self.initialize_cells(<Name>Cell, **kwargs)`). Use `mgu_cell.py` as
   the reference template, including the docstring style.
2. Re-export both classes from `torchrecurrent/cells/__init__.py` (import +
   `__all__`) and from `torchrecurrent/__init__.py` (both import lists + `__all__`).
3. Add the cell to `CELL_CASES` in `tests/test_cells.py` and the layer to
   `tests/test_layers.py`.
4. Add docs: an entry under `docs/api/` and an autosummary stub under
   `docs/generated/`, plus the model catalog in `docs/models.rst`.

## Development workflow

```bash
pip install -e .[test]      # editable install with pytest + coverage

pytest                      # run the test suite
coverage run -m pytest      # how CI runs it

pre-commit run --all-files  # black + ruff --fix
black .                     # line length 92
flake8                      # excludes docs/, benchmarks/, tests/
```

- Code style: **black**, line length **92** (configured in both `pyproject.toml`
  and `.flake8`). Run black/ruff before committing — pre-commit enforces it.
- Tests are parametrized tables of model classes; keep them in sync when you add
  or rename a model.

## Conventions for agents

- **Keep cell and layer in the same file**, and keep the three export sites
  (`cells/__init__.py`, top-level `__init__.py`, and each `__all__`) consistent —
  a model missing from any of them won't be importable.
- Match the existing docstring format (math block + arXiv link + Args/Inputs/
  Outputs/Variables); docs generation depends on it.
- Don't commit into `benchmarks/.../runs/` — those are saved experiment artifacts.
- `rhn_cell.py` exists but is intentionally not exported; don't wire it up unless
  asked.
- Only `torch` may be added as a runtime dependency without discussion; keep the
  package dependency-light.
- Respect third-party licenses: `NASCell` is an Apache-2.0 re-implementation.
```
