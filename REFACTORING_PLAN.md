# emle-mace Refactoring Plan

## Background

The `EnergyEMLEMACE` model class is currently implemented in the `emle-mace` branch of the
`mace` fork (`mace/mace/modules/models.py`). This is a model that predicts — in a single
forward pass — energies, forces, valence widths, core/valence charges, static atomic
dipoles, and molecular polarizabilities. These extra outputs feed directly into the EMLE
QM/MM embedding workflow.

Keeping `EnergyEMLEMACE` in the `mace` repository is undesirable:

- It is EMLE-specific and has no relevance to upstream `mace`.
- Its training infrastructure is scattered across 7 files in `mace`, making the `mace`
  fork increasingly hard to keep in sync with upstream.
- The goal is to be able to train `EnergyEMLEMACE` models using only **standard mace**,
  **emle-engine** (emle-mace branch), and a new **emle-mace** package.

---

## Inventory of EMLE-specific code in the mace fork

| File | Contents | ~Lines |
|------|----------|--------|
| `mace/modules/models.py` | `EnergyEMLEMACE` class | 435 |
| `mace/modules/loss.py` | `WeightedEnergyForcesEMLELoss`, `mean_squared_error_emle_polarizability` | 60 |
| `mace/tools/train.py` | EMLE metric accumulators (`valence_widths_computed`, `core_charges_computed`, …) + `EnergyEMLERMSE` log branch | 50 |
| `mace/tools/tables_utils.py` | `EnergyEMLERMSE` table columns and row formatting | 30 |
| `mace/tools/model_script_utils.py` | `EnergyEMLEMACE` construction branch, `extract_config_mace_model` patch, `a_Thole`/`elements_alpha_v_ratios` optimizer param group | 20 |
| `mace/tools/scripts_utils.py` | `energy_emle` loss selection in `get_loss_fn` and `get_swa`, `EnergyEMLEMACE` in model-name checks | 25 |
| `mace/tools/arg_parser.py` | `"EnergyEMLEMACE"` model choice, `"EnergyEMLERMSE"` error-table choice, five EMLE loss-weight args | 15 |
| `mace/cli/run_train.py` | Three lines setting `args.compute_emle` | 3 |

Total: ~640 lines across 7 files. The code is shallow but wide — it touches every layer
of the training stack.

---

## Options

### Option A — Standalone `emle-mace` package (recommended)

`EnergyEMLEMACE` and everything needed to train it lives exclusively in `emle-mace`.
Standard `mace` and `emle-engine` are not modified (beyond removing the EMLE additions
from the `mace` fork).

**Proposed package layout:**

```
emle-mace/
  emle_mace/
    __init__.py
    models/
      __init__.py
      _emle_mace.py        # EnergyEMLEMACE (from mace/modules/models.py)
    loss.py                # WeightedEnergyForcesEMLELoss (from mace/modules/loss.py)
    tools/
      __init__.py
      arg_parser.py        # build_emle_arg_parser() — wraps mace's parser, adds EMLE args
      model_utils.py       # configure_model() — constructs EnergyEMLEMACE
      metrics.py           # EnergyEMLERMSE accumulator, table, per-epoch logging
      scripts_utils.py     # get_loss_fn() and get_swa() for the energy_emle case
    cli/
      run_train.py         # training entry point
  pyproject.toml
  tests/
```

**Training CLI design:**

`mace/cli/run_train.py` is 1082 lines, but only 3 of them are EMLE-specific. The
EMLE logic is spread across the utility modules it calls. `emle_mace/cli/run_train.py`
can therefore be a thin wrapper (~150–200 lines) that:

1. Imports mace's arg parser and appends EMLE-specific arguments.
2. Calls mace's data-loading utilities unchanged — they are fully generic and load
   whatever fields are present in `.xyz` / HDF5 files.
3. Substitutes `emle_mace.tools.model_utils.configure_model()` for mace's version.
4. Substitutes `emle_mace.tools.scripts_utils.get_loss_fn()` for mace's version.
5. Calls mace's `train.py` training loop — the EMLE metric accumulators in
   `train.py` are already guarded by `if output.get("valence_widths") is not None`
   and will fire automatically when `EnergyEMLEMACE` is used.
6. Provides its own `EnergyEMLERMSE` table construction and logging (the only piece
   that cannot be reused from standard mace without modification).

**Dependency graph:**

```
mace (standard, unmodified)
        ↑
   emle-mace  ←→  emle-engine (emle-mace branch, inference only)
```

**Note on `torch.load()` deserialization:** `MACEEMLEJoint` in emle-engine loads a
pre-trained model via `torch.load()`. For deserialization to succeed, `EnergyEMLEMACE`
must be importable. If this class lives in `emle-mace`, QM/MM inference users need
`emle-mace` installed even when they are not training. This can be mitigated by adding
a re-export stub in emle-engine:

```python
# emle-engine/emle/models/_emle_mace_compat.py
try:
    from emle_mace.models import EnergyEMLEMACE  # noqa: F401 — needed for torch.load()
except ImportError:
    EnergyEMLEMACE = None
```

**Pros:**
- Cleanest separation: `mace` stays general-purpose, `emle-engine` stays as a QM/MM
  simulation framework, `emle-mace` is the EMLE ML-potential training package.
- Dependency direction is unambiguous: `emle-mace` depends on `mace` and `emle-engine`;
  nothing depends on `emle-mace` except a user's training environment.
- `emle-engine`'s training code (GPR-based: `EMLETrainer`, `_gpr.py`, `_ivm.py`) remains
  architecturally consistent — GPR only, no NN training mixed in.
- Straightforward to publish as an independent package on PyPI.

**Cons:**
- QM/MM inference users must install `emle-mace` even when not training, unless the
  re-export stub above is used.
- Three packages to keep in sync when mace's graph API or data format changes.

---

### Option B — Move everything into emle-engine

`EnergyEMLEMACE` and the gradient-based training infrastructure move into
`emle-engine/emle/models/` and `emle-engine/emle/train/`.

**Pros:**
- Two packages instead of three.
- Inference users do not need a separate package to deserialize saved models.

**Cons:**
- `emle-engine`'s `train/` module becomes architecturally inconsistent: it now mixes
  GPR-based (EMLETrainer) and gradient-based NN training with completely different data
  formats, paradigms, and tooling.
- `emle-engine` would need `e3nn`, `matscipy`, and other heavy MACE build-time
  dependencies for all users — including those who never use MACE.
- The training CLI would need to call `mace.tools.*` utilities heavily, giving
  `emle-engine` a strong coupling to mace's internals at the training level.

---

### Option C — Model in emle-engine, training CLI in emle-mace

A hybrid: `EnergyEMLEMACE` lives in `emle-engine` (next to `MACEEMLEJoint` which uses
it at inference time); the training CLI and loss functions live in `emle-mace`.

```
emle-engine/emle/models/_emle_mace.py    # EnergyEMLEMACE model class
emle-mace/emle_mace/
  loss.py                                # WeightedEnergyForcesEMLELoss
  tools/                                 # training utilities
  cli/run_train.py                       # training entry point
```

**Dependency graph:**

```
mace (standard)
      ↑              ↑
emle-engine      emle-mace
      ↑               ↑
      └───────────────┘
   (emle-mace imports EnergyEMLEMACE from emle-engine)
```

**Pros:**
- Inference users only need `mace + emle-engine`; `emle-mace` is not required to
  load saved models.
- `emle-engine` owns the model class it uses at inference time.
- `emle-mace` becomes a thin training-only package (~300 lines total).
- `emle-engine`'s `train/` module stays GPR-only.

**Cons:**
- `emle-engine` gains `e3nn` as a hard dependency (currently optional via the MACE
  backend). `EnergyEMLEMACE.__init__` directly instantiates `e3nn.o3.Irreps` and mace
  `InteractionBlock` objects.
- The model is defined in the package that uses it for inference, but its primary
  creation path is training in `emle-mace` — a slightly awkward ownership split.

---

## Recommendation

**Option A** is the recommended starting point. It gives the cleanest separation and
avoids pulling heavy mace/e3nn build-time dependencies into `emle-engine`. The
`torch.load()` concern is addressed by the re-export stub described above.

**Option C is a strong alternative** if the priority is a minimal install for QM/MM
inference users (`pip install emle-engine` and everything works without `emle-mace`).
The tradeoff is `emle-engine` gaining `e3nn` as a hard dependency.

---

## Implementation Steps (Option A)

### Step 1 — Scaffold the package

Create `emle-mace/pyproject.toml` (or `setup.cfg`) with:

- `name = "emle-mace"`
- `install_requires`: `mace-torch`, `emle-engine` (emle-mace branch), `torch`, `e3nn`
- `console_scripts`: `emle-mace-train = emle_mace.cli.run_train:main`

### Step 2 — Move `EnergyEMLEMACE`

Copy `mace/mace/modules/models.py:1417–1852` to
`emle_mace/models/_emle_mace.py`. The class only imports from `mace.modules.*`
and standard libraries — no changes to the class body are needed. Update
`mace/mace/modules/__init__.py` to remove the `EnergyEMLEMACE` export.

Add the re-export stub to emle-engine as described above.

### Step 3 — Move loss code

Copy `WeightedEnergyForcesEMLELoss` and `mean_squared_error_emle_polarizability`
from `mace/mace/modules/loss.py` to `emle_mace/loss.py`. Remove them from
`mace/mace/modules/loss.py` and `mace/mace/modules/__init__.py`.

### Step 4 — Create `emle_mace/tools/`

**`arg_parser.py`**

```python
from mace.tools.arg_parser import build_default_arg_parser

def build_emle_arg_parser():
    parser = build_default_arg_parser()
    # Add "EnergyEMLEMACE" to --model choices and "EnergyEMLERMSE" to
    # --error_table choices, then add the five EMLE loss-weight arguments
    # (valence_widths_weight, core_charges_weight, charges_weight,
    #  atomic_dipoles_weight, polarizability_weight).
    # Lifted from mace/mace/tools/arg_parser.py.
    return parser
```

**`model_utils.py`**

Lift the `if args.model == "EnergyEMLEMACE":` block from
`mace/mace/tools/model_script_utils.py:335–352` and the
`extract_config_mace_model` patch for `EnergyEMLEMACE`
(lines 328–338 of the same file). Also move the `a_Thole` /
`elements_alpha_v_ratios` optimizer parameter group (line 847).

**`metrics.py`**

Lift the `EnergyEMLERMSE` table-column definition and row-formatting code from
`mace/mace/tools/tables_utils.py:110–124` and `279–299`, plus the
`EnergyEMLERMSE` log-line formatter from `mace/mace/tools/train.py:149–164`.

**`scripts_utils.py`**

Lift the two `elif args.loss == "energy_emle":` branches from
`mace/mace/tools/scripts_utils.py:661–677` (primary) and `741–757` (SWA).

### Step 5 — Write `emle_mace/cli/run_train.py`

The script imports mace utilities for all generic steps and substitutes the
EMLE-specific pieces:

```python
from mace.cli.run_train import (
    setup_distributed, load_datasets, build_z_table, ...
)
from emle_mace.tools.arg_parser import build_emle_arg_parser
from emle_mace.tools.model_utils import configure_model, get_optimizer_options
from emle_mace.tools.scripts_utils import get_loss_fn, get_swa
from emle_mace.tools.metrics import create_emle_error_table, log_emle_errors

def main():
    parser = build_emle_arg_parser()
    args = parser.parse_args()
    # ... standard mace setup: distributed, seeds, dtype, device ...
    # ... mace data loading (unchanged) ...
    model = configure_model(args, ...)           # returns EnergyEMLEMACE
    loss_fn = get_loss_fn(args)                  # returns WeightedEnergyForcesEMLELoss
    # ... mace training loop, passing our model + loss ...
```

### Step 6 — Clean up the mace fork

Remove the EMLE-specific diffs from all 7 files listed in the inventory. After
this, the `emle-mace` branch of the `mace` fork either:

- (a) Becomes identical to upstream `mace` and can be replaced by a direct
  `mace-torch` PyPI dependency, or
- (b) Retains only the small set of changes that are genuinely needed for
  non-EMLE reasons (if any exist).

### Step 7 — Update emle-engine

`MACEEMLEJoint` in `emle-engine/emle/models/_mace.py` loads pre-trained models
via `torch.load()`. No structural changes are needed. Add a conditional import
at the top of the file (or in `emle/models/__init__.py`) so that `EnergyEMLEMACE`
is registered in the Python namespace when emle-mace is installed:

```python
try:
    from emle_mace.models import EnergyEMLEMACE  # noqa: F401
except ImportError:
    pass  # emle-mace not installed; torch.load() will fail if model uses EnergyEMLEMACE
```

### Step 8 — Tests

Add tests in `emle-mace/tests/` covering:

- `EnergyEMLEMACE` forward pass (unit): construct a small model and verify output
  keys and tensor shapes.
- `WeightedEnergyForcesEMLELoss` (unit): check that all loss components are
  computed and weighted correctly.
- End-to-end training (integration): a minimal training run on synthetic data,
  borrowing the fixture helpers from `mace/tests/test_run_train.py`.