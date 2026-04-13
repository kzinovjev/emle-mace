# emle-mace

A standalone Python package for training and evaluating the `EnergyEMLEMACE` model — a
[MACE](https://github.com/ACEsuit/mace) potential extended with per-atom electrostatic
properties required by the [EMLE-Engine](https://github.com/chemle/emle-engine) ML/MM
framework.

The integration is implemented almost entirely as monkey patches applied at training time
to `mace.cli.run_train` and `mace.tools.train`.  The patched functions are restored after
training completes, so the standard MACE entry points are unaffected.

## What the package provides

- **`EnergyEMLEMACE` model** (`emle_mace.models`) — a MACE variant with additional
  per-atom output heads for valence widths (`s`), core charges (`q_core`), total charges
  (`q`), and atomic dipoles (`mu`), plus learnable global Thole damping parameter
  (`a_Thole`) and element-wise polarizability volume ratios
  (`elements_alpha_v_ratios`).  Molecular polarizability tensors are derived from these
  quantities via the Thole model for training on reference QM values.

- **Loss function** (`emle_mace.loss`) — extends the standard MACE loss with weighted
  RMSE terms for each EMLE property.

- **Training CLI** (`emle-mace-train`) — thin wrapper around `mace_run_train.run()` that
  patches in the EMLE-specific model factory, loss function, optimizer param groups,
  error table, and evaluation function before delegating to the standard MACE training
  loop.

- **Evaluation CLI** (`emle-mace-eval`) — evaluates a trained model and reports RMSE for
  all EMLE properties.

## Installation

MACE and EMLE-Engine must be installed separately before installing this package.

```bash
# 3. Install this package
git clone https://github.com/kzinovjev/emle-mace.git
cd emle-mace
pip install -e .
```

## Usage

Training follows the standard MACE command-line interface with additional arguments for
EMLE property keys and loss weights:

```bash
emle-mace-train \
    --model EnergyEMLEMACE \
    --loss energy_emle \
    --error_table EnergyEMLERMSE \
    --train_file train.extxyz \
    --valid_file valid.extxyz \
    --valence_widths_key s \
    --core_charges_key q_core \
    --charges_key q \
    --atomic_dipoles_key mu \
    --polarizability_key alpha \
    --valence_widths_weight 1000 \
    --core_charges_weight 1000 \
    --charges_weight 10000 \
    --atomic_dipoles_weight 100 \
    --polarizability_weight 0.01 \
    ...  # all standard mace_run_train arguments are accepted
```

Evaluation:

```bash
emle-mace-eval \
    --model_path model.pt \
    --test_file test.extxyz \
    --output results.json \
    ...  # all standard mace_eval_configs arguments are accepted
```
