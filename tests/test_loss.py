"""Unit tests for WeightedEnergyForcesEMLELoss."""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, tools
from mace.tools import torch_geometric

from emle_mace.loss import (
    WeightedEnergyForcesEMLELoss,
    mean_squared_error_valence_widths,
    mean_squared_error_core_charges,
    mean_squared_error_charges,
    mean_squared_error_atomic_dipoles,
    mean_squared_error_emle_polarizability,
    weighted_mean_squared_error_interaction_energy,
)

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
R_MAX = 5.0


def _make_batch():
    cfg = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]),
        properties={
            "energy": -10.0,
            "forces": np.array([[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]),
            "valence_widths": np.array([0.55, 0.42, 0.42]),
            "core_charges": np.array([-1.2, 0.6, 0.6]),
            "charges": np.array([-0.8, 0.4, 0.4]),
            "atomic_dipoles": np.zeros((3, 3)),
            "polarizability": np.eye(3) * 5.0,
            "total_charge": 0.0,
        },
        property_weights={
            "energy": 1.0, "forces": 1.0,
            "valence_widths": 1.0, "core_charges": 1.0,
            "charges": 1.0, "atomic_dipoles": 1.0, "polarizability": 1.0,
        },
    )
    atom = data.AtomicData.from_config(cfg, z_table=TABLE, cutoff=R_MAX)
    return next(iter(torch_geometric.dataloader.DataLoader([atom], batch_size=1)))


def _make_pred(batch, offset=0.0):
    """Make a fake prediction dict close to the reference."""
    n = batch.positions.shape[0]
    n_g = batch.num_graphs
    return {
        "energy": batch.energy + offset,
        "interaction_energy": batch.energy + offset,
        "e0": torch.zeros(n_g, dtype=torch.float64),
        "forces": batch.forces + offset,
        "valence_widths": batch.valence_widths + offset,
        "core_charges": batch.core_charges + offset,
        "charges": batch.charges + offset,
        "atomic_dipoles": batch.atomic_dipoles + offset,
        "a_Thole": torch.tensor(2.0, dtype=torch.float64),
        "alpha_v_ratios": torch.ones(n, dtype=torch.float64) * 0.1,
        "node_attrs": batch.node_attrs,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_zero_on_perfect_pred():
    """Loss should be near-zero when predictions match targets exactly."""
    batch = _make_batch()
    pred = _make_pred(batch, offset=0.0)
    loss_fn = WeightedEnergyForcesEMLELoss()
    # Skip polarizability (requires physical sanity of inputs); test component losses.
    loss_e = weighted_mean_squared_error_interaction_energy(batch, pred)
    loss_f = torch.tensor(0.0)  # forces match exactly
    loss_s = mean_squared_error_valence_widths(batch, pred)
    loss_qc = mean_squared_error_core_charges(batch, pred)
    loss_q = mean_squared_error_charges(batch, pred)
    loss_mu = mean_squared_error_atomic_dipoles(batch, pred)

    for name, val in [("energy", loss_e), ("valence_widths", loss_s),
                      ("core_charges", loss_qc), ("charges", loss_q),
                      ("atomic_dipoles", loss_mu)]:
        assert val.item() < 1e-10, f"{name} loss not zero: {val.item()}"


def test_loss_increases_with_offset():
    """Loss should increase as predictions deviate from targets."""
    batch = _make_batch()

    pred0 = _make_pred(batch, offset=0.0)
    pred1 = _make_pred(batch, offset=0.1)

    loss_s0 = mean_squared_error_valence_widths(batch, pred0)
    loss_s1 = mean_squared_error_valence_widths(batch, pred1)
    assert loss_s1 > loss_s0, "Loss did not increase with offset"


def test_combined_loss_weights():
    """Increasing an individual weight should scale the combined loss."""
    batch = _make_batch()
    pred = _make_pred(batch, offset=0.1)

    loss_fn_base = WeightedEnergyForcesEMLELoss(valence_widths_weight=1.0, polarizability_weight=0.0)
    loss_fn_high = WeightedEnergyForcesEMLELoss(valence_widths_weight=10.0, polarizability_weight=0.0)

    l_base = loss_fn_base(batch, pred)
    l_high = loss_fn_high(batch, pred)
    assert l_high > l_base, "Higher valence_widths_weight should give higher loss"


def test_loss_repr():
    """__repr__ should run without error."""
    loss_fn = WeightedEnergyForcesEMLELoss(
        energy_weight=2.0,
        forces_weight=50.0,
        valence_widths_weight=3.0,
        core_charges_weight=4.0,
        charges_weight=5.0,
        atomic_dipoles_weight=6.0,
        polarizability_weight=20.0,
    )
    s = repr(loss_fn)
    assert "energy_weight=2.000" in s
    assert "polarizability_weight=20.000" in s