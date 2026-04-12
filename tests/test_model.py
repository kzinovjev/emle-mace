"""Unit tests for EnergyEMLEMACE."""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, tools
from mace.tools import torch_geometric

from emle_mace.models import EnergyEMLEMACE

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TABLE = tools.AtomicNumberTable([1, 8])
ATOMIC_ENERGIES = np.array([1.0, 3.0], dtype=float)

R_MAX = 5.0


def _make_model():
    from mace import modules

    return EnergyEMLEMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(TABLE),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=3.0,
        atomic_numbers=TABLE.zs,
        correlation=3,
        gate=torch.nn.functional.silu,
    )


def _make_batch(n_configs=1):
    """Build a minimal PyG batch from a water molecule."""
    configs = []
    for _ in range(n_configs):
        cfg = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]),
            properties={
                "energy": -10.0,
                "forces": np.zeros((3, 3)),
                "valence_widths": np.array([0.5, 0.4, 0.4]),
                "core_charges": np.array([-1.0, 0.5, 0.5]),
                "charges": np.array([-0.8, 0.4, 0.4]),
                "atomic_dipoles": np.zeros((3, 3)),
                "polarizability": np.eye(3),
                "total_charge": 0.0,
            },
            property_weights={
                "energy": 1.0,
                "forces": 1.0,
                "valence_widths": 1.0,
                "core_charges": 1.0,
                "charges": 1.0,
                "atomic_dipoles": 1.0,
                "polarizability": 1.0,
            },
        )
        configs.append(cfg)

    atoms_list = [
        data.AtomicData.from_config(cfg, z_table=TABLE, cutoff=R_MAX)
        for cfg in configs
    ]
    return next(
        iter(
            torch_geometric.dataloader.DataLoader(
                atoms_list, batch_size=len(atoms_list), shuffle=False
            )
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_output_keys():
    """EnergyEMLEMACE.forward() must return all expected output keys."""
    model = _make_model()
    batch = _make_batch()
    out = model(batch.to_dict(), training=False, compute_force=True)

    required_keys = [
        "energy", "forces", "interaction_energy", "e0",
        "valence_widths", "core_charges", "charges",
        "atomic_dipoles", "a_Thole", "alpha_v_ratios",
    ]
    for key in required_keys:
        assert key in out, f"Missing output key: {key}"


def test_forward_output_shapes():
    """Check that tensor shapes are consistent with input."""
    model = _make_model()
    batch = _make_batch(n_configs=2)
    n_atoms = batch.positions.shape[0]
    n_graphs = batch.num_graphs

    out = model(batch.to_dict(), training=False, compute_force=True)

    assert out["energy"].shape == (n_graphs,), f"energy shape {out['energy'].shape}"
    assert out["forces"].shape == (n_atoms, 3), f"forces shape {out['forces'].shape}"
    assert out["valence_widths"].shape == (n_atoms,)
    assert out["core_charges"].shape == (n_atoms,)
    assert out["charges"].shape == (n_atoms,)
    assert out["atomic_dipoles"].shape == (n_atoms, 3)


def test_charge_conservation():
    """Net charge per molecule must match total_charge after correction."""
    model = _make_model()
    batch = _make_batch(n_configs=1)
    out = model(batch.to_dict(), training=False, compute_force=False)

    # Sum charges over all atoms (single graph)
    total_q = out["charges"].sum().item()
    target_q = batch.total_charge.item()
    assert abs(total_q - target_q) < 1e-5, (
        f"Charge not conserved: sum={total_q:.6f}, target={target_q:.6f}"
    )


def test_forward_no_nan():
    """No NaN values in any output tensor."""
    model = _make_model()
    batch = _make_batch()
    out = model(batch.to_dict(), training=False, compute_force=True)

    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            assert not torch.isnan(val).any(), f"NaN in output['{key}']"


def test_model_parameters_exist():
    """a_Thole and elements_alpha_v_ratios must be registered parameters."""
    model = _make_model()
    param_names = [n for n, _ in model.named_parameters()]
    assert "a_Thole" in param_names
    assert "elements_alpha_v_ratios" in param_names