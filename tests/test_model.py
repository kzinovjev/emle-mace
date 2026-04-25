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


def _build_batch_from_positions(positions: np.ndarray):
    cfg = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions,
        properties={
            "energy": 0.0,
            "forces": np.zeros((3, 3)),
            "valence_widths": np.zeros(3),
            "core_charges": np.zeros(3),
            "charges": np.zeros(3),
            "atomic_dipoles": np.zeros((3, 3)),
            "polarizability": np.zeros((3, 3)),
            "total_charge": 0.0,
        },
        property_weights={
            "energy": 1.0, "forces": 1.0,
            "valence_widths": 1.0, "core_charges": 1.0,
            "charges": 1.0, "atomic_dipoles": 1.0,
            "polarizability": 1.0,
        },
    )
    atoms = [data.AtomicData.from_config(cfg, z_table=TABLE, cutoff=R_MAX)]
    return next(iter(torch_geometric.dataloader.DataLoader(
        atoms, batch_size=1, shuffle=False)))


def test_dipole_equivariance():
    """Atomic dipoles rotate covariantly with input positions; scalars are invariant.

    Regression test for the dead-dipole bug: the previous final readout used a
    scalar-only Activation, silently zeroing the deep dipole-path weights. The
    Gate-based replacement must contribute a real, equivariant dipole signal.
    """
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    R = o3.rand_matrix().to(torch.get_default_dtype())

    positions = np.array([
        [0.00, 0.00, 0.00],
        [0.96, 0.00, 0.00],
        [-0.24, 0.93, 0.00],
    ])
    rotated_positions = positions @ R.numpy().T

    batch_orig = _build_batch_from_positions(positions)
    batch_rot = _build_batch_from_positions(rotated_positions)

    out_orig = model(batch_orig.to_dict(), training=False, compute_force=False)
    out_rot = model(batch_rot.to_dict(), training=False, compute_force=False)

    # Scalars are rotation-invariant.
    assert torch.allclose(out_orig["energy"], out_rot["energy"], atol=1e-8)
    assert torch.allclose(out_orig["valence_widths"], out_rot["valence_widths"], atol=1e-8)
    assert torch.allclose(out_orig["core_charges"], out_rot["core_charges"], atol=1e-8)
    assert torch.allclose(out_orig["charges"], out_rot["charges"], atol=1e-8)

    # Dipoles transform as a vector: mu_rot = mu_orig @ R^T.
    expected = out_orig["atomic_dipoles"] @ R.T
    assert torch.allclose(out_rot["atomic_dipoles"], expected, atol=1e-7)

    # Sanity: the total dipole signal is non-zero at init.
    assert out_orig["atomic_dipoles"].abs().sum().item() > 0


def test_deep_readout_contributes_to_dipoles():
    """The final non-linear readout must contribute a non-zero l=1 output.

    With the old scalar-only Activation, the l=1 channel of linear_2 was forced
    to zero by equivariance — the deep readout's dipole output was identically
    zero for any input. This test fails on the buggy code.
    """
    torch.manual_seed(0)
    model = _make_model()
    deep_readout = model.readouts[-1]

    in_dim = deep_readout.linear_1.irreps_in.dim
    x = torch.randn(4, in_dim, dtype=torch.get_default_dtype())
    y = deep_readout(x)

    # Output irreps are "4x0e + 1x1o" — components 4:7 are the dipole.
    dipole_part = y[..., 4:]
    assert dipole_part.abs().sum().item() > 0