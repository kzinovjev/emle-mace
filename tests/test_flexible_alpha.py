"""Unit tests for the flexible (environment-dependent) polarizability head.

Covers:
  (a) a flexible forward pass produces a per-atom k_alpha of the right shape;
  (b) with k_alpha forced to 1 the molecular polarizability equals the fixed-
      alpha result (exact backward-compatibility check);
  (c) the k_alpha regularization term computes and differentiates;
  plus a TorchScript-compile smoke test of the flexible model.
"""

import numpy as np
import torch
from e3nn import o3

from mace import data, tools
from mace.tools import torch_geometric

from emle_mace.models import EnergyEMLEMACE
from emle_mace.loss import (
    compute_molecular_polarizabilities,
    mean_squared_error_k_alpha,
)

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
ATOMIC_ENERGIES = np.array([1.0, 3.0], dtype=float)
R_MAX = 5.0


def _make_model(use_flexible_alpha: bool):
    from mace import modules

    return EnergyEMLEMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(TABLE),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=3.0,
        atomic_numbers=TABLE.zs,
        correlation=3,
        gate=torch.nn.functional.silu,
        use_flexible_alpha=use_flexible_alpha,
    )


def _make_batch(n_configs=1):
    configs = []
    for _ in range(n_configs):
        cfg = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
            ),
            properties={
                "energy": -10.0,
                "forces": np.zeros((3, 3)),
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
# (a) shape / presence of the per-atom k_alpha output
# ---------------------------------------------------------------------------

def test_flexible_forward_k_alpha_shape():
    model = _make_model(use_flexible_alpha=True)
    batch = _make_batch(n_configs=2)
    n_atoms = batch.positions.shape[0]
    out = model(batch.to_dict(), training=False, compute_force=False)

    assert "k_alpha" in out and "k_alpha_sqrt" in out
    assert out["k_alpha"].shape == (n_atoms,)
    assert out["k_alpha_sqrt"].shape == (n_atoms,)
    # k_alpha = k_alpha_sqrt**2 and is strictly positive.
    assert torch.all(out["k_alpha"] > 0)
    assert torch.allclose(out["k_alpha"], out["k_alpha_sqrt"] ** 2)
    # Extra readout scalar must not corrupt the other per-atom outputs.
    assert out["valence_widths"].shape == (n_atoms,)
    assert out["atomic_dipoles"].shape == (n_atoms, 3)


def test_fixed_model_k_alpha_is_unity():
    """Fixed model still exposes k_alpha, identically 1 (readout is 4x0e+1x1o)."""
    model = _make_model(use_flexible_alpha=False)
    batch = _make_batch()
    n_atoms = batch.positions.shape[0]
    out = model(batch.to_dict(), training=False, compute_force=False)

    assert out["k_alpha"].shape == (n_atoms,)
    assert torch.allclose(out["k_alpha"], torch.ones(n_atoms))
    assert torch.allclose(out["k_alpha_sqrt"], torch.ones(n_atoms))
    # Readout has exactly 4 scalar + l=1 lanes -> dipoles at column 4.
    assert model.readouts[-1].irreps_out == o3.Irreps("4x0e + 1x1o")


def test_flexible_readout_has_extra_scalar():
    model = _make_model(use_flexible_alpha=True)
    assert model.readouts[-1].irreps_out == o3.Irreps("5x0e + 1x1o")


# ---------------------------------------------------------------------------
# (b) k_alpha == 1  =>  molecular alpha equals the fixed-alpha result
# ---------------------------------------------------------------------------

def test_k_alpha_unity_recovers_fixed_alpha():
    """With k_alpha forced to 1 the Thole molecular alpha equals the fixed path.

    The fixed path is emulated by removing k_alpha from the prediction dict, in
    which case compute_molecular_polarizabilities uses alpha_v_ratios alone --
    exactly what the pre-flexible code did.
    """
    model = _make_model(use_flexible_alpha=True)
    batch = _make_batch(n_configs=2)
    out = model(batch.to_dict(), training=False, compute_force=False)

    # Force k_alpha (and its sqrt) to unity.
    out_k1 = dict(out)
    out_k1["k_alpha"] = torch.ones_like(out["k_alpha"])
    out_k1["k_alpha_sqrt"] = torch.ones_like(out["k_alpha_sqrt"])
    alpha_k1 = compute_molecular_polarizabilities(batch, out_k1)

    # Fixed path: no k_alpha key at all.
    out_fixed = {k: v for k, v in out.items() if k not in ("k_alpha", "k_alpha_sqrt")}
    alpha_fixed = compute_molecular_polarizabilities(batch, out_fixed)

    assert torch.allclose(alpha_k1, alpha_fixed, atol=1e-10), (
        f"k_alpha=1 did not recover fixed alpha:\n{alpha_k1}\nvs\n{alpha_fixed}"
    )


def test_nonunit_k_alpha_changes_alpha():
    """A genuinely non-unit k_alpha must change the molecular polarizability
    (confirms k_alpha is actually wired into the Thole A matrix)."""
    model = _make_model(use_flexible_alpha=True)
    batch = _make_batch()
    out = model(batch.to_dict(), training=False, compute_force=False)

    out_fixed = {k: v for k, v in out.items() if k not in ("k_alpha", "k_alpha_sqrt")}
    alpha_fixed = compute_molecular_polarizabilities(batch, out_fixed)

    out_scaled = dict(out)
    out_scaled["k_alpha"] = torch.full_like(out["k_alpha"], 1.5)
    alpha_scaled = compute_molecular_polarizabilities(batch, out_scaled)

    assert not torch.allclose(alpha_scaled, alpha_fixed, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) regularization term computes and differentiates
# ---------------------------------------------------------------------------

def test_k_alpha_regularization_differentiable():
    model = _make_model(use_flexible_alpha=True)
    batch = _make_batch(n_configs=2)
    out = model(batch.to_dict(), training=True, compute_force=False)

    reg = mean_squared_error_k_alpha(batch, out)
    assert reg.ndim == 0
    assert torch.isfinite(reg)

    # It is a real function of the readout parameters -> non-zero gradient.
    reg.backward()
    grads = [
        p.grad for p in model.readouts[-1].parameters() if p.grad is not None
    ]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_k_alpha_regularization_zero_for_fixed_model():
    """A fixed model (k_alpha == 1) yields zero regularization."""
    model = _make_model(use_flexible_alpha=False)
    batch = _make_batch()
    out = model(batch.to_dict(), training=False, compute_force=False)
    reg = mean_squared_error_k_alpha(batch, out)
    assert reg.item() < 1e-12


# ---------------------------------------------------------------------------
# smoke: the flexible model TorchScript-compiles (needed for emle-engine)
# ---------------------------------------------------------------------------

def test_flexible_model_compiles():
    from e3nn.util import jit as e3nn_jit

    model = _make_model(use_flexible_alpha=True)
    model.eval()
    cmodel = e3nn_jit.compile(model)
    batch = _make_batch()
    out = cmodel(batch.to_dict(), training=False, compute_force=False)
    assert out["k_alpha"] is not None
    assert out["k_alpha"].shape == (batch.positions.shape[0],)
