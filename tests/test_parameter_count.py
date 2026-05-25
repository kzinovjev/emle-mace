"""EnergyEMLEMACE should add a bounded parameter overhead over a vanilla MACE
with the same backbone: above zero (the equivariant l=1/l=2 readout heads carry
real weights) and below an upper sanity bound.
"""

import numpy as np
import torch
from e3nn import o3

from mace import modules, tools
from mace.modules.models import MACE

from emle_mace.models import EnergyEMLEMACE

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 8])
ATOMIC_ENERGIES = np.array([1.0, 3.0], dtype=float)
R_MAX = 5.0


def _common_kwargs():
    return dict(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(TABLE),
        hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=3.0,
        atomic_numbers=TABLE.zs,
        correlation=3,
        gate=torch.nn.functional.silu,
    )


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_parameter_count_overhead():
    """EnergyEMLEMACE should add a bounded parameter overhead vs. plain MACE."""
    emle_model = EnergyEMLEMACE(**_common_kwargs())
    mace_model = MACE(**_common_kwargs())

    n_emle = _count_params(emle_model)
    n_mace = _count_params(mace_model)

    delta = (n_emle - n_mace) / n_mace
    # Overhead is a few percent on this small fixture; the bounds guard against
    # the equivariant readout heads vanishing (delta -> 0) or ballooning.
    assert 0.025 < delta < 0.10, (
        f"Unexpected EnergyEMLEMACE/MACE parameter overhead on small fixture: "
        f"{delta:.2%} (emle={n_emle}, mace={n_mace})"
    )
