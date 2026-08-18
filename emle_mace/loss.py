###########################################################################################
# Loss functions for EnergyEMLEMACE training
# Extracted from the emle-mace MACE fork (mace/mace/modules/loss.py).
###########################################################################################

from typing import Optional

import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


def _pad_to_shape(tensor, shape, value=0):
    pad = [p for m, s in reversed(list(zip(shape, tensor.shape)))
           for p in [0, int(m - s)]]
    return torch.nn.functional.pad(tensor, pad, value=value)


def _flat_to_padded(flat_tensor, ptr, value=0):
    tensors = [flat_tensor[first:last] for first, last in zip(ptr[:-1], ptr[1:])]
    shape = torch.max(torch.tensor([list(t.shape) for t in tensors], dtype=torch.float), dim=0).values
    return torch.stack([_pad_to_shape(t, shape, value) for t in tensors])


# ---------------------------------------------------------------------------
# Molecular polarizability helper (depends on emle-engine)
# ---------------------------------------------------------------------------

def compute_molecular_polarizabilities(batch: Batch, output: TensorDict):
    """Compute molecular polarizabilities via the Thole damping model.

    Uses emle-engine's EMLEBase and TholeLoss utilities.
    """
    from emle.models._emle_base import EMLEBase
    from emle.train._loss import TholeLoss
    from emle._units import _ANGSTROM_TO_BOHR

    mask = _flat_to_padded(torch.ones_like(batch.batch, dtype=torch.bool), batch.ptr)
    positions_mol = _flat_to_padded(batch.positions, batch.ptr)

    r_data = EMLEBase._get_r_data(positions_mol * _ANGSTROM_TO_BOHR, mask)
    s = _flat_to_padded(output["valence_widths"], batch.ptr) * mask
    q_val = _flat_to_padded(output["charges"] - output["core_charges"], batch.ptr) * mask
    # Flexible-alpha: multiply the per-element ratio by the per-atom k_alpha
    # correction. Absent or unity -> exact fixed-alpha result.
    alpha_v_ratios = output["alpha_v_ratios"]
    k_alpha = output.get("k_alpha")
    if k_alpha is not None:
        alpha_v_ratios = alpha_v_ratios * k_alpha
    k = _flat_to_padded(alpha_v_ratios, batch.ptr) * mask

    A_thole = EMLEBase._get_A_thole(r_data, s, q_val, k, output["a_Thole"])
    return TholeLoss._get_alpha_mol(A_thole, mask)[0]


def _is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def _reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    if ddp is None:
        ddp = _is_ddp_enabled()
    if ddp:
        dist.all_reduce(raw_loss, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return raw_loss.mean() / world_size
    return raw_loss.mean()


# ---------------------------------------------------------------------------
# Per-property loss helpers
# ---------------------------------------------------------------------------

def weighted_mean_squared_error_interaction_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs]
    ref_interaction_energy = ref["energy"] - pred["e0"]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref_interaction_energy - pred["interaction_energy"]) / num_atoms)
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_valence_widths(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_valence_widths_weight = torch.repeat_interleave(
        ref.valence_widths_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_valence_widths_weight
        * torch.square(ref["valence_widths"] - pred["valence_widths"])
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_core_charges(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_core_charges_weight = torch.repeat_interleave(
        ref.core_charges_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_core_charges_weight
        * torch.square(ref["core_charges"] - pred["core_charges"])
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_charges(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_charges_weight = torch.repeat_interleave(
        ref.charges_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_charges_weight
        * torch.square(ref["charges"] - pred["charges"])
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_atomic_dipoles(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_atomic_dipoles_weight = torch.repeat_interleave(
        ref.atomic_dipoles_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_atomic_dipoles_weight
        * torch.square(ref["atomic_dipoles"] - pred["atomic_dipoles"])
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_atomic_quadrupoles(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """MSE on the symmetric traceless Cartesian atomic quadrupoles.

    The model predicts a (n_atoms, 3, 3) tensor; the reference is stored as the
    6 upper-triangle components [xx, xy, xz, yy, yz, zz], so the prediction is
    reduced to those 6 components before comparison.
    """
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_atomic_quadrupoles_weight = torch.repeat_interleave(
        ref.atomic_quadrupoles_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    triu_row, triu_col = torch.triu_indices(3, 3, offset=0)
    pred_triu = pred["atomic_quadrupoles"][:, triu_row, triu_col]  # [n_atoms, 6]
    raw_loss = (
        configs_weight
        * configs_atomic_quadrupoles_weight
        * torch.square(ref["atomic_quadrupoles"] - pred_triu)
    )
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_emle_polarizability(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    alpha_ref = ref["polarizability"]

    valence_charges = pred["charges"] - pred["core_charges"]

    if (
        torch.min(pred["valence_widths"]) < 0.3
        or torch.max(pred["valence_widths"]) > 1
        or torch.max(valence_charges) > -0.5
    ):
        alpha_pred = torch.zeros_like(alpha_ref)
    else:
        alpha_pred = compute_molecular_polarizabilities(ref, pred)

    triu_row, triu_col = torch.triu_indices(3, 3, offset=0)
    alpha_ref_triu = alpha_ref[:, triu_row, triu_col]
    alpha_pred_triu = alpha_pred[:, triu_row, triu_col]

    raw_loss = torch.square(alpha_ref_triu - alpha_pred_triu)
    return _reduce_loss(raw_loss, ddp)


def mean_squared_error_k_alpha(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """Regularization keeping the flexible polarizability scaling near unity.

    Mirrors the GPR flexible-model regularization in emle-engine
    (``emle/train/_loss.py`` ``TholeLoss.forward``), which penalizes the squared
    departure of the per-reference ``ref_values_sqrtk`` from 1, mean-normalized:

        l2_reg * sum((ref_values_sqrtk - 1)**2 * mask) / sum(n_ref)

    Here the per-atom analogue of ``sqrtk`` is ``k_alpha_sqrt`` (with
    ``k_alpha = k_alpha_sqrt**2``), so we penalize ``(k_alpha_sqrt - 1)**2`` as an
    unweighted mean over atoms. This drives ``k_alpha -> 1`` (i.e. alpha back to
    the fixed MBIS-volume value) unless the polarizability fit demands otherwise.
    Returns 0 for fixed models (``k_alpha_sqrt`` absent or identically 1).
    """
    k_alpha_sqrt = pred.get("k_alpha_sqrt")
    if k_alpha_sqrt is None:
        return torch.zeros((), device=ref.weight.device, dtype=ref.weight.dtype)
    raw_loss = torch.square(k_alpha_sqrt - 1.0)
    return _reduce_loss(raw_loss, ddp)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class WeightedEnergyForcesEMLELoss(torch.nn.Module):
    """Combined loss for EnergyEMLEMACE training.

    Sums weighted MSE losses over:
      energy (interaction), forces, valence widths, core charges,
      total charges, atomic dipoles, atomic quadrupoles, and molecular
      polarizability.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        valence_widths_weight: float = 1.0,
        core_charges_weight: float = 1.0,
        charges_weight: float = 1.0,
        atomic_dipoles_weight: float = 1.0,
        atomic_quadrupoles_weight: float = 1.0,
        polarizability_weight: float = 10.0,
        k_alpha_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "valence_widths_weight",
            torch.tensor(valence_widths_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "core_charges_weight",
            torch.tensor(core_charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "atomic_dipoles_weight",
            torch.tensor(atomic_dipoles_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "atomic_quadrupoles_weight",
            torch.tensor(atomic_quadrupoles_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "k_alpha_reg_weight",
            torch.tensor(k_alpha_reg_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        from mace.modules.loss import mean_squared_error_forces

        loss_energy = weighted_mean_squared_error_interaction_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_valence_widths = mean_squared_error_valence_widths(ref, pred, ddp)
        loss_core_charges = mean_squared_error_core_charges(ref, pred, ddp)
        loss_charges = mean_squared_error_charges(ref, pred, ddp)
        # Multipole heads above the model's max_static_L are absent (None).
        _zero = torch.zeros_like(loss_energy)
        loss_atomic_dipoles = (
            mean_squared_error_atomic_dipoles(ref, pred, ddp)
            if pred.get("atomic_dipoles") is not None
            else _zero
        )
        loss_atomic_quadrupoles = (
            mean_squared_error_atomic_quadrupoles(ref, pred, ddp)
            if pred.get("atomic_quadrupoles") is not None
            else _zero
        )
        loss_polarizability = mean_squared_error_emle_polarizability(ref, pred, ddp)
        loss_k_alpha = mean_squared_error_k_alpha(ref, pred, ddp)

        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.valence_widths_weight * loss_valence_widths
            + self.core_charges_weight * loss_core_charges
            + self.charges_weight * loss_charges
            + self.atomic_dipoles_weight * loss_atomic_dipoles
            + self.atomic_quadrupoles_weight * loss_atomic_quadrupoles
            + self.polarizability_weight * loss_polarizability
            + self.k_alpha_reg_weight * loss_k_alpha
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"valence_widths_weight={self.valence_widths_weight:.3f}, "
            f"core_charges_weight={self.core_charges_weight:.3f}, "
            f"charges_weight={self.charges_weight:.3f}, "
            f"atomic_dipoles_weight={self.atomic_dipoles_weight:.3f}, "
            f"atomic_quadrupoles_weight={self.atomic_quadrupoles_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f}, "
            f"k_alpha_reg_weight={self.k_alpha_reg_weight:.3f})"
        )