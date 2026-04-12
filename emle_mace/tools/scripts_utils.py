"""Training script utilities for EnergyEMLEMACE.

Provides get_loss_fn() and get_swa() overrides for the ``energy_emle`` loss
type.  For all other loss types the calls are forwarded to mace's equivalents.
"""

import logging

from emle_mace.loss import WeightedEnergyForcesEMLELoss


def get_loss_fn(args, *extra_args, **extra_kwargs):
    """Return the appropriate loss function for the given args.

    Handles ``args.loss == "energy_emle"``; delegates everything else to
    ``mace.tools.scripts_utils.get_loss_fn``.

    Extra positional/keyword args (e.g. ``dipole_only``, ``compute_dipole``)
    are forwarded to mace for non-EMLE losses but are irrelevant for
    EnergyEMLEMACE (which always computes dipoles/EMLE properties).
    """
    if args.loss != "energy_emle":
        from mace.tools.scripts_utils import get_loss_fn as _mace_get_loss_fn
        return _mace_get_loss_fn(args, *extra_args, **extra_kwargs)

    loss_fn = WeightedEnergyForcesEMLELoss(
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        valence_widths_weight=getattr(args, "valence_widths_weight", 1.0),
        core_charges_weight=getattr(args, "core_charges_weight", 1.0),
        charges_weight=getattr(args, "charges_weight", 1.0),
        atomic_dipoles_weight=getattr(args, "atomic_dipoles_weight", 1.0),
        polarizability_weight=getattr(args, "polarizability_weight", 10.0),
    )
    logging.info(f"Loss function: {loss_fn}")
    return loss_fn


def get_swa(args, model, optimizer, swas, dipole_only=False):
    """Return SWA (Stage Two) loss and configuration.

    Handles ``args.loss == "energy_emle"``; delegates everything else to
    ``mace.tools.scripts_utils.get_swa``.
    """
    if args.loss != "energy_emle":
        from mace.tools.scripts_utils import get_swa as _mace_get_swa
        return _mace_get_swa(args, model, optimizer, swas, dipole_only)

    assert not dipole_only, "Stage Two for dipole fitting not implemented"
    swas.append(True)
    if args.start_swa is None:
        args.start_swa = max(1, args.max_num_epochs // 4 * 3)
    else:
        if args.start_swa >= args.max_num_epochs:
            logging.warning(
                f"start_swa ({args.start_swa}) must be < max_num_epochs "
                f"({args.max_num_epochs}); disabling Stage Two"
            )
            swas[-1] = False

    loss_fn_energy = WeightedEnergyForcesEMLELoss(
        energy_weight=args.swa_energy_weight,
        forces_weight=args.swa_forces_weight,
        valence_widths_weight=getattr(args, "valence_widths_weight", 1.0),
        core_charges_weight=getattr(args, "core_charges_weight", 1.0),
        charges_weight=getattr(args, "charges_weight", 1.0),
        atomic_dipoles_weight=getattr(args, "atomic_dipoles_weight", 1.0),
        polarizability_weight=getattr(args, "polarizability_weight", 10.0),
    )
    logging.info(
        f"Stage Two (after {args.start_swa} epochs): {loss_fn_energy}, "
        f"energy_weight={args.swa_energy_weight}, "
        f"forces_weight={args.swa_forces_weight}, "
        f"lr={args.swa_lr}"
    )

    import torch
    from torch.optim.swa_utils import AveragedModel, SWALR
    from mace.tools.scripts_utils import SWAContainer

    swa = SWAContainer(
        model=AveragedModel(model),
        scheduler=SWALR(
            optimizer=optimizer,
            swa_lr=args.swa_lr,
            anneal_epochs=1,
            anneal_strategy="linear",
        ),
        start=args.start_swa,
        loss_fn=loss_fn_energy,
    )
    return swa, swas