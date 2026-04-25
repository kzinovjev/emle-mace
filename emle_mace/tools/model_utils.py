"""Model construction utilities for EnergyEMLEMACE.

Provides configure_model() which is a drop-in replacement for
mace.tools.model_script_utils.configure_model() when the selected model is
EnergyEMLEMACE.  For all other model types the call is forwarded to mace.
"""

import ast
import logging

from e3nn import o3

from mace import modules as mace_modules
from mace.tools.model_script_utils import configure_model as _mace_configure_model

from emle_mace.models import EnergyEMLEMACE


def configure_model(args, train_loader, atomic_energies, model_foundation=None,
                    heads=None, z_table=None, head_configs=None):
    """Configure and return the appropriate model.

    If ``args.model == "EnergyEMLEMACE"`` this function constructs an
    EnergyEMLEMACE instance directly.  For all other model types it delegates
    to ``mace.tools.model_script_utils.configure_model``.

    Returns
    -------
    model : torch.nn.Module
    output_args : dict
    """
    if args.model != "EnergyEMLEMACE":
        return _mace_configure_model(
            args, train_loader, atomic_energies,
            model_foundation=model_foundation, heads=heads,
            z_table=z_table, head_configs=head_configs,
        )

    assert args.loss == "energy_emle", (
        "Use --loss energy_emle with EnergyEMLEMACE"
    )
    assert args.error_table == "EnergyEMLERMSE", (
        "Use --error_table EnergyEMLERMSE with EnergyEMLEMACE"
    )

    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": False,
        "dipoles": False,
        "emle": True,
        "polarizabilities": False,
    }
    logging.info("===========MODEL DETAILS===========")
    logging.info("Building EnergyEMLEMACE model")

    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for hc in head_configs:
                hc.std = 1.0
        logging.info("No scaling selected")

    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=mace_modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
        atomic_energies=atomic_energies,
        apply_cutoff=args.apply_cutoff,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        use_reduced_cg=args.use_reduced_cg,
        use_so3=args.use_so3,
        cueq_config=None,
    )

    model = EnergyEMLEMACE(
        **model_config,
        correlation=args.correlation,
        gate=mace_modules.gate_dict[args.gate],
        interaction_cls_first=mace_modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        MLP_irreps=o3.Irreps(args.MLP_irreps),
    )

    logging.info(
        f"Message passing with {args.num_channels} channels and max_L={args.max_L} "
        f"({args.hidden_irreps})"
    )
    logging.info(
        f"{args.num_interactions} layers, correlation order {args.correlation}, "
        f"spherical harmonics up to l={args.max_ell}"
    )
    logging.info(
        f"Radial cutoff: {args.r_max} Å  |  "
        f"{args.num_radial_basis} radial basis, {args.num_cutoff_basis} cutoff basis"
    )

    return model, output_args


def extract_config_emle_mace_model(model):
    """Extract constructor kwargs from a saved EnergyEMLEMACE model.

    Mirrors ``mace.tools.scripts_utils.extract_config_mace_model`` but returns
    only the keys that ``EnergyEMLEMACE.__init__`` accepts (in particular it
    omits ``atomic_inter_scale`` and ``atomic_inter_shift`` which belong to
    ``ScaleShiftMACE`` and are not parameters of ``EnergyEMLEMACE``).

    Returns a plain ``dict`` on success or ``{"error": "<message>"}`` on failure.
    """
    import torch
    import numpy as np

    model_name = getattr(model, "original_name", model.__class__.__name__)
    if not isinstance(model, EnergyEMLEMACE) and model_name != "EnergyEMLEMACE":
        return {"error": f"Model is not an EnergyEMLEMACE instance (got {model.__class__.__name__})"}

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    def radial_to_transform(radial):
        if not hasattr(radial, "distance_transform"):
            return None
        name = radial.distance_transform.__class__.__name__
        if name == "AgnesiTransform":
            return "Agnesi"
        if name == "SoftTransform":
            return "Soft"
        return name

    heads = model.heads if hasattr(model, "heads") else ["Default"]
    model_mlp_irreps = (
        o3.Irreps(str(model.readouts[-1].hidden_irreps))
        if model.num_interactions.item() > 1
        else 1
    )
    try:
        correlation = (
            len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = model.products[0].symmetric_contractions.contraction_degree

    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "edge_irreps": model.edge_irreps if hasattr(model, "edge_irreps") else None,
        "MLP_irreps": (
            o3.Irreps(f"{model_mlp_irreps.count((0, 1)) // len(heads)}x0e")
            if model.num_interactions.item() > 1
            else 1
        ),
        "gate": (
            model.readouts[-1]
            .equivariant_nonlin.act_scalars._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "use_reduced_cg": getattr(model, "use_reduced_cg", False),
        "use_so3": getattr(model, "use_so3", False),
        "use_agnostic_product": getattr(model, "use_agnostic_product", False),
        "use_last_readout_only": getattr(model, "use_last_readout_only", False),
        "use_embedding_readout": hasattr(model, "embedding_readout"),
        "cueq_config": getattr(model, "cueq_config", None),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": correlation,
        "radial_type": radial_to_name(model.radial_embedding.bessel_fn.__class__.__name__),
        "embedding_specs": getattr(model, "embedding_specs", None),
        "apply_cutoff": getattr(model, "apply_cutoff", True),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "pair_repulsion": hasattr(model, "pair_repulsion_fn"),
        "distance_transform": radial_to_transform(model.radial_embedding),
        "heads": heads,
    }
    return config


def get_emle_params_options(args, model):
    """Return optimizer param groups for EnergyEMLEMACE.

    Wraps mace's get_params_options and adds an extra param group for the
    learnable Thole damping parameter (a_Thole) and element-wise polarizability
    ratios (elements_alpha_v_ratios).
    """
    from mace.tools.scripts_utils import get_params_options

    param_options = get_params_options(args, model)

    if hasattr(model, "a_Thole"):
        param_options["params"].append(
            {
                "name": "emle",
                "params": [model.a_Thole, model.elements_alpha_v_ratios],
                "weight_decay": 0.0,
            }
        )
    return param_options