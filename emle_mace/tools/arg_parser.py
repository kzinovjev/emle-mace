"""Argument parser for emle-mace training.

Wraps mace's default parser and adds EnergyEMLEMACE-specific choices and arguments.
"""

import argparse

from mace.tools.arg_parser import build_default_arg_parser


def build_emle_arg_parser() -> argparse.ArgumentParser:
    """Return an argument parser that extends mace's default parser with EMLE options.

    Changes vs the standard mace parser:
      - Adds "EnergyEMLEMACE" to --model choices
      - Adds "EnergyEMLERMSE" to --error_table choices
      - Adds "energy_emle" to --loss choices
      - Adds five EMLE loss-weight arguments
    """
    parser = build_default_arg_parser()

    # ------------------------------------------------------------------ model
    # Patch the --model argument to include EnergyEMLEMACE.
    # argparse does not support in-place mutation of choices, so we rebuild it.
    for action in parser._actions:
        if action.dest == "model":
            if "EnergyEMLEMACE" not in action.choices:
                action.choices.append("EnergyEMLEMACE")
            action.default = "EnergyEMLEMACE"
            break

    # ------------------------------------------------------------ error_table
    for action in parser._actions:
        if action.dest == "error_table":
            if "EnergyEMLERMSE" not in action.choices:
                action.choices.append("EnergyEMLERMSE")
            break

    # ------------------------------------------------------------------- loss
    for action in parser._actions:
        if action.dest == "loss":
            if "energy_emle" not in action.choices:
                action.choices.append("energy_emle")
            break

    # ------------------------------------------------ EMLE data-key arguments
    parser.add_argument(
        "--valence_widths_key",
        help="key for per-atom valence widths (s) in the data files",
        type=str,
        default="s",
    )
    parser.add_argument(
        "--core_charges_key",
        help="key for per-atom core charges (q_core) in the data files",
        type=str,
        default="q_core",
    )
    parser.add_argument(
        "--atomic_dipoles_key",
        help="key for per-atom atomic dipoles (mu) in the data files",
        type=str,
        default="mu",
    )
    parser.add_argument(
        "--atomic_quadrupoles_key",
        help=(
            "key for per-atom atomic quadrupoles (theta) in the data files; "
            "the 6 components of the symmetric traceless Cartesian quadrupole "
            "in order [xx, xy, xz, yy, yz, zz]"
        ),
        type=str,
        default="theta",
    )

    # ------------------------------------------------ EMLE loss-weight arguments
    parser.add_argument(
        "--valence_widths_weight",
        help="weight of valence widths (s) loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--core_charges_weight",
        help="weight of core charges (q_core) loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--charges_weight",
        help="weight of total charges (q) loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--atomic_dipoles_weight",
        help="weight of atomic dipoles (mu) loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--atomic_quadrupoles_weight",
        help="weight of atomic quadrupoles (theta) loss",
        type=float,
        default=1.0,
    )
    # polarizability_weight may already exist from the base parser (it is used by
    # AtomicDielectricMACE); only add it if not already present.
    existing_dests = {a.dest for a in parser._actions}
    if "polarizability_weight" not in existing_dests:
        parser.add_argument(
            "--polarizability_weight",
            help="weight of molecular polarizability (alpha) loss",
            type=float,
            default=10.0,
        )

    # ---------------------------------------- flexible polarizability (k_alpha)
    parser.add_argument(
        "--use_flexible_alpha",
        help="predict a per-atom, environment-dependent polarizability "
        "correction k_alpha (decouples alpha from the valence width s)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--k_alpha_reg_weight",
        help="weight of the (sqrt(k_alpha) - 1)^2 regularization keeping the "
        "flexible polarizability correction near unity; 0.0 disables it",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--q_core_fixed",
        help="Per-element constant core charges (dataset means), ordered by z_table. "
        "Removes the q_core readout head; set core_charges_weight=0 with this.",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--k_alpha_cap_lo",
        help="Lower bound on k_alpha; used with --k_alpha_cap_hi as a two-sided bound.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--k_alpha_cap_hi",
        help="Smooth upper bound on k_alpha (must be > 1); k_alpha == 1 at zero "
        "readout. 0.0 (default) disables the cap.",
        type=float,
        default=0.0,
    )

    return parser