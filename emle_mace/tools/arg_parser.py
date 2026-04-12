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

    return parser