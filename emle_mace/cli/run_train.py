###########################################################################################
# Training entry point for EnergyEMLEMACE
#
# Design: mace's run_train.run() contains ~1000 lines of well-tested infrastructure
# (distributed setup, data loading, checkpointing, SWA, etc.) that we want to reuse
# without modification.  Rather than copying that code, we:
#
#   1. Parse args with our extended parser (adds EnergyEMLEMACE / EnergyEMLERMSE /
#      energy_emle choices and EMLE loss-weight args).
#   2. Set the EMLE compute flags on the args namespace.
#   3. Monkey-patch name bindings inside mace modules that differ for the EMLE case:
#        - mace.cli.run_train.configure_model    → emle_mace configure_model
#        - mace.cli.run_train.get_loss_fn        → emle_mace get_loss_fn
#        - mace.cli.run_train.get_swa            → emle_mace get_swa
#        - mace.cli.run_train.create_error_table → emle_mace create_emle_error_table
#        - mace.tools.train.valid_err_log        → wrapper that routes EnergyEMLERMSE
#      (Python looks these up in the module's __dict__ at call time, so patching the
#      module attribute before calling run() is the correct interception point.)
#   4. Delegate to mace.cli.run_train.run(args).
###########################################################################################

import importlib
import logging

import mace.cli.run_train as _mace_run_train
# mace.tools.train is shadowed by the train() function in mace.tools.__init__,
# so we must import the module explicitly via importlib.
_mace_train_module = importlib.import_module("mace.tools.train")

from emle_mace.tools.arg_parser import build_emle_arg_parser
from emle_mace.tools.model_utils import configure_model as _emle_configure_model
from emle_mace.tools.scripts_utils import (
    get_loss_fn as _emle_get_loss_fn,
    get_swa as _emle_get_swa,
)
from emle_mace.tools.metrics import (
    create_emle_error_table as _emle_create_error_table,
    log_emle_errors as _emle_log_errors,
)
import emle_mace.data as _emle_data


def _make_valid_err_log_wrapper(original):
    """Return a drop-in replacement for mace.tools.train.valid_err_log.

    Intercepts calls with ``log_errors == "EnergyEMLERMSE"`` and routes them
    to the EMLE-specific logger; all other cases are forwarded to *original*.
    """
    def _wrapper(valid_loss, eval_metrics, logger, log_errors,
                 epoch=None, valid_loader_name="Default"):
        if log_errors == "EnergyEMLERMSE":
            _emle_log_errors(
                log_errors=log_errors,
                eval_metrics=eval_metrics,
                valid_loss=valid_loss,
                valid_loader_name=valid_loader_name,
            )
        else:
            original(valid_loss, eval_metrics, logger, log_errors, epoch, valid_loader_name)
    return _wrapper


def _patch_mace(args):
    """Monkey-patch mace's run_train module with EMLE-aware implementations.

    Patches the bindings that differ for the EnergyEMLEMACE case.
    Returns a dict of the original bindings so they can be restored.
    """
    originals = {
        "configure_model": _mace_run_train.configure_model,
        "get_loss_fn": _mace_run_train.get_loss_fn,
        "get_swa": _mace_run_train.get_swa,
        "create_error_table": _mace_run_train.create_error_table,
        "valid_err_log": _mace_train_module.valid_err_log,
    }

    _mace_run_train.configure_model = _emle_configure_model
    _mace_run_train.get_loss_fn = _emle_get_loss_fn
    _mace_run_train.get_swa = _emle_get_swa
    _mace_run_train.create_error_table = _emle_create_error_table
    _mace_train_module.valid_err_log = _make_valid_err_log_wrapper(originals["valid_err_log"])

    return originals


def _restore_mace(originals):
    """Restore the original mace function bindings."""
    _mace_run_train.configure_model = originals["configure_model"]
    _mace_run_train.get_loss_fn = originals["get_loss_fn"]
    _mace_run_train.get_swa = originals["get_swa"]
    _mace_run_train.create_error_table = originals["create_error_table"]
    _mace_train_module.valid_err_log = originals["valid_err_log"]


def run(args) -> None:
    """Run EnergyEMLEMACE training.

    Sets EMLE compute flags on args, patches mace's internal function references,
    then delegates to mace.cli.run_train.run().
    """
    # Set compute flags for EnergyEMLEMACE
    args.compute_emle = True
    args.compute_dipole = False
    args.compute_energy = True
    args.compute_forces = True
    args.compute_virials = False
    args.compute_stress = False
    args.compute_polarizability = False

    _emle_data.patch()
    originals = _patch_mace(args)
    try:
        _mace_run_train.run(args)
    finally:
        _restore_mace(originals)
        _emle_data.restore()


def main() -> None:
    """CLI entry point: emle-mace-train"""
    parser = build_emle_arg_parser()
    args = parser.parse_args()

    if args.model != "EnergyEMLEMACE":
        logging.warning(
            f"emle-mace-train is intended for EnergyEMLEMACE models; "
            f"got --model={args.model}.  Proceeding anyway via standard mace."
        )

    run(args)


if __name__ == "__main__":
    main()