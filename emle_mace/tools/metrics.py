"""EMLE-specific metrics: error-table construction and per-epoch logging.

These are the EnergyEMLERMSE counterparts of the helpers in
mace.tools.tables_utils and mace.tools.train, extracted here so that standard
mace can be used as an unmodified dependency.
"""

import logging
from typing import Any, Dict

from prettytable import PrettyTable


def create_emle_error_table(
    table_type: str,
    all_data_loaders,
    model,
    loss_fn,
    output_args: Dict[str, bool],
    device,
    distributed: bool = False,
    **kwargs,
) -> PrettyTable:
    """Build a PrettyTable of EMLE validation metrics.

    For ``table_type == "EnergyEMLERMSE"`` this produces a custom table with
    EMLE columns.  For all other table types it delegates to
    ``mace.tools.tables_utils.create_error_table``.
    """
    from mace.tools.tables_utils import create_error_table as _mace_create_error_table

    if table_type != "EnergyEMLERMSE":
        return _mace_create_error_table(
            table_type=table_type,
            all_data_loaders=all_data_loaders,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            device=device,
            distributed=distributed,
            **kwargs,
        )

    from mace.tools.train import evaluate

    table = PrettyTable()
    table.field_names = [
        "config_type",
        "RMSE E / meV / atom",
        "RMSE F / meV / Å",
        "rel F RMSE %",
        "RMSE s [Bohr]",
        "rel RMSE s %",
        "RMSE q_core [e]",
        "rel RMSE q_core %",
        "RMSE q [e]",
        "rel RMSE q %",
        "RMSE mu [e·Bohr]",
        "rel RMSE mu %",
        "RMSE alpha [Bohr³]",
        "rel RMSE alpha %",
    ]

    def _custom_key(name):
        if name == "train":
            return "aaaa"
        if name == "valid":
            return "aaab"
        return name

    for name in sorted(all_data_loaders, key=_custom_key):
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            import torch.distributed
            torch.distributed.barrier()

        table.add_row(
            [
                name,
                f"{metrics.get('rmse_e_per_atom', float('nan')) * 1000:8.1f}",
                f"{metrics.get('rmse_f', float('nan')) * 1000:8.1f}",
                f"{metrics.get('rel_rmse_f', float('nan')):8.1f}",
                f"{metrics.get('rmse_emle_s', float('nan')):8.4f}",
                f"{metrics.get('rel_rmse_emle_s', float('nan')):8.1f}",
                f"{metrics.get('rmse_emle_q_core', float('nan')):8.4f}",
                f"{metrics.get('rel_rmse_emle_q_core', float('nan')):8.1f}",
                f"{metrics.get('rmse_emle_q', float('nan')):8.4f}",
                f"{metrics.get('rel_rmse_emle_q', float('nan')):8.1f}",
                f"{metrics.get('rmse_emle_mu', float('nan')):8.4f}",
                f"{metrics.get('rel_rmse_emle_mu', float('nan')):8.1f}",
                f"{metrics.get('rmse_emle_alpha', float('nan')):8.4f}",
                f"{metrics.get('rel_rmse_emle_alpha', float('nan')):8.1f}",
            ]
        )

    return table


def log_emle_errors(
    log_errors: str,
    eval_metrics: Dict[str, Any],
    valid_loss: float,
    valid_loader_name: str,
    initial_phrase: str = "Validation",
) -> None:
    """Log per-epoch EMLE validation metrics.

    For ``log_errors == "EnergyEMLERMSE"`` this emits a detailed EMLE line.
    For all other values it calls ``mace.tools.train.log_error_table_entry``
    (or equivalent) from mace.
    """
    if log_errors != "EnergyEMLERMSE":
        # Non-EMLE modes are handled by the valid_err_log wrapper in cli/run_train.py.
        # This branch should not be reached when patching is active.
        raise ValueError(
            f"log_emle_errors called with unsupported log_errors={log_errors!r}. "
            "Only 'EnergyEMLERMSE' is handled here."
        )

    error_e = eval_metrics.get("rmse_e_per_atom", float("nan")) * 1e3
    error_f = eval_metrics.get("rmse_f", float("nan")) * 1e3
    error_s = eval_metrics.get("rmse_emle_s", float("nan"))
    error_q_core = eval_metrics.get("rmse_emle_q_core", float("nan"))
    error_q = eval_metrics.get("rmse_emle_q", float("nan"))
    error_mu = eval_metrics.get("rmse_emle_mu", float("nan"))
    error_alpha = eval_metrics.get("rmse_emle_alpha", float("nan"))

    logging.info(
        f"{initial_phrase}: head: {valid_loader_name}, "
        f"loss={valid_loss:8.8f}, "
        f"RMSE_E_per_atom={error_e:8.2f} meV, "
        f"RMSE_F={error_f:8.2f} meV/Å, "
        f"RMSE_s={error_s:7.4f} Bohr, "
        f"RMSE_q_core={error_q_core:7.4f} e, "
        f"RMSE_q={error_q:7.4f} e, "
        f"RMSE_mu={error_mu:7.4f} e·Bohr, "
        f"RMSE_alpha={error_alpha:7.3f} Bohr³"
    )