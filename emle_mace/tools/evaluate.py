"""EMLE-aware evaluate wrapper.

Wraps mace's ``evaluate`` function to also compute EMLE-specific RMSE metrics
(valence_widths/s, core_charges/q_core, charges/q, atomic_dipoles/mu,
polarizability/alpha) which are not tracked by mace's standard ``MACELoss``.
"""

import torch


def _compute_rmse(delta):
    """Root-mean-square error over all elements of a concatenated tensor."""
    if delta.numel() == 0:
        return float("nan")
    return delta.pow(2).mean().sqrt().item()


def make_emle_evaluate(original_evaluate):
    """Return a drop-in replacement for mace.tools.train.evaluate that adds EMLE metrics.

    Calls the original evaluate, then runs one extra pass over the data loader
    to collect EMLE-specific prediction errors.
    """

    def _emle_evaluate(model, loss_fn, data_loader, output_args, device):
        avg_loss, aux = original_evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )

        # Collect EMLE residuals over the full dataset.
        delta_s, s_ref = [], []
        delta_q_core, q_core_ref = [], []
        delta_q, q_ref = [], []
        delta_mu, mu_ref = [], []
        delta_alpha_triu, alpha_triu_ref = [], []

        for param in model.parameters():
            param.requires_grad = False

        try:
            for batch in data_loader:
                batch = batch.to(device)
                batch_dict = batch.to_dict()
                output = model(
                    batch_dict,
                    training=False,
                    compute_force=output_args.get("forces", True),
                    compute_virials=output_args.get("virials", False),
                    compute_stress=output_args.get("stress", False),
                )

                if output.get("valence_widths") is not None and hasattr(batch, "valence_widths") and batch.valence_widths is not None:
                    pred_s = output["valence_widths"]
                    ref_s = batch.valence_widths
                    delta_s.append((pred_s - ref_s).detach().cpu())
                    s_ref.append(ref_s.detach().cpu())

                if output.get("core_charges") is not None and hasattr(batch, "core_charges") and batch.core_charges is not None:
                    pred_qc = output["core_charges"]
                    ref_qc = batch.core_charges
                    delta_q_core.append((pred_qc - ref_qc).detach().cpu())
                    q_core_ref.append(ref_qc.detach().cpu())

                if output.get("charges") is not None and hasattr(batch, "charges") and batch.charges is not None:
                    pred_q = output["charges"]
                    ref_q = batch.charges
                    delta_q.append((pred_q - ref_q).detach().cpu())
                    q_ref.append(ref_q.detach().cpu())

                if output.get("atomic_dipoles") is not None and hasattr(batch, "atomic_dipoles") and batch.atomic_dipoles is not None:
                    pred_mu = output["atomic_dipoles"]
                    ref_mu = batch.atomic_dipoles
                    delta_mu.append((pred_mu - ref_mu).detach().cpu())
                    mu_ref.append(ref_mu.detach().cpu())

                # Polarizability: computed via Thole model from model outputs.
                ref_alpha = getattr(batch, "polarizability", None)
                if (
                    ref_alpha is not None
                    and output.get("valence_widths") is not None
                    and output.get("charges") is not None
                    and output.get("core_charges") is not None
                    and output.get("a_Thole") is not None
                    and output.get("alpha_v_ratios") is not None
                ):
                    try:
                        from emle_mace.loss import compute_molecular_polarizabilities
                        pred_alpha = compute_molecular_polarizabilities(batch, output)
                        triu_row, triu_col = torch.triu_indices(3, 3, offset=0)
                        pred_triu = pred_alpha[:, triu_row, triu_col].detach().cpu()
                        ref_triu = ref_alpha[:, triu_row, triu_col].detach().cpu()
                        delta_alpha_triu.append(pred_triu - ref_triu)
                        alpha_triu_ref.append(ref_triu)
                    except Exception:
                        pass
        finally:
            for param in model.parameters():
                param.requires_grad = True

        if delta_s:
            d = torch.cat(delta_s)
            r = torch.cat(s_ref)
            aux["rmse_emle_s"] = _compute_rmse(d)
            aux["rel_rmse_emle_s"] = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt() * 100).item() if r.pow(2).mean() > 0 else float("nan")

        if delta_q_core:
            d = torch.cat(delta_q_core)
            r = torch.cat(q_core_ref)
            aux["rmse_emle_q_core"] = _compute_rmse(d)
            aux["rel_rmse_emle_q_core"] = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt() * 100).item() if r.pow(2).mean() > 0 else float("nan")

        if delta_q:
            d = torch.cat(delta_q)
            r = torch.cat(q_ref)
            aux["rmse_emle_q"] = _compute_rmse(d)
            aux["rel_rmse_emle_q"] = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt() * 100).item() if r.pow(2).mean() > 0 else float("nan")

        if delta_mu:
            d = torch.cat(delta_mu)
            r = torch.cat(mu_ref)
            aux["rmse_emle_mu"] = _compute_rmse(d)
            aux["rel_rmse_emle_mu"] = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt() * 100).item() if r.pow(2).mean() > 0 else float("nan")

        if delta_alpha_triu:
            d = torch.cat(delta_alpha_triu)
            r = torch.cat(alpha_triu_ref)
            aux["rmse_emle_alpha"] = _compute_rmse(d)
            aux["rel_rmse_emle_alpha"] = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt() * 100).item() if r.pow(2).mean() > 0 else float("nan")

        return avg_loss, aux

    return _emle_evaluate
