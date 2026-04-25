###########################################################################################
# Gate-based non-linear readout block for EnergyEMLEMACE.
#
# The default mace.modules.blocks.NonLinearReadoutBlock uses e3nn.nn.Activation, which is
# scalar-only. When the readout's output contains an l=1 irrep (the atomic dipole), the
# inner MLP_irreps must remain scalar-only and equivariance silently zeroes every weight
# that would couple the scalar MLP to the l=1 output. This block uses e3nn.nn.Gate instead,
# preserving equivariance while keeping a real (non-zero) parameter path to the dipole
# output.
###########################################################################################

from typing import Any, Callable, Dict, Optional

import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    Linear,
    OEQConfig,
)


@compile_mode("script")
class EMLENonLinearReadoutBlock(torch.nn.Module):
    """Gate-based non-linear readout for EnergyEMLEMACE (default output: 4x0e + 1x1o)."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        irreps_out: o3.Irreps = o3.Irreps("4x0e + 1x1o"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)

        # Auto-augment scalar-only MLP_irreps with matching l=1 channels so the Gate has
        # gated lanes to operate on (the CLI default --MLP_irreps "16x0e" is scalar-only).
        MLP_irreps = o3.Irreps(MLP_irreps)
        if all(ir.l == 0 for _, ir in MLP_irreps) and any(
            ir.l > 0 for _, ir in self.irreps_out
        ):
            scalar_mul = sum(mul for mul, ir in MLP_irreps if ir.l == 0)
            extra = o3.Irreps(
                [(scalar_mul, ir) for _, ir in self.irreps_out if ir.l > 0]
            )
            MLP_irreps = (MLP_irreps + extra).simplify()
        self.hidden_irreps = MLP_irreps

        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=cueq_config,
        )
        self.linear_2 = Linear(
            irreps_in=self.equivariant_nonlin.irreps_out,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps_out.dim]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)
