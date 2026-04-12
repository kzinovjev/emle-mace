"""CLI entry point wrapper for emle-mace-eval.

Sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before any torch/e3nn imports so that
e3nn's constants.pt (which contains a slice object) can be loaded under
PyTorch >= 2.6, which changed the default of torch.load's weights_only to True.
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    # Deferred import: the env var must be set before e3nn is imported.
    from emle_mace.cli.eval_configs import main as _main
    _main()
