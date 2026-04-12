"""End-to-end training test for EnergyEMLEMACE.

Writes a small synthetic dataset (water molecules with EMLE labels) to a
temp directory, runs a 3-epoch training pass via emle_mace.cli.run_train.run(),
and checks that a model file is saved.
"""

import os

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from emle_mace.cli.run_train import run as emle_run
from emle_mace.tools.arg_parser import build_emle_arg_parser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(name="emle_configs")
def fixture_emle_configs():
    """Small synthetic dataset: isolated atoms + perturbed water molecules."""
    configs = []

    # Isolated atom references (needed for E0s)
    for z, e0 in [(8, -2000.0), (1, -500.0)]:
        a = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=[True]*3)
        a.info["REF_energy"] = e0
        a.info["config_type"] = "IsolatedAtom"
        configs.append(a)

    rng = np.random.default_rng(42)
    water_base = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        cell=[8, 8, 8],
        pbc=[True]*3,
    )
    for _ in range(10):
        c = water_base.copy()
        c.positions += rng.normal(0, 0.05, c.positions.shape)
        n = len(c)

        c.info["REF_energy"] = float(rng.normal(-2500, 10))
        c.new_array("REF_forces", rng.normal(0, 0.5, (n, 3)))
        c.new_array("REF_valence_widths", np.abs(rng.normal(0.55, 0.05, n)))
        c.new_array("REF_core_charges", rng.normal(0, 1.0, n))
        # Ensure charges sum to zero
        q = rng.normal(0, 0.5, n)
        q -= q.mean()
        c.new_array("REF_charges", q)
        c.new_array("REF_atomic_dipoles", rng.normal(0, 0.1, (n, 3)))
        # 3x3 symmetric polarizability stored as flat 9-vector in atoms.info
        # (polarizability is a per-structure property, not per-atom)
        alpha = np.eye(3) * rng.uniform(4, 8)
        c.info["REF_polarizability"] = alpha.flatten().tolist()
        c.info["REF_total_charge"] = 0.0

        configs.append(c)

    return configs


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_emle_mace_train(tmp_path, emle_configs):
    """A minimal EnergyEMLEMACE training run should complete and save a model."""
    xyz_path = tmp_path / "emle_fit.xyz"
    ase.io.write(str(xyz_path), emle_configs)

    # Build args from the full parser (gets all mace defaults), then override.
    parser = build_emle_arg_parser()
    args = parser.parse_args([
        "--name", "EMLEMACETest",
        "--train_file", str(xyz_path),
        "--model", "EnergyEMLEMACE",
        "--loss", "energy_emle",
        "--error_table", "EnergyEMLERMSE",
    ])

    # Override defaults for a fast test run.
    args.valid_fraction = 0.2
    args.log_dir = str(tmp_path)
    args.checkpoints_dir = str(tmp_path)
    args.model_dir = str(tmp_path)
    args.results_dir = str(tmp_path)
    args.work_dir = str(tmp_path)
    args.r_max = 3.5
    args.num_radial_basis = 6
    args.num_cutoff_basis = 5
    args.max_ell = 2
    args.interaction = "RealAgnosticResidualInteractionBlock"
    args.interaction_first = "RealAgnosticInteractionBlock"
    args.num_interactions = 2
    args.num_channels = 8
    args.max_L = 1
    args.hidden_irreps = "8x0e + 8x1o"
    args.MLP_irreps = "8x0e"
    args.correlation = 3
    args.radial_MLP = "[16, 16, 16]"
    args.energy_key = "REF_energy"
    args.forces_key = "REF_forces"
    args.valence_widths_key = "REF_valence_widths"
    args.core_charges_key = "REF_core_charges"
    args.charges_key = "REF_charges"
    args.atomic_dipoles_key = "REF_atomic_dipoles"
    args.polarizability_key = "REF_polarizability"
    args.total_charge_key = "REF_total_charge"
    args.virials_key = ""
    args.stress_key = ""
    args.dipole_key = ""
    args.seed = 42
    args.device = "cpu"
    args.default_dtype = "float64"
    args.batch_size = 4
    args.valid_batch_size = 4
    args.max_num_epochs = 3
    args.patience = 10
    args.eval_interval = 1
    args.lr = 0.01
    args.weight_decay = 0.0
    args.amsgrad = False
    args.optimizer = "adam"
    args.scheduler = "ReduceLROnPlateau"
    args.lr_factor = 0.8
    args.scheduler_patience = 2
    args.scaling = "std_scaling"
    args.E0s = "average"
    args.avg_num_neighbors = None
    args.num_workers = 0
    args.pin_memory = False
    args.swa = False
    args.ema = False
    args.wandb = False
    args.distributed = False
    args.plot = False
    args.save_all_checkpoints = False
    args.restart_latest = False
    args.keep_checkpoints = False
    args.save_cpu = False
    args.multiheads_finetuning = False
    args.foundation_model = None
    args.lbfgs = False

    emle_run(args)

    model_file = tmp_path / "EMLEMACETest.model"
    assert model_file.exists(), f"Model file not found at {model_file}"

    # Check we can reload the model
    model = torch.load(str(model_file), map_location="cpu")
    assert hasattr(model, "a_Thole"), "Reloaded model missing a_Thole"