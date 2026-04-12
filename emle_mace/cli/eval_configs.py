###########################################################################################
# Evaluation script for EnergyEMLEMACE models
#
# Runs inference and writes per-atom EMLE properties (valence_widths/s,
# core_charges/q_core, charges/q, atomic_dipoles/mu) alongside energy and forces
# to the output XYZ file.
###########################################################################################

import argparse

import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils

from emle_mace.models import EnergyEMLEMACE  # ensures class is importable for torch.load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="return per-body-order energy contributions",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_node_energies",
        help="return per-atom node energies",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces, stress and EMLE output keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Inject EnergyEMLEMACE into mace.modules.models so torch.load can unpickle it.
    import mace.modules.models as _mace_models_module
    if not hasattr(_mace_models_module, "EnergyEMLEMACE"):
        _mace_models_module.EnergyEMLEMACE = EnergyEMLEMACE

    model = torch.load(f=args.model, map_location=args.device, weights_only=False)
    model = model.to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    # Load configurations
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    # Apply EMLE data patch so from_config handles pbc tuple(numpy.bool_) correctly.
    import emle_mace.data as _emle_data
    _emle_data.patch()
    try:
        dataset = [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ]
    finally:
        _emle_data.restore()

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect outputs
    energies_list = []
    forces_collection = []
    stresses_list = []
    contributions_list = []
    node_energies_list = []
    emle_s_collection = []
    emle_q_core_collection = []
    emle_q_collection = []
    emle_mu_collection = []
    emle_alpha_list = []  # per-structure [3, 3]

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=args.compute_stress,
        )
        ptr = batch.ptr[1:].cpu()

        energies_list.append(torch_tools.to_numpy(output["energy"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=ptr,
            axis=0,
        )
        forces_collection.append(forces[:-1])

        if args.compute_stress and output.get("stress") is not None:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if args.return_contributions and output.get("contributions") is not None:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if args.return_node_energies and output.get("node_energy") is not None:
            node_energies_list.append(
                np.split(
                    torch_tools.to_numpy(output["node_energy"]),
                    indices_or_sections=ptr,
                    axis=0,
                )[:-1]
            )

        for key, collection in [
            ("valence_widths", emle_s_collection),
            ("core_charges", emle_q_core_collection),
            ("charges", emle_q_collection),
            ("atomic_dipoles", emle_mu_collection),
        ]:
            if output.get(key) is not None:
                split = np.split(
                    torch_tools.to_numpy(output[key]),
                    indices_or_sections=ptr,
                    axis=0,
                )
                collection.append(split[:-1])

        # Molecular polarizability via Thole model (per-structure, shape [n_mols, 3, 3]).
        if (
            output.get("a_Thole") is not None
            and output.get("alpha_v_ratios") is not None
            and output.get("valence_widths") is not None
            and output.get("charges") is not None
            and output.get("core_charges") is not None
        ):
            try:
                from emle_mace.loss import compute_molecular_polarizabilities
                alpha = compute_molecular_polarizabilities(batch, output)
                emle_alpha_list.extend(torch_tools.to_numpy(alpha))
            except Exception:
                pass

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [f for batch in forces_collection for f in batch]
    assert len(atoms_list) == len(energies) == len(forces_list)

    if args.compute_stress and stresses_list:
        stresses = np.concatenate(stresses_list, axis=0)

    if args.return_contributions and contributions_list:
        contributions = np.concatenate(contributions_list, axis=0)

    if args.return_node_energies and node_energies_list:
        node_energies = [ne for batch in node_energies_list for ne in batch]

    emle_s = [s for batch in emle_s_collection for s in batch]
    emle_q_core = [q for batch in emle_q_core_collection for q in batch]
    emle_q = [q for batch in emle_q_collection for q in batch]
    emle_mu = [m for batch in emle_mu_collection for m in batch]

    # Store results in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces

        if args.compute_stress and stresses_list:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.return_contributions and contributions_list:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if args.return_node_energies and node_energies_list:
            atoms.arrays[args.info_prefix + "node_energies"] = node_energies[i]

        if emle_s:
            atoms.arrays[args.info_prefix + "s"] = emle_s[i]
        if emle_q_core:
            atoms.arrays[args.info_prefix + "q_core"] = emle_q_core[i]
        if emle_q:
            atoms.arrays[args.info_prefix + "q"] = emle_q[i]
        if emle_mu:
            atoms.arrays[args.info_prefix + "mu"] = emle_mu[i]
        if emle_alpha_list:
            atoms.info[args.info_prefix + "alpha"] = emle_alpha_list[i]

    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
