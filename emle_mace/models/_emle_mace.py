###########################################################################################
# EnergyEMLEMACE model
# Implements a MACE-based model that jointly predicts energies/forces and EMLE properties
# (valence widths, core/valence charges, atomic dipoles, molecular polarizability).
# Originally developed as part of the emle-mace MACE fork; extracted here so that standard
# mace can be used as an unmodified dependency.
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.util.jit import compile_mode

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    RadialEmbeddingBlock,
)
from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_mean, scatter_sum
from mace.modules.utils import (
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)

from ._readouts import EMLENonLinearReadoutBlock

# Per-readout output irreps are built per-instance from the max_static_L /
# use_flexible_alpha settings (see __init__): base 4 scalars (energy, valence_widths,
# core_charges, charges), an optional 5th scalar (sqrt(k_alpha) deviation, flexible
# polarizability), an l=1 odd vector (atomic dipole, max_static_L >= 1) and an l=2
# even tensor (atomic quadrupole, traceless symmetric Cartesian in the e3nn 2e
# basis, max_static_L == 2).


@compile_mode("script")
class EnergyEMLEMACE(torch.nn.Module):
    """MACE model that jointly predicts energies/forces and EMLE embedding properties.

    In addition to the standard MACE energy and forces, this model outputs per-atom:
      - valence_widths      (s)
      - core_charges        (q_core)
      - charges             (q, total = core + valence)
      - atomic_dipoles      (mu, l=1; only when max_static_L >= 1)
      - atomic_quadrupoles  (theta, l=2 traceless symmetric Cartesian 3x3;
                             only when max_static_L == 2)

    and global per-graph:
      - a_Thole         (learnable Thole damping parameter)
      - alpha_v_ratios  (per-element polarizability scaling ratios)

    These outputs feed directly into the EMLE QM/MM embedding framework
    (MACEEMLEJoint in emle-engine).
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        max_static_L: int = 0,
        use_flexible_alpha: bool = False,
        k_alpha_cap_lo: float = 0.0,
        k_alpha_cap_hi: float = 0.0,
        q_cap_lo: Optional[List[float]] = None,
        q_cap_hi: Optional[List[float]] = None,
        q_core_cap_lo: Optional[List[float]] = None,
        q_core_cap_hi: Optional[List[float]] = None,
        s_cap_lo: Optional[List[float]] = None,
        s_cap_hi: Optional[List[float]] = None,
        q_core_fixed: Optional[List[float]] = None,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only

        # Flexible (environment-dependent) polarizability: when enabled, each
        # readout emits one extra per-atom scalar (log k_alpha) so the atomic
        # polarizability ratio can vary per environment instead of being fixed
        # per element. See DESIGN.md / analysis/flexpol.
        self.use_flexible_alpha = use_flexible_alpha
        # Optional smooth two-sided bound on k_alpha: lo <= k_alpha <= hi, with
        #     k_alpha = lo + (hi-lo)*sigmoid(a*z + z0)
        # z0 chosen so k_alpha == 1 at zero readout (the value the
        # (sqrt(k_alpha)-1)^2 regularizer pulls toward), and a chosen for UNIT
        # GRADIENT there, so in-range behaviour is natural and both bounds are
        # unreachable asymptotes. hi = 0.0 (default) disables the cap, exact
        # backward compatibility; lo = 0 gives a one-sided upper bound.
        self.k_alpha_cap_lo = float(k_alpha_cap_lo)
        self.k_alpha_cap_hi = float(k_alpha_cap_hi)
        _z0 = 0.0
        _a = 1.0
        if self.k_alpha_cap_hi > 0.0:
            if self.k_alpha_cap_hi <= 1.0:
                raise ValueError("k_alpha_cap_hi must be > 1 (k_alpha=1 must be attainable)")
            if not (0.0 <= self.k_alpha_cap_lo < 1.0):
                raise ValueError("k_alpha_cap_lo must satisfy 0 <= lo < 1")
            import math as _math
            _w = self.k_alpha_cap_hi - self.k_alpha_cap_lo
            _p = (1.0 - self.k_alpha_cap_lo) / _w          # sigmoid value giving k_alpha=1
            _z0 = _math.log(_p / (1.0 - _p))
            _a = 1.0 / (_w * _p * (1.0 - _p))              # unit gradient at k_alpha=1
        self.k_alpha_cap_z0 = _z0
        self.k_alpha_cap_a = _a

        # Per-element physical caps on the MBIS property heads (q, q_core, s).
        # Bounds are [dataset_min - margin, dataset_max + margin] per element, so
        # the heads are bounded BY CONSTRUCTION and a runaway is impossible.
        # Mapping (per atom, with lo/hi gathered from node_attrs):
        #     x = lo + (hi-lo) * sigmoid(4*(x_raw - mid)/(hi-lo)),  mid=(lo+hi)/2
        # which has UNIT GRADIENT at the midpoint, so in-range behaviour matches
        # the uncapped head and saturation is smooth (C-infinity) at the bounds.
        # Applied to the raw head outputs BEFORE the total-charge correction; the
        # correction subtracts a per-molecule constant built from capped values,
        # so the composition stays bounded.
        _ne = num_elements
        def _capbuf(v, name):
            if v is None:
                return torch.zeros(0)
            t = torch.tensor(v, dtype=torch.get_default_dtype())
            assert t.numel() == _ne, f"{name} must have one entry per element ({_ne})"
            return t
        # q_core as a per-element CONSTANT (dataset mean), as in the reference GPR
        # EMLE model (Zinovjev 2023, Table 1: "average MBIS values over training
        # set"). MBIS core charges are essentially element-determined -- natural
        # spread is std ~0.006-0.009 e (H exactly 1.0) -- so a learned head adds a
        # large extrapolation error (up to 0.8 e off-manifold) for no benefit.
        self.register_buffer(
            "q_core_fixed",
            torch.tensor(q_core_fixed, dtype=torch.get_default_dtype())
            if q_core_fixed is not None
            else torch.zeros(0),
        )
        # Each property cap is INDEPENDENT: pass lo+hi to enable, omit both to
        # disable that property's cap.
        for _nm, _l, _h in (
            ("q_cap", q_cap_lo, q_cap_hi),
            ("q_core_cap", q_core_cap_lo, q_core_cap_hi),
            ("s_cap", s_cap_lo, s_cap_hi),
        ):
            if (_l is None) != (_h is None):
                raise ValueError(f"{_nm}_lo and {_nm}_hi must be given together")
        self.use_q_cap = q_cap_lo is not None
        self.use_q_core_cap = q_core_cap_lo is not None
        self.use_s_cap = s_cap_lo is not None
        self.register_buffer("q_cap_lo", _capbuf(q_cap_lo, "q_cap_lo"))
        self.register_buffer("q_cap_hi", _capbuf(q_cap_hi, "q_cap_hi"))
        self.register_buffer("q_core_cap_lo", _capbuf(q_core_cap_lo, "q_core_cap_lo"))
        self.register_buffer("q_core_cap_hi", _capbuf(q_core_cap_hi, "q_core_cap_hi"))
        self.register_buffer("s_cap_lo", _capbuf(s_cap_lo, "s_cap_lo"))
        self.register_buffer("s_cap_hi", _capbuf(s_cap_hi, "s_cap_hi"))
        # Highest static multipole order: 0 = charges only, 1 = +dipoles,
        # 2 = +quadrupoles.
        if max_static_L not in (0, 1, 2):
            raise ValueError("max_static_L must be 0 (q), 1 (q+mu) or 2 (q+mu+theta)")
        self.max_static_L = int(max_static_L)
        # q_core as a per-element CONSTANT (reference GPR EMLE, Zinovjev 2023 Table 1:
        # "average MBIS values over training set"). When enabled the q_core READOUT
        # HEAD IS REMOVED ENTIRELY -- MBIS core charges are element-determined
        # (std ~0.006 e; H exactly 1.0), so a learned head only adds extrapolation
        # error (up to 0.8 e off-manifold). Set core_charges_weight=0 to match.
        self.use_fixed_q_core = q_core_fixed is not None
        # Scalar readout layout: energy, valence_widths, [q_core], charges, [k_alpha]
        n_emle_scalars = 3
        if not self.use_fixed_q_core:
            n_emle_scalars += 1
        if use_flexible_alpha:
            n_emle_scalars += 1
        # Column indices (-1 = head absent), precomputed for TorchScript.
        _i = 2
        self._col_q_core = -1
        if not self.use_fixed_q_core:
            self._col_q_core = _i
            _i += 1
        self._col_charges = _i
        _i += 1
        self._col_k_alpha = -1
        if use_flexible_alpha:
            self._col_k_alpha = _i
            _i += 1
        self._col_mu = n_emle_scalars if self.max_static_L >= 1 else -1
        self._col_theta = n_emle_scalars + 3 if self.max_static_L >= 2 else -1
        _irreps_str = f"{n_emle_scalars}x0e"
        if self.max_static_L >= 1:
            _irreps_str += " + 1x1o"
        if self.max_static_L >= 2:
            _irreps_str += " + 1x2e"
        self._emle_readout_irreps = o3.Irreps(_irreps_str)

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        embedding_size = node_feats_irreps.count(o3.Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    self._emle_readout_irreps,
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                max_readout_l = self.max_static_L
                if max_readout_l >= 1:
                    assert any(
                        ir.l == 1 for _, ir in hidden_irreps
                    ), "To predict dipoles use at least l=1 hidden_irreps"
                if max_readout_l >= 2:
                    assert any(
                        ir.l == 2 for _, ir in hidden_irreps
                    ), "To predict quadrupoles, hidden_irreps must contain an l=2 (2e) irrep"
                # Last layer feeds the multipole readouts up to max_static_L.
                hidden_irreps_out = str(
                    o3.Irreps(
                        [(mul, ir) for mul, ir in hidden_irreps if ir.l <= max_readout_l]
                    )
                )
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    EMLENonLinearReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        irreps_out=self._emle_readout_irreps,
                        cueq_config=cueq_config,
                        oeq_config=oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        self._emle_readout_irreps,
                        cueq_config,
                        oeq_config,
                    )
                )
            self.a_Thole = torch.nn.Parameter(torch.tensor(2., dtype=torch.get_default_dtype()))
            self.elements_alpha_v_ratios = torch.nn.Parameter(
                torch.ones(num_elements, dtype=torch.get_default_dtype()) * 0.1
            )

        # Change-of-basis from the readout's l=2 output (5 e3nn 2e components) to a
        # symmetric traceless 3x3 Cartesian quadrupole. CartesianTensor("ij=ji")
        # decomposes as 1x0e + 1x2e; row 0 is the trace, which we drop.
        _ct = CartesianTensor("ij=ji")
        _basis = _ct.to_cartesian(torch.eye(6, dtype=torch.float64))  # [6, 3, 3]
        self.register_buffer(
            "quad_to_cartesian",
            _basis[1:].to(torch.get_default_dtype()),  # [5, 3, 3]
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        if training:
            self.elements_alpha_v_ratios.requires_grad_(True)

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

        node_valence_widths_list = []
        node_core_charges_list = []
        node_charges_list = []
        node_atomic_dipoles_list: List[torch.Tensor] = []
        node_atomic_quadrupoles_list: List[torch.Tensor] = []
        node_k_alpha_sqrt_dev_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i

            node_out = readout(node_feats_concat[feat_idx], node_heads)

            node_es = node_out[num_atoms_arange, 0]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

            # Column layout: 0 energy, 1 valence_widths, [q_core], charges, [k_alpha],
            # then 3 dipole (1x1o), then 5 quadrupole (1x2e). q_core is absent when it
            # is a fixed per-element constant. Indices precomputed in __init__.
            node_valence_widths_list.append(node_out[num_atoms_arange, 1])
            if self._col_q_core >= 0:
                node_core_charges_list.append(
                    node_out[num_atoms_arange, self._col_q_core]
                )
            node_charges_list.append(node_out[num_atoms_arange, self._col_charges])
            if self._col_k_alpha >= 0:
                node_k_alpha_sqrt_dev_list.append(
                    node_out[num_atoms_arange, self._col_k_alpha]
                )
            if self._col_mu >= 0:
                node_atomic_dipoles_list.append(
                    node_out[num_atoms_arange, self._col_mu : self._col_mu + 3]
                )
            if self._col_theta >= 0:
                node_atomic_quadrupoles_list.append(
                    node_out[num_atoms_arange, self._col_theta : self._col_theta + 5]
                )

        contributions = torch.stack(energies, dim=-1)
        interaction_energy = torch.sum(contributions[:, 1:], dim=-1)
        total_energy = e0 + interaction_energy  # [n_graphs, ]
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

        contributions_valence_widths = torch.stack(node_valence_widths_list, dim=-1)
        contributions_charges = torch.stack(node_charges_list, dim=-1)
        valence_widths = torch.sum(contributions_valence_widths, dim=-1)  # [n_nodes]
        if self.use_fixed_q_core:
            # Element-determined constant; no readout head exists for it.
            core_charges = data["node_attrs"] @ self.q_core_fixed
        else:
            core_charges = torch.sum(
                torch.stack(node_core_charges_list, dim=-1), dim=-1
            )  # [n_nodes]
        charges = torch.sum(contributions_charges, dim=-1)  # [n_nodes]

        # Physical per-element caps on the property heads (see __init__); each
        # property's cap is applied independently, iff its lo/hi were provided.
        # NOTE: the raw head output is ~0 for an untrained model, so the
        # sigmoid argument must NOT subtract the physical midpoint -- doing so
        # pins s (mid~0.39) and q_core (mid~6.4) at their lower bound with a
        # vanishing gradient, and they never learn. Mapping raw=0 -> midpoint
        # (unit gradient there) makes the head predict a DEVIATION from the
        # middle of the allowed band, which trains normally.
        na = data["node_attrs"]
        if self.use_s_cap:
            _lo = na @ self.s_cap_lo
            _hi = na @ self.s_cap_hi
            _w = _hi - _lo
            valence_widths = _lo + _w * torch.sigmoid(4.0 * valence_widths / _w)
        if self.use_q_core_cap and not self.use_fixed_q_core:
            _lo = na @ self.q_core_cap_lo
            _hi = na @ self.q_core_cap_hi
            _w = _hi - _lo
            core_charges = _lo + _w * torch.sigmoid(4.0 * core_charges / _w)
        if self.use_q_cap:
            _lo = na @ self.q_cap_lo
            _hi = na @ self.q_cap_hi
            _w = _hi - _lo
            charges = _lo + _w * torch.sigmoid(4.0 * charges / _w)

        # Correct total charge per graph to match the target total charge
        num_atoms = (data["ptr"][1:] - data["ptr"][:-1]).to(charges)
        total_charge_excess = scatter_mean(
            src=charges, index=data["batch"], dim=0, dim_size=num_graphs
        ) - (data["total_charge"] / num_atoms)
        charges = charges - total_charge_excess[data["batch"]]

        # Static multipoles above max_static_L have no heads -> None in the output.
        atomic_dipoles: Optional[torch.Tensor] = None
        if self.max_static_L >= 1:
            atomic_dipoles = torch.sum(
                torch.stack(node_atomic_dipoles_list, dim=-1), dim=-1
            )  # [n_nodes, 3]

        # Sum the l=2 contributions, then map the 5 e3nn 2e components to a
        # symmetric traceless 3x3 Cartesian quadrupole via the fixed change-of-basis.
        atomic_quadrupoles: Optional[torch.Tensor] = None
        if self.max_static_L >= 2:
            atomic_quadrupoles_2e = torch.sum(
                torch.stack(node_atomic_quadrupoles_list, dim=-1), dim=-1
            )  # [n_nodes, 5]
            atomic_quadrupoles = torch.einsum(
                "kij,nk->nij", self.quad_to_cartesian, atomic_quadrupoles_2e
            )  # [n_nodes, 3, 3]

        # Per-atom polarizability correction k_alpha (flexible mode). This mirrors
        # the GPR flexible model (emle alpha_mode="reference"), which predicts a
        # per-atom sqrt(k) and squares it for positivity, regularizing sqrt(k)->1.
        #
        # UNCAPPED (k_alpha_cap_hi == 0, default): the readout predicts the per-atom
        # deviation of sqrt(k_alpha) from 1 (summed over layers), so:
        #   k_alpha_sqrt = 1 + sum_layers(dev)      (== 1 when readout output is 0)
        #   k_alpha      = k_alpha_sqrt**2 > 0       (== 1 -> recovers fixed alpha)
        #
        # CAPPED (k_alpha_cap_hi > 1): the readout sum z is mapped logistically onto
        # (lo, hi), and k_alpha_sqrt is derived FROM the capped k_alpha:
        #   k_alpha      = lo + (hi-lo) * sigmoid(a*z + z0)          (==1 at z=0)
        #   k_alpha_sqrt = sqrt(k_alpha)
        # Deriving k_alpha_sqrt from the capped value is deliberate: the
        # (k_alpha_sqrt - 1)^2 regularization must act on the POST-cap k_alpha,
        # not on the raw readout, so that the penalty always refers to the value
        # the induction model actually uses.
        # The multiplicative correction acts on the volume-based ratio
        # (alpha_v_ratios) so atomic alpha stays proportional to the MBIS volume.
        # Fixed mode: k_alpha = k_alpha_sqrt = 1 for every atom (exact backward
        # compatibility). k_alpha_sqrt is exposed so the loss can regularize it
        # toward 1 exactly as the GPR training regularizes ref_values_sqrtk.
        if self.use_flexible_alpha:
            _z = torch.sum(torch.stack(node_k_alpha_sqrt_dev_list, dim=-1), dim=-1)
            if self.k_alpha_cap_hi > 0.0:
                # Smooth logistic saturation onto (lo, hi); k_alpha == 1 at _z == 0.
                k_alpha = self.k_alpha_cap_lo + (
                    self.k_alpha_cap_hi - self.k_alpha_cap_lo
                ) * torch.sigmoid(self.k_alpha_cap_a * _z + self.k_alpha_cap_z0)
                k_alpha_sqrt = torch.sqrt(k_alpha)
            else:
                k_alpha_sqrt = 1.0 + _z  # [n_nodes]
                k_alpha = k_alpha_sqrt**2
        else:
            k_alpha_sqrt = torch.ones_like(valence_widths)
            k_alpha = torch.ones_like(valence_widths)

        alpha_v_ratios = data["node_attrs"] @ self.elements_alpha_v_ratios

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "e0": e0,
            "node_energy": node_energy,
            "interaction_energy": interaction_energy,
            "contributions": contributions,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "valence_widths": valence_widths,
            "core_charges": core_charges,
            "charges": charges,
            "atomic_dipoles": atomic_dipoles,
            "atomic_quadrupoles": atomic_quadrupoles,
            "a_Thole": self.a_Thole,
            "alpha_v_ratios": alpha_v_ratios,
            "k_alpha": k_alpha,
            "k_alpha_sqrt": k_alpha_sqrt,
        }