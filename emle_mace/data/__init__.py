"""Data-layer patches for EnergyEMLEMACE training.

Standard ``mace`` (main branch) does not know about the EMLE per-atom
properties (valence_widths, core_charges, atomic_dipoles).  When training
with ``EnergyEMLEMACE`` we need two small additions to the mace data layer:

1. ``update_keyspec_from_kwargs`` — recognise the three EMLE per-atom keys
   (``valence_widths_key``, ``core_charges_key``, ``atomic_dipoles_key``) as
   per-atom *arrays* (not per-structure *info*) so they are read from
   ``atoms.arrays`` when loading an XYZ file.

2. ``AtomicData.from_config`` — extract the EMLE properties and their per-
   sample loss weights from a ``Configuration`` object and attach them to the
   returned ``AtomicData`` graph.

Call ``patch()`` before starting the training loop and ``restore()`` after to
leave the mace namespace clean.
"""

import torch

import mace.data.utils as _mace_data_utils
from mace.data.atomic_data import AtomicData as _AtomicData


# ---------------------------------------------------------------------------
# Saved originals (populated by patch())
# ---------------------------------------------------------------------------
_orig_update_keyspec = None
_orig_from_config = None
_patched = False  # guard against double-patching


# ---------------------------------------------------------------------------
# Patched implementations
# ---------------------------------------------------------------------------

def _emle_update_keyspec(keyspec, keydict):
    """Extend mace's keyspec builder to include EMLE per-atom array keys."""
    result = _orig_update_keyspec(keyspec, keydict)
    emle_array_keys = ["valence_widths_key", "core_charges_key", "atomic_dipoles_key"]
    arrays_keys = {}
    for key in emle_array_keys:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(arrays_keys=arrays_keys)
    return result


@classmethod  # type: ignore[misc]
def _emle_from_config(cls, config, z_table, cutoff, heads=None, **kwargs):
    """Extend AtomicData.from_config to also populate EMLE fields."""
    data = _orig_from_config.__func__(cls, config, z_table, cutoff, heads=heads, **kwargs)
    num_atoms = len(config.atomic_numbers)
    _dt = torch.get_default_dtype()

    # Per-atom EMLE properties (with sensible zero defaults when absent).
    for name, default_shape in [
        ("valence_widths", (num_atoms,)),
        ("core_charges", (num_atoms,)),
        ("atomic_dipoles", (num_atoms, 3)),
    ]:
        val = config.properties.get(name)
        setattr(
            data,
            name,
            torch.tensor(val, dtype=_dt) if val is not None
            else torch.zeros(default_shape, dtype=_dt),
        )

    # Per-sample loss weights (default 1.0 when absent).
    for name in ("valence_widths", "core_charges", "atomic_dipoles"):
        w = config.property_weights.get(name)
        setattr(
            data,
            f"{name}_weight",
            torch.tensor(w, dtype=_dt) if w is not None
            else torch.tensor(1.0, dtype=_dt),
        )

    return data


# ---------------------------------------------------------------------------
# Public patch / restore API
# ---------------------------------------------------------------------------

def patch():
    """Monkey-patch mace's data layer to support EMLE properties.

    Idempotent: safe to call multiple times (only the first call has effect).
    """
    global _orig_update_keyspec, _orig_from_config, _patched  # noqa: PLW0603
    if _patched:
        return

    _orig_update_keyspec = _mace_data_utils.update_keyspec_from_kwargs
    _orig_from_config = _AtomicData.from_config

    _mace_data_utils.update_keyspec_from_kwargs = _emle_update_keyspec
    _AtomicData.from_config = _emle_from_config
    _patched = True


def restore():
    """Restore mace's data layer to its original state."""
    global _patched  # noqa: PLW0603
    if not _patched:
        return
    if _orig_update_keyspec is not None:
        _mace_data_utils.update_keyspec_from_kwargs = _orig_update_keyspec
    if _orig_from_config is not None:
        _AtomicData.from_config = _orig_from_config
    _patched = False