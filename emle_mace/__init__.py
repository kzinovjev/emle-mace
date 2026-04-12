__all__ = ["EnergyEMLEMACE", "WeightedEnergyForcesEMLELoss"]


def __getattr__(name):
    if name == "EnergyEMLEMACE":
        from emle_mace.models import EnergyEMLEMACE
        return EnergyEMLEMACE
    if name == "WeightedEnergyForcesEMLELoss":
        from emle_mace.loss import WeightedEnergyForcesEMLELoss
        return WeightedEnergyForcesEMLELoss
    raise AttributeError(f"module 'emle_mace' has no attribute {name!r}")
