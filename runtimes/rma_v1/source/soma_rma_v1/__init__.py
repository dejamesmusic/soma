"""soma residual metabolic aperture v1."""

SPECIES = "soma_rma_v1"

__all__ = ["SPECIES", "SOMA", "RMAConfig"]


def __getattr__(name):
    if name == "SOMA":
        from .runtime import SOMA
        return SOMA
    if name == "RMAConfig":
        from .config import RMAConfig
        return RMAConfig
    raise AttributeError(name)
