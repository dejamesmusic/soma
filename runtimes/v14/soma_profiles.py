"""shared capacity profiles for soma v13."""

from __future__ import annotations


DEFAULT_CAPACITY = "medium"
CAPACITY_NAMES = ("small", "medium", "large", "extra large")
DEFAULT_BANDS = 48
DEFAULT_AUTO_BATCH = {
    "small": 1024,
    "medium": 512,
    "large": 512,
    "extra large": 256,
}

# measured mps driver working sets on the 48-band production profiles,
# rounded upward into allocation buckets and given a 15% admission margin.
_MPS_WORKING_SET_GB = {
    "small": ((128, 0.93), (256, 1.46), (512, 2.68), (1024, 2.70)),
    "medium": ((128, 2.57), (256, 2.57), (512, 2.57), (1024, 5.04)),
    "large": ((128, 2.61), (256, 2.63), (512, 3.83), (1024, 5.04)),
    "extra large": ((128, 5.10), (256, 5.60), (512, 7.80), (1024, 10.50)),
}


def capacity_values(name: str, bands: int) -> dict:
    """derive one coherent architecture from a user-facing capacity choice."""
    name = str(name or DEFAULT_CAPACITY).lower()
    bands = max(1, int(bands))
    if name == "small":
        return {
            "symbol_dim": 256,
            "scale_rank": 64,
            "scale_depth": 3,
            "summaries": max(1, min(bands, round(bands / 6))),
            "model_dim": 1024,
            "trunk_depth": 4,
            "expansion_dim": 2048,
        }
    if name == "medium":
        return {
            "symbol_dim": 384,
            "scale_rank": 96,
            "scale_depth": 3,
            "summaries": max(1, min(bands, round(bands / 4))),
            "model_dim": 1536,
            "trunk_depth": 4,
            "expansion_dim": 3072,
        }
    if name == "large":
        values = capacity_values("medium", bands)
        values["trunk_depth"] = 6
        return values
    if name == "extra large":
        return {
            "symbol_dim": 512,
            "scale_rank": 128,
            "scale_depth": 4,
            "summaries": max(1, min(bands, round(bands / 3))),
            "model_dim": 2048,
            "trunk_depth": 8,
            "expansion_dim": 4096,
        }
    raise ValueError(f"unknown capacity: {name}")


def parameter_count(
    name: str,
    bands: int,
    vocab_size: int = 1024,
    scale_kernel: int = 5,
) -> int:
    values = capacity_values(name, bands)
    symbol = values["symbol_dim"]
    rank = values["scale_rank"]
    scale_depth = values["scale_depth"]
    summaries = values["summaries"]
    model = values["model_dim"]
    depth = values["trunk_depth"]
    expansion = values["expansion_dim"]
    total = vocab_size * symbol
    total += scale_depth * (
        symbol * scale_kernel + 2 * symbol * rank + 6 * symbol + 2
    )
    total += symbol * 3
    total += (summaries + 1) * symbol * model
    total += depth * (3 * model * expansion)
    total += model * symbol
    return int(total)


def default_batch_size(name: str) -> int:
    name = str(name or DEFAULT_CAPACITY).lower()
    if name not in DEFAULT_AUTO_BATCH:
        raise ValueError(f"unknown capacity: {name}")
    return DEFAULT_AUTO_BATCH[name]


def estimated_mps_training_bytes(name: str, bands: int, batch: int) -> int:
    """conservative mps working-set estimate from measured allocation buckets."""

    name = str(name or DEFAULT_CAPACITY).lower()
    if name not in _MPS_WORKING_SET_GB:
        raise ValueError(f"unknown capacity: {name}")
    bands = int(bands)
    batch = int(batch)
    if bands < 1 or batch < 1:
        raise ValueError("bands and batch must be positive")
    if batch > 1024:
        return 2**63 - 1
    measured_gb = next(
        working_gb
        for ceiling, working_gb in _MPS_WORKING_SET_GB[name]
        if batch <= ceiling
    )
    default_params = parameter_count(name, DEFAULT_BANDS)
    shape_scale = max(
        parameter_count(name, bands) / default_params,
        bands / DEFAULT_BANDS,
    )
    return int(measured_gb * shape_scale * 1_000_000_000)


def mps_fit_status(
    name: str,
    bands: int,
    batch: int,
    recommended_memory: int,
) -> tuple[str, int]:
    estimate = estimated_mps_training_bytes(name, bands, batch)
    limit = max(1, int(recommended_memory))
    ratio = estimate / limit
    if ratio <= 0.70:
        status = "comfortable"
    elif ratio <= 0.90:
        status = "pressure"
    else:
        status = "unsupported"
    return status, estimate


def recommended_mps_batch(
    name: str,
    bands: int,
    recommended_memory: int,
) -> int:
    for batch in (1024, 512, 256, 128, 64):
        status, _estimate = mps_fit_status(
            name, bands, batch, recommended_memory
        )
        if status == "comfortable":
            return batch
    return 64


def infer_capacity(config) -> str:
    """name a saved shape when it exactly matches a current profile."""
    def read(key, default=None):
        if isinstance(config, dict):
            aliases = {
                "bands": ("bands", "n_bands"),
                "symbol_dim": ("symbol_dim", "token_dim"),
                "trunk_depth": ("trunk_depth", "language_depth"),
                "expansion_dim": (
                    "expansion_dim", "language_expansion"
                ),
            }
            for candidate in aliases.get(key, (key,)):
                if candidate in config:
                    return config[candidate]
            return default
        return getattr(config, key, default)

    bands = int(read("bands", read("n_bands", 48)))
    keys = (
        "symbol_dim",
        "scale_rank",
        "scale_depth",
        "summaries",
        "model_dim",
        "trunk_depth",
        "expansion_dim",
    )
    for name in CAPACITY_NAMES:
        expected = capacity_values(name, bands)
        if all(int(read(key, -1)) == expected[key] for key in keys):
            return name
    return "custom"
