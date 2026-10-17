"""CaP-X-owned public force/time instructions for paired strap experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_seed_sequence(manifest_path: str | Path, seed: int) -> list[float]:
    """Read one frozen instruction without changing OpenTacBench task code."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "tension_strap_seed_sequences.v1":
        raise ValueError("unsupported tension-strap seed manifest")
    bounds = payload.get("target_range_N")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("manifest must declare two force bounds")
    low, high = (float(value) for value in bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 24):
        raise ValueError("manifest force bounds must lie within 0-24 N")
    sequences = payload.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        raise ValueError("manifest must contain seed instructions")
    for key, sequence in sequences.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"invalid manifest seed {key!r}")
        if not isinstance(sequence, list) or len(sequence) != 4:
            raise ValueError(f"seed {key} must have two force/time stages")
        if any(isinstance(value, bool) for value in sequence):
            raise ValueError(f"seed {key} contains a boolean force or duration")
        values = [float(value) for value in sequence]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"seed {key} contains a nonfinite value")
        if not (low <= values[0] <= high and low <= values[2] <= high):
            raise ValueError(f"seed {key} force exceeds the declared range")
        if values[1] <= 0 or values[3] <= 0:
            raise ValueError(f"seed {key} hold duration must be positive")
    try:
        return [float(value) for value in sequences[str(seed)]]
    except KeyError as exc:
        raise KeyError(f"seed {seed} is absent from the frozen manifest") from exc
