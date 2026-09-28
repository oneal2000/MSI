"""Stable local GPU selection across CUDA visibility re-numbering."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GPUSelectionError(ValueError):
    pass


def inventory() -> tuple[dict[str, str], set[str]]:
    """Return physical-index -> UUID using the driver rather than CUDA ordinals."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}, set()
    if result.returncode != 0:
        return {}, set()
    by_index: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and all(parts):
            by_index[parts[0]] = parts[1]
    return by_index, set(by_index.values())


def resolve(raw: str | None, *, allow_unresolved: bool = False) -> list[str]:
    """Normalize explicit physical IDs and inherited constraints to GPU UUIDs."""
    by_index, uuids = inventory()
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    constrained = visible_raw is not None
    visible = [item.strip() for item in (visible_raw or "").split(",") if item.strip()]

    def normalize(token: str) -> str:
        if token in by_index:
            return by_index[token]
        if token in uuids or token.startswith("MIG-"):
            return token
        if allow_unresolved:
            return token
        raise GPUSelectionError(
            f"GPU {token!r} cannot be mapped to a stable UUID with nvidia-smi"
        )

    allowed = {normalize(token) for token in visible} if constrained else uuids
    requested = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if raw is not None and (not requested or len(requested) != len(set(requested))):
        raise GPUSelectionError(
            "--gpu must contain one or more distinct comma-separated devices"
        )
    selected = [normalize(token) for token in requested] if requested else (
        [normalize(token) for token in visible] if constrained
        else [by_index[index] for index in sorted(by_index, key=int)]
    )
    if constrained and not set(selected) <= allowed:
        raise GPUSelectionError("--gpu requests devices excluded by CUDA_VISIBLE_DEVICES")
    if len(selected) != len(set(selected)):
        raise GPUSelectionError("GPU selection resolves to duplicate UUIDs")
    return selected


def physical_index(device: str) -> tuple[str, str]:
    """Resolve a stable selected device to the current driver index and UUID.

    Internal scheduling keeps UUIDs, but vLLM 0.20.x parses every
    ``CUDA_VISIBLE_DEVICES`` token as an integer in its architecture-inspection
    subprocess. The service launcher therefore converts only at its final exec
    boundary and independently verifies the CUDA binding.
    """
    by_index, _ = inventory()
    if device in by_index:
        return device, by_index[device]
    matches = [index for index, uuid in by_index.items() if uuid == device]
    if len(matches) != 1:
        raise GPUSelectionError(
            f"GPU {device!r} does not map to one physical nvidia-smi index"
        )
    return matches[0], device


def child_environment(gpu: str, cache_dir: Path, temp_dir: Path) -> dict[str, str]:
    """Isolate one CUDA child while keeping every cache on shared storage."""
    cache_dir = cache_dir.resolve()
    temp_dir = temp_dir.resolve()
    return {
        "CUDA_VISIBLE_DEVICES": gpu,
        "HF_HOME": str(cache_dir / "huggingface"),
        "XDG_CACHE_HOME": str(cache_dir),
        "XDG_CONFIG_HOME": str(cache_dir / "config"),
        "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        "TRANSFORMERS_CACHE": str(cache_dir / "huggingface"),
        "VLLM_CACHE_ROOT": str(cache_dir / "vllm"),
        "TORCH_HOME": str(cache_dir / "torch"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        "CUDA_CACHE_PATH": str(cache_dir / "cuda"),
        "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
        "TMPDIR": str(temp_dir),
    }
