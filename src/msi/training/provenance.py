"""Fail-closed provenance checks shared by all training entry points."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

from msi import REPO_ROOT


_FORBIDDEN_PARTS = (
    "/data/bench/instances/",
    "/bench/instances/",
    "/results/retrieval/",
)


def canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def file_sha256(path: str | Path) -> str:
    """Return a content identity for a training input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def skill_data_path(root: str | Path, skill_id: str) -> Path:
    """Return the only supported release path for one skill artifact."""
    base = Path(root)
    dataset = skill_id.rsplit("_", 1)[0]
    return base / dataset / f"{skill_id}.json"


def synthetic_stage_path(
    root: str | Path, skill_id: str, stage: str,
) -> Path:
    """Return one canonical synthetic-stage artifact.

    Tasks and trajectories are intentionally independent public datasets.  A
    missing trajectory is represented by the absence of its task ID from the
    trajectory artifact, never by a persisted ``pending`` state.
    """
    if stage not in {"tasks", "trajectories"}:
        raise ValueError(f"unsupported synthetic stage: {stage}")
    base = Path(root)
    dataset = skill_id.rsplit("_", 1)[0]
    return base / dataset / stage / f"{skill_id}.json"


def assert_train_path(path: str | Path, label: str = "training input") -> str:
    resolved = canonical(path)
    lowered = resolved.lower().replace("\\", "/")
    if any(part in lowered for part in _FORBIDDEN_PARTS):
        raise SystemExit(f"{label} resolves into a test-derived path: {resolved}")
    if not Path(resolved).is_file():
        raise SystemExit(f"missing {label}: {resolved}")
    return resolved


def assert_shared_output(path: str | Path) -> str:
    """Require all durable outputs to live under an explicitly shared root."""
    resolved = canonical(path)
    configured = os.environ.get("MSI_SHARED_ROOTS", str(REPO_ROOT))
    roots = [canonical(root) for root in configured.split(":") if root.strip()]
    if not any(resolved == root or resolved.startswith(root + os.sep) for root in roots):
        raise SystemExit(
            f"{resolved} is outside shared roots {roots}. "
            "Set MSI_SHARED_ROOTS explicitly if another shared mount is intended."
        )
    return resolved
