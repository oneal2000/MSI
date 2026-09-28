"""Published artifact contracts shared by both anchor stages."""

from __future__ import annotations

import json
from pathlib import Path


TASK_POOL_SCHEMA = "msi.anchor-task-pool"
ASSIGNMENT_SCHEMA = "msi.anchor-task-assignment"


def same_skill_ids(left, right) -> bool:
    """Compare unordered skill-ID collections using the published semantics."""
    return sorted(set(left or [])) == sorted(set(right or []))


def published_json_matches(path: str | Path, payload: dict, kind: str) -> bool:
    """Return whether an identical artifact exists; reject incompatible output."""
    destination = Path(path)
    if not destination.exists():
        return False
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"published {kind} is unreadable: {destination}") from error
    if existing != payload:
        raise SystemExit(f"published {kind} conflicts with current inputs: {destination}")
    return True


def load_task_pool(path: str | Path) -> tuple[dict, list[dict]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TASK_POOL_SCHEMA:
        raise SystemExit(f"invalid published anchor task pool: {source}")
    rows = payload.get("instances")
    if not isinstance(rows, list):
        raise SystemExit(f"anchor task pool has no instances list: {source}")
    ids = [str(row.get("instance_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"anchor task pool has empty or duplicate instance IDs: {source}")
    return payload.get("metadata", {}), rows


def assignment_path(root: str | Path, skill_id: str) -> Path:
    return Path(root) / skill_id.rsplit("_", 1)[0] / f"{skill_id}.json"


def load_assignments(paths: list[str | Path]) -> dict[str, dict]:
    assignments: dict[str, dict] = {}
    for raw in paths:
        source = Path(raw)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != ASSIGNMENT_SCHEMA:
            raise SystemExit(f"invalid anchor task assignment: {source}")
        skill_id = str(payload.get("skill_id", ""))
        if not skill_id or skill_id in assignments:
            raise SystemExit(f"empty or duplicate assignment skill ID: {source}")
        if source.stem != skill_id or source.parent.name != skill_id.rsplit("_", 1)[0]:
            raise SystemExit(f"assignment path does not match its skill ID: {source}")
        if not isinstance(payload.get("retrieval"), dict) or not isinstance(
            payload.get("sampling"), dict
        ):
            raise SystemExit(f"assignment is missing retrieval/sampling provenance: {source}")
        near = payload.get("near")
        random_rows = payload.get("random")
        if not isinstance(near, list) or not isinstance(random_rows, list):
            raise SystemExit(f"assignment near/random fields must be lists: {source}")
        ids = [str(item) for item in [*near, *random_rows]]
        if not all(ids) or len(ids) != len(set(ids)):
            raise SystemExit(f"assignment has empty or duplicate task IDs: {source}")
        assignments[skill_id] = payload
    if not assignments:
        raise SystemExit("at least one anchor task assignment is required")
    profiles = {row.get("retrieval", {}).get("profile") for row in assignments.values()}
    if len(profiles) != 1 or None in profiles:
        raise SystemExit("anchor task assignments use inconsistent retrieval profiles")
    return assignments


def assigned_instance_ids(assignments: dict[str, dict]) -> list[str]:
    return sorted({
        str(instance_id)
        for row in assignments.values()
        for key in ("near", "random")
        for instance_id in row[key]
    })
