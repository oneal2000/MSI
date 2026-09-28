"""Canonical task/trajectory artifacts for the two-stage generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from msi.generation.admission import validate_admitted_instance
from msi.training.provenance import normalize_question


TASK_SCHEMA = "msi.tasks"
TRAJECTORY_SCHEMA = "msi.trajectories"
TASK_PROGRESS_SCHEMA = "msi.task-progress"
TRAJECTORY_PROGRESS_SCHEMA = "msi.trajectory-progress"


def task_id(skill_id: str, question: object) -> str:
    normalized = normalize_question(question)
    if not normalized:
        raise ValueError("cannot assign a task ID to an empty question")
    digest = hashlib.sha256(
        f"{skill_id}\0{normalized}".encode("utf-8")
    ).hexdigest()[:20]
    return f"task_{digest}"


def canonical_task(skill_id: str, task: dict) -> dict:
    question = str(task.get("question", "")).strip()
    result = dict(task)
    result.pop("request_index", None)
    result.pop("status", None)
    result["question"] = question
    present = str(result.get("task_id", "")).strip()
    result["task_id"] = present or task_id(skill_id, question)
    return result


def instance_task_id(skill_id: str, instance: dict) -> str:
    present = str(instance.get("task_id", "")).strip()
    return present or task_id(skill_id, instance.get("question", ""))


def read_tasks(path: Path, skill_id: str) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != TASK_SCHEMA
        or payload.get("skill_id") != skill_id
        or not isinstance(payload.get("tasks"), list)
    ):
        raise RuntimeError(f"invalid canonical task artifact: {path}")
    rows = []
    for original in payload["tasks"]:
        if not str(original.get("task_id", "")):
            raise RuntimeError(f"canonical task is missing persisted task_id: {path}")
        rows.append(canonical_task(skill_id, original))
    return rows


def read_instances(path: Path, skill_id: str) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != TRAJECTORY_SCHEMA
        or payload.get("skill_id") != skill_id
        or not isinstance(payload.get("instances"), list)
    ):
        raise RuntimeError(f"invalid canonical trajectory artifact: {path}")
    rows = []
    for original in payload["instances"]:
        row = dict(original)
        if not str(row.get("task_id", "")):
            raise RuntimeError(
                f"canonical trajectory is missing persisted task_id: {path}"
            )
        expected_task = instance_task_id(skill_id, row)
        expected_instance = "syn_" + expected_task.removeprefix("task_")
        if row.get("instance_id") != expected_instance:
            raise RuntimeError(
                f"canonical trajectory has unstable instance_id: {path}"
            )
        admitted, reason = validate_admitted_instance(row)
        if not admitted:
            raise RuntimeError(
                f"non-admitted trajectory in canonical artifact: {path}: {reason}"
            )
        rows.append(row)
    return rows
