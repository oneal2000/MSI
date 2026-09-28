"""Direct completeness checks for resumable anchor answer caches.

This module deliberately imports no model/client dependencies: the zero-install
launcher runs under the host Python, while model stages run under the cluster
virtualenv.
"""

from __future__ import annotations

import json
from pathlib import Path

from msi.anchors.artifacts import (
    assigned_instance_ids, load_assignments, load_task_pool,
)


def answer_is_usable(answer: dict | None, instance: dict) -> bool:
    """Return whether a cached answer satisfies the shared resume contract."""
    answer = answer or {}
    return bool(
        answer.get("answer")
        and not answer.get("error")
        and answer.get("finish_reason") == "stop"
        and answer.get("question") == instance.get("question")
        and answer.get("dataset") == instance.get("dataset")
    )


def answer_is_retryable(answer: dict | None, instance: dict) -> bool:
    """Return whether a cached answer is a transient failure worth retrying.

    A non-empty, error-free non-``stop`` completion is a terminal candidate
    rejection.  Retrying a capped runaway or server-aborted generation can
    reproduce the same partial output indefinitely; the assignment's
    oversample margin lets assembly skip it instead.
    """
    answer = answer or {}
    if (
        answer.get("answer")
        and not answer.get("error")
        and answer.get("finish_reason") != "stop"
    ):
        return False
    return bool(
        not answer.get("answer")
        or answer.get("error")
        or answer.get("question") != instance.get("question")
        or answer.get("dataset") != instance.get("dataset")
    )


def answer_cache_complete(
    output: str | Path, task_pool: str | Path, assignment_paths: list[str | Path],
    base_model: str, served_model: str, max_tokens: int,
) -> bool:
    """Return whether every assigned input is usable or terminally rejected."""
    output = Path(output)
    if not output.is_file():
        return False
    _, rows = load_task_pool(task_pool)
    pool = {row["instance_id"]: row for row in rows}
    wanted = assigned_instance_ids(load_assignments(assignment_paths))
    if any(instance_id not in pool for instance_id in wanted):
        raise SystemExit("anchor assignment references an ID outside the published task pool")
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = {
        "source_kind": "synthetic_train_pool",
        "model": base_model,
        "served_model": served_model,
        "max_tokens": max_tokens,
    }
    if payload.get("provenance") != expected:
        raise SystemExit(
            f"existing answer cache belongs to different model settings: {output}; "
            "use a new run ID"
        )
    answers = payload.get("answers", {})
    return not any(
        answer_is_retryable(answers.get(instance_id), pool[instance_id])
        for instance_id in wanted
    )
