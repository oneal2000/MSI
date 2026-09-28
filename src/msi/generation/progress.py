"""Append-only generation progress journals.

Progress is run-local state.  Published task and trajectory JSON files are the
only complete artifacts; the journals retain every successful or failed
attempt so an operator can choose whether failed work should be retried.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Hashable

from msi.generation.artifacts import (
    TASK_PROGRESS_SCHEMA,
    TRAJECTORY_PROGRESS_SCHEMA,
)


def rows(path: Path, schema: str) -> list[dict]:
    """Read complete journal rows, ignoring only a trailing partial write."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                break
            raise RuntimeError(
                f"corrupt progress row {line_number}: {path}"
            ) from error
        if row.get("schema") != schema:
            raise RuntimeError(f"unexpected progress schema in {path}")
        result.append(row)
    return result


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def prepare(path: Path, schema: str) -> None:
    """Remove only a crash-truncated final row before appending new attempts."""
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    content_indexes = [index for index, line in enumerate(lines) if line.strip()]
    if not content_indexes:
        return
    last_content = content_indexes[-1]
    for index in content_indexes:
        try:
            row = json.loads(lines[index])
        except json.JSONDecodeError as error:
            if index != last_content:
                raise RuntimeError(
                    f"corrupt progress row {index + 1}: {path}"
                ) from error
            path.write_text("".join(lines[:index]), encoding="utf-8")
            return
        if row.get("schema") != schema:
            raise RuntimeError(f"unexpected progress schema in {path}")


def _latest(
    journal_rows: list[dict], key: Callable[[dict], Hashable], statuses: set[str],
) -> dict[Hashable, dict]:
    latest: dict[Hashable, dict] = {}
    for row in journal_rows:
        status = str(row.get("status", ""))
        if status not in statuses:
            raise RuntimeError(f"invalid progress status: {status!r}")
        attempt = int(row.get("attempt", 1))
        if attempt < 1:
            raise RuntimeError("progress attempt must be positive")
        identifier = key(row)
        previous = latest.get(identifier)
        if previous is None or attempt >= int(previous.get("attempt", 1)):
            latest[identifier] = row
    return latest


def latest_tasks(path: Path) -> dict[int, dict]:
    def request_id(row: dict) -> int:
        value = row.get("request_id", row.get("request_index", -1))
        identifier = int(value)
        if identifier < 0:
            raise RuntimeError("task progress requires a non-negative request_id")
        return identifier

    return _latest(
        rows(path, TASK_PROGRESS_SCHEMA), request_id, {"success", "fail"},
    )


def latest_trajectories(path: Path) -> dict[str, dict]:
    def task_id(row: dict) -> str:
        identifier = str(
            row.get("task_id") or (row.get("instance") or {}).get("task_id") or ""
        )
        if not identifier:
            raise RuntimeError("trajectory progress requires task_id")
        return identifier

    return _latest(
        rows(path, TRAJECTORY_PROGRESS_SCHEMA),
        task_id,
        {"accepted", "failed"},
    )


def next_attempt(previous: dict | None) -> int:
    return int(previous.get("attempt", 1)) + 1 if previous else 1
