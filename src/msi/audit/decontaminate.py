"""Remove benchmark-overlapping synthetic rows outside the generator process.

The generator never receives benchmark paths or contents.  This independent
post-generation audit compares only completed synthetic artifacts with the
four benchmark files, removes exact/near overlaps as paired task/trajectory
rows, and persists only rejected stable task IDs for future generator runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msi.audit.leakage import (
    build_near_index,
    near_matches,
    normalize,
    test_questions,
)
from msi.generation.artifacts import TASK_SCHEMA, TRAJECTORY_SCHEMA
from msi.utils import atomic_json


BLOCKLIST_SCHEMA = "msi.decontaminated-task-ids"
REMOVED = 3


def _payload(path: Path, schema: str, list_key: str, skill_id: str) -> dict:
    if not path.is_file():
        return {"schema": schema, "skill_id": skill_id, list_key: []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema") != schema
        or data.get("skill_id") != skill_id
        or not isinstance(data.get(list_key), list)
    ):
        raise SystemExit(f"invalid canonical synthetic artifact: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent destructive decontamination of synthetic artifacts",
    )
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--test-instances", action="append", required=True)
    parser.add_argument("--near-threshold", type=float, default=0.92)
    args = parser.parse_args()

    test_paths = [str(Path(path).resolve()) for path in args.test_instances]
    if len(test_paths) != 4 or len(set(test_paths)) != 4:
        raise SystemExit("decontamination requires four distinct benchmark files")
    if not 0 < args.near_threshold <= 1:
        raise SystemExit("--near-threshold must be in (0, 1]")

    tests = test_questions(test_paths)
    exact = {row["normalized"] for row in tests}
    inverted, test_grams = build_near_index(tests)

    def overlaps(value: object) -> bool:
        normalized = normalize(value)
        if not normalized:
            return False
        if normalized in exact:
            return True
        return bool(near_matches(
            normalized, tests, inverted, test_grams, args.near_threshold,
        ))

    tasks_path = Path(args.tasks)
    trajectories_path = Path(args.trajectories)
    tasks = _payload(tasks_path, TASK_SCHEMA, "tasks", args.skill_id)
    trajectories = _payload(
        trajectories_path, TRAJECTORY_SCHEMA, "instances", args.skill_id,
    )

    rejected = {
        str(row.get("task_id", ""))
        for row in tasks["tasks"]
        if overlaps(row.get("question"))
    }
    rejected.update(
        str(row.get("task_id", ""))
        for row in trajectories["instances"]
        if overlaps(row.get("question"))
    )
    rejected.discard("")

    state_dir = Path(args.state_dir)
    blocklist_path = state_dir / "decontaminated_task_ids.json"
    previous: set[str] = set()
    if blocklist_path.is_file():
        blocklist = json.loads(blocklist_path.read_text(encoding="utf-8"))
        if (
            blocklist.get("schema") != BLOCKLIST_SCHEMA
            or blocklist.get("skill_id") != args.skill_id
            or not isinstance(blocklist.get("task_ids"), list)
        ):
            raise SystemExit(f"invalid decontamination blocklist: {blocklist_path}")
        previous = {
            str(task_id) for task_id in blocklist.get("task_ids", []) if str(task_id)
        }
    blocked = previous | rejected

    original_tasks = len(tasks["tasks"])
    original_trajectories = len(trajectories["instances"])
    tasks["tasks"] = [
        row for row in tasks["tasks"] if str(row.get("task_id", "")) not in blocked
    ]
    trajectories["instances"] = [
        row for row in trajectories["instances"]
        if str(row.get("task_id", "")) not in blocked
    ]
    removed_tasks = original_tasks - len(tasks["tasks"])
    removed_trajectories = original_trajectories - len(trajectories["instances"])

    if blocked != previous:
        atomic_json(blocklist_path, {
            "schema": BLOCKLIST_SCHEMA,
            "skill_id": args.skill_id,
            "task_ids": sorted(blocked),
        })
    if removed_tasks:
        tasks["stats"] = {
            **(tasks.get("stats") or {}), "task_count": len(tasks["tasks"]),
        }
        atomic_json(tasks_path, tasks)
    if removed_trajectories:
        trajectories["stats"] = {
            **(trajectories.get("stats") or {}),
            "accepted_count": len(trajectories["instances"]),
        }
        atomic_json(trajectories_path, trajectories)

    changed = bool(rejected or removed_tasks or removed_trajectories)
    print(json.dumps({
        "skill_id": args.skill_id,
        "removed_tasks": removed_tasks,
        "removed_trajectories": removed_trajectories,
        "blocked_task_ids": len(blocked),
        "status": "decontaminated" if changed else "clean",
    }, sort_keys=True))
    return REMOVED if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
