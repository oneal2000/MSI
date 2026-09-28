"""Publish the canonical, globally deduplicated synthetic anchor task pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from msi.anchors.artifacts import TASK_POOL_SCHEMA
from msi.audit.leakage import normalize
from msi.generation.artifacts import TASK_SCHEMA
from msi.protocol import SKILL_UNIVERSE_SIZE, inspect_skill_universe
from msi.training.provenance import (
    assert_shared_output, assert_train_path, synthetic_stage_path,
)
from msi.utils import atomic_json


def _rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "instances", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise SystemExit("anchor source file must be a JSON list or contain records/instances/data")


def _source_origins(row: dict, skill_ids: set[str]) -> list[str]:
    explicit = (
        row.get("origin_skill_ids") or row.get("gold_skill_ids")
        or row.get("skill_id") or row.get("origin_skill_id")
    )
    origins = [explicit] if isinstance(explicit, str) else [
        str(item) for item in (explicit or [])
    ]
    if not origins:
        instance_id = str(row.get("instance_id", ""))
        origins = [skill for skill in skill_ids if instance_id.startswith(skill)]
    origins = sorted(set(origins))
    if not origins or not set(origins) <= skill_ids:
        raise SystemExit(
            "anchor source row has missing/unknown origin skills: "
            f"instance_id={row.get('instance_id')!r}, origins={origins!r}"
        )
    return origins


def main() -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--task-dir")
    inputs.add_argument("--trajectory-dir")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=80)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fixed-val-offset", type=int, default=400)
    inputs.add_argument("--source-file")
    parser.add_argument("--skill-corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_shared_output(args.output)

    corpus_path = Path(args.skill_corpus).resolve()
    corpus_rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    ordered_skills, counts, valid_universe = inspect_skill_universe(corpus_rows)
    skill_ids = set(ordered_skills)
    if not valid_universe:
        raise SystemExit(f"invalid final {SKILL_UNIVERSE_SIZE}-skill universe: {counts}")

    candidates: list[tuple[dict, list[str], list[str]]] = []
    if args.trajectory_dir:
        from msi.training.dataset import load_training_data, split_train_val
        held_out = set()
        for skill_id in ordered_skills:
            source = synthetic_stage_path(args.trajectory_dir, skill_id, "trajectories")
            data = load_training_data(assert_train_path(source, "anchor trajectory"))
            if data.get("skill_id") != skill_id:
                raise SystemExit(f"trajectory skill mismatch: {source}")
            validation = split_train_val(
                data, n_train=args.num_train, n_val=args.num_val,
                seed=args.split_seed, fix_val=True,
                fixed_val_offset=args.fixed_val_offset,
            )
            held_out.update(normalize(row["question"]) for row in validation)
            for row in data["instances"]:
                candidates.append((
                    {"question": row["question"]}, [skill_id], [row["instance_id"]],
                ))
        # A question may occur under several skills. Exclude every validation
        # question globally so it cannot become another adapter's anchor.
        candidates = [item for item in candidates
                      if normalize(item[0]["question"]) not in held_out]
        source_kind = "synthetic_training_trajectories"
    elif args.task_dir:
        task_root = Path(args.task_dir).resolve()
        for skill_id in ordered_skills:
            source = synthetic_stage_path(task_root, skill_id, "tasks").resolve()
            if not source.is_file():
                continue
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("skill_id") != skill_id or payload.get("schema") != TASK_SCHEMA:
                raise SystemExit(f"non-canonical task artifact for {skill_id}: {source}")
            seen = set()
            for row in payload.get("tasks", []):
                question = str(row.get("question", "")).strip()
                normalized = normalize(question)
                if not normalized or normalized in seen:
                    raise SystemExit(f"empty/duplicate admitted task question in {source}")
                seen.add(normalized)
                row_id = str(row.get("task_id", ""))
                if not row_id:
                    raise SystemExit(f"admitted task row without task_id: {source}")
                candidates.append((row, [skill_id], [row_id]))
        source_kind = "admitted_synthetic_task_directory"
    else:
        source = Path(assert_train_path(args.source_file, "anchor source file"))
        for row in _rows(json.loads(source.read_text(encoding="utf-8"))):
            origins = _source_origins(row, skill_ids)
            source_ids = row.get("source_instance_ids")
            if not isinstance(source_ids, list) or not source_ids:
                source_ids = [str(row.get("instance_id", ""))]
            candidates.append((
                row, origins, [str(item) for item in source_ids if str(item)]
            ))
        source_kind = "explicit_synthetic_source_file"

    aggregated: dict[str, dict] = {}
    for row, origins, source_ids in candidates:
        question = str(row.get("question", "")).strip()
        normalized = normalize(question)
        if not normalized:
            raise SystemExit(f"empty question in anchor source row {row.get('instance_id')!r}")
        clean_id = "syn_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        item = aggregated.setdefault(clean_id, {
            "instance_id": clean_id, "source_instance_ids": [], "question": question,
            "origin_skill_ids": [], "gold_skill_ids": [], "datasets": [],
        })
        if normalize(item["question"]) != normalized:
            raise SystemExit(f"stable anchor-source ID collision: {clean_id}")
        for source_id in source_ids:
            if source_id and source_id not in item["source_instance_ids"]:
                item["source_instance_ids"].append(source_id)
        for skill_id in origins:
            if skill_id not in item["origin_skill_ids"]:
                item["origin_skill_ids"].append(skill_id)
                item["gold_skill_ids"].append(skill_id)
            dataset = skill_id.rsplit("_", 1)[0]
            if dataset not in item["datasets"]:
                item["datasets"].append(dataset)

    if not aggregated:
        raise SystemExit("anchor source contains no usable synthetic questions")
    pool = sorted(aggregated.values(), key=lambda row: row["instance_id"])
    for row in pool:
        for key in ("source_instance_ids", "origin_skill_ids", "gold_skill_ids", "datasets"):
            row[key].sort()
        row["dataset"] = row["datasets"][0]
    assignments = sum(len(row["origin_skill_ids"]) for row in pool)
    cross_groups = sum(len(row["origin_skill_ids"]) > 1 for row in pool)
    task_pool_id = hashlib.sha256(json.dumps(
        pool, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    destination = Path(args.output).resolve()
    atomic_json(destination, {
        "schema": TASK_POOL_SCHEMA,
        "metadata": {
            "source_kind": source_kind,
            "task_pool_id": task_pool_id,
            "skill_count": len(ordered_skills),
            "source_assignment_count": assignments,
            "question_count": len(pool),
            "cross_skill_duplicate_groups": cross_groups,
            "cross_skill_duplicate_assignments": assignments - len(pool),
        },
        "instances": pool,
    })
    print(json.dumps({
        "questions": len(pool), "source_assignments": assignments,
        "cross_skill_duplicate_groups": cross_groups, "task_pool": str(destination),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
