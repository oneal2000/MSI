#!/usr/bin/env python3
"""Publish deterministic per-skill anchor task assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from msi.anchors.artifacts import (
    ASSIGNMENT_SCHEMA, assignment_path, load_task_pool, published_json_matches,
    same_skill_ids,
)
from msi.training.provenance import assert_shared_output
from msi.utils import atomic_json


def stable_seed(seed: int, skill_id: str) -> int:
    token = hashlib.sha256(f"{seed}:{skill_id}".encode()).digest()[:8]
    return int.from_bytes(token, "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--skill-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-retriever", default="bge_base_rerank")
    parser.add_argument("--n-nearmiss", type=int, default=80)
    parser.add_argument("--n-random", type=int, default=80)
    parser.add_argument("--oversample", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts-config")
    parser.add_argument("--target-prefix", action="append", default=[])
    parser.add_argument("--target-skill", action="append", default=[])
    args = parser.parse_args()
    assert_shared_output(args.output_dir)
    if args.n_nearmiss < 0 or args.n_random < 0 or args.oversample < 1:
        raise SystemExit("assignment counts must be non-negative and oversample must be >= 1")

    pool_metadata, pool = load_task_pool(args.task_pool)
    task_pool_id = str(pool_metadata.get("task_pool_id", ""))
    pool_ids = {row["instance_id"] for row in pool}
    pool_by_id = {row["instance_id"]: row for row in pool}
    retrieval_data = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))
    metadata = retrieval_data.get("metadata", {})
    if not task_pool_id or metadata.get("task_pool_id") != task_pool_id:
        raise SystemExit("anchor retrieval belongs to a different task pool")
    if metadata.get("source_kind") != "synthetic_train_pool":
        raise SystemExit("anchor retrieval source_kind must be synthetic_train_pool")
    if metadata.get("retriever") != args.required_retriever:
        raise SystemExit(
            f"anchor retrieval must use {args.required_retriever}, got "
            f"{metadata.get('retriever')}"
        )
    if args.required_retriever.endswith("_rerank") and not metadata.get("reranker"):
        raise SystemExit("anchor retrieval is missing reranker provenance")
    rows = retrieval_data.get("results", [])
    if len(rows) != len({row.get("instance_id") for row in rows}):
        raise SystemExit("anchor retrieval has duplicate instance IDs")
    retrieval = {row["instance_id"]: row for row in rows}
    if set(retrieval) != pool_ids:
        raise SystemExit("retrieval IDs do not exactly match the published task pool")
    for instance_id, row in retrieval.items():
        if not same_skill_ids(
            row.get("gold_skill_ids"), pool_by_id[instance_id].get("gold_skill_ids")
        ):
            raise SystemExit(f"retrieval/task-pool origin mismatch for {instance_id}")

    all_skills = [row["skill_id"] for row in json.loads(
        Path(args.skill_corpus).read_text(encoding="utf-8")
    )]
    skill_set = set(all_skills)
    invalid = [
        item.get("skill_id") for row in rows for item in row.get("retrieved", [])
        if item.get("skill_id") not in skill_set
    ]
    if invalid:
        raise SystemExit(f"anchor retrieval contains candidates outside the corpus: {invalid[:20]}")
    duplicate_rankings = [
        row.get("instance_id") for row in rows
        if len([item.get("skill_id") for item in row.get("retrieved", [])])
        != len({item.get("skill_id") for item in row.get("retrieved", [])})
    ]
    if duplicate_rankings:
        raise SystemExit(f"anchor retrieval contains duplicate rankings: {duplicate_rankings[:20]}")
    counts = json.loads(Path(args.counts_config).read_text(encoding="utf-8")) \
        if args.counts_config else {}
    prefixes, explicit = set(args.target_prefix), set(args.target_skill)
    if prefixes and explicit:
        raise SystemExit("use either --target-prefix or --target-skill, not both")
    selected = [
        skill for skill in all_skills
        if ((explicit and skill in explicit)
            or (prefixes and skill.rsplit("_", 1)[0] in prefixes)
            or (not explicit and not prefixes))
    ]
    unknown_prefixes = prefixes - {skill.rsplit("_", 1)[0] for skill in all_skills}
    unknown_skills = explicit - skill_set
    if unknown_prefixes or unknown_skills or not selected:
        raise SystemExit(
            f"invalid or empty target selection: prefixes={sorted(unknown_prefixes)}, "
            f"skills={sorted(unknown_skills)}"
        )

    published, deficits = [], []
    for skill_id in selected:
        dataset = skill_id.rsplit("_", 1)[0]
        skill_counts = counts.get(skill_id, counts.get(dataset, {}))
        n_nearmiss = int(skill_counts.get("n_nearmiss", args.n_nearmiss))
        n_random = int(skill_counts.get("n_random", args.n_random))
        if n_nearmiss < 0 or n_random < 0:
            raise SystemExit(f"assignment counts must be non-negative for {skill_id}")
        strict, loose, random_pool = [], [], []
        for instance_id, row in retrieval.items():
            if skill_id in set(pool_by_id[instance_id].get("gold_skill_ids") or []):
                continue
            ranked = [item.get("skill_id") for item in row.get("retrieved", [])]
            if ranked and ranked[0] == skill_id:
                strict.append(instance_id)
            elif skill_id in ranked[1:]:
                loose.append(instance_id)
            else:
                random_pool.append(instance_id)
        rng = random.Random(stable_seed(args.seed, skill_id))
        for values in (strict, loose, random_pool):
            rng.shuffle(values)
        wanted_near = math.ceil(n_nearmiss * args.oversample)
        strict_selected = strict[:wanted_near]
        loose_selected = loose[:max(0, wanted_near - len(strict_selected))]
        near = strict_selected + loose_selected
        random_selected = random_pool[:math.ceil(n_random * args.oversample)]
        if len(near) < n_nearmiss or len(random_selected) < n_random:
            deficits.append({
                "skill_id": skill_id, "strict_near_available": len(strict),
                "loose_near_available": len(loose), "near_required": n_nearmiss,
                "random_available": len(random_selected), "random_required": n_random,
            })
        published.append({
            "schema": ASSIGNMENT_SCHEMA,
            "skill_id": skill_id,
            "retrieval": {
                "profile": metadata["retriever"], "model": metadata["model"],
                "reranker": metadata.get("reranker"), "top_k": metadata.get("top_k"),
                "dtype": metadata.get("dtype"), "task_pool_id": task_pool_id,
            },
            "sampling": {
                "near_policy": "retrieval_top1_then_topk_loose",
                "n_nearmiss": n_nearmiss, "n_random": n_random,
                "oversample": args.oversample, "seed": args.seed,
            },
            "near": near, "random": random_selected,
            "strict_near_available": len(strict), "loose_near_available": len(loose),
            "strict_near_selected": len(strict_selected),
            "loose_near_selected": len(loose_selected),
        })
    if deficits:
        raise SystemExit(
            "anchor candidate pools are insufficient: "
            f"{json.dumps(deficits[:20], ensure_ascii=False)}"
        )
    output_dir = Path(args.output_dir)
    destinations = [
        assignment_path(output_dir, assignment["skill_id"])
        for assignment in published
    ]
    existing = [
        published_json_matches(destination, assignment, "anchor task assignment")
        for destination, assignment in zip(destinations, published)
    ]
    for destination, assignment, matches in zip(destinations, published, existing):
        if not matches:
            atomic_json(destination, assignment)
    created = existing.count(False)
    print(
        f"published {created}, reused {len(published) - created} per-skill anchor "
        f"task assignments -> {output_dir}"
    )


if __name__ == "__main__":
    main()
