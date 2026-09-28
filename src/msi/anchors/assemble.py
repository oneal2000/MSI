#!/usr/bin/env python3
"""Assemble per-skill anchor files from a model's deduplicated answer cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sragents.prompts import build_prompt
from msi.training.provenance import assert_shared_output
from msi.utils import atomic_json
from msi.anchors.artifacts import (
    assigned_instance_ids, load_assignments, load_task_pool,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--assignment", action="append", required=True)
    parser.add_argument("--answer-cache", required=True)
    parser.add_argument("--skill-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    assert_shared_output(args.output_dir)

    _, pool = load_task_pool(args.task_pool)
    by_id = {row["instance_id"]: row for row in pool}
    assignments = load_assignments(args.assignment)
    cache = json.loads(Path(args.answer_cache).read_text(encoding="utf-8"))
    corpus = {
        row["skill_id"]: row for row in json.loads(
            Path(args.skill_corpus).read_text(encoding="utf-8")
        )
    }
    wanted = set(assigned_instance_ids(assignments))
    if not wanted <= set(by_id):
        raise SystemExit("anchor assignment references questions outside the task pool")
    if not set(assignments) <= set(corpus):
        raise SystemExit("anchor assignment references skills outside the current corpus")
    if cache.get("provenance", {}).get("source_kind") != "synthetic_train_pool":
        raise SystemExit("answer cache is not from the synthetic training pool")
    if cache.get("provenance", {}).get("model") != args.base_model:
        raise SystemExit("answer cache targets a different logical base model")
    if not cache.get("provenance", {}).get("served_model"):
        raise SystemExit("answer cache is missing its endpoint served-model provenance")
    changed_inputs = [
        instance_id for instance_id in wanted
        if (
            cache.get("answers", {}).get(instance_id, {}).get("question")
            != by_id[instance_id].get("question")
            or cache.get("answers", {}).get(instance_id, {}).get("dataset")
            != by_id[instance_id].get("dataset")
        )
    ]
    if changed_inputs:
        raise SystemExit(
            "answer cache does not match current synthetic questions: "
            f"{changed_inputs[:20]}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = []
    for skill_id, candidates in assignments.items():
        skill_content = corpus[skill_id]["content"]
        sampling = candidates["sampling"]
        requested_near = int(sampling["n_nearmiss"])
        requested_random = int(sampling["n_random"])

        def collect(ids, limit):
            samples = []
            for iid in ids:
                cache_row = cache["answers"].get(iid, {})
                answer = cache_row.get("answer", "")
                finish_reason = cache_row.get("finish_reason")
                instance = by_id[iid]
                # Only a normal endpoint stop is a complete behavioral target.
                # Markers may occur well before an aborted or capped runaway's
                # truncated tail, so they cannot make a non-stop output usable.
                if not answer or finish_reason != "stop":
                    continue
                system, injected = build_prompt(instance, skills=[skill_content])
                samples.append({
                    "instance_id": iid,
                    "source_question": instance["question"],
                    "source_skill_ids": list(instance.get("origin_skill_ids") or []),
                    "system": system,
                    "messages": [
                        {"role": "user", "content": injected},
                        {"role": "assistant", "content": answer},
                    ],
                })
                if len(samples) == limit:
                    break
            return samples

        near = collect(candidates["near"], requested_near)
        random_rows = collect(candidates["random"], requested_random)
        payload = {
            "skill_id": skill_id,
            "provenance": {
                "source_kind": "synthetic_train_pool",
                "retriever": candidates["retrieval"]["profile"],
                "retriever_model": candidates["retrieval"]["model"],
                "reranker_model": candidates["retrieval"].get("reranker"),
                "near_policy": sampling["near_policy"],
                "model": cache["provenance"]["model"],
                "served_model": cache["provenance"].get("served_model"),
            },
            "n_nearmiss_kept": len(near),
            "n_random_kept": len(random_rows),
            "n_nearmiss_requested": requested_near,
            "n_random_requested": requested_random,
            "n_nearmiss_strict_candidates": candidates.get("strict_near_selected", 0),
            "n_nearmiss_loose_candidates": candidates.get("loose_near_selected", 0),
            "samples": near + random_rows,
        }
        dataset_dir = out_dir / skill_id.rsplit("_", 1)[0]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(dataset_dir / f"{skill_id}.json", payload)
        if len(near) < requested_near or len(random_rows) < requested_random:
            short.append({"skill_id": skill_id, "near": len(near), "random": len(random_rows)})
    print(f"assembled {len(assignments)} skills; shortfalls={len(short)} -> {out_dir}")
    if short:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
