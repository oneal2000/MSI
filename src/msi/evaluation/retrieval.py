"""Validate closed-world benchmark retrieval before evaluation.

Validation is deliberately stateless: a successful check exits zero and a
failed check exits non-zero.  No sidecar receipt is written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from msi.protocol import (
    DATASET_SKILL_COUNTS, SKILL_UNIVERSE_SIZE, inspect_skill_universe,
)


def validate_retrieval(
    *, retrieval: str | Path, instances: str | Path, corpus: str | Path,
    dataset: str, required_profile: str, allow_empty: bool = False,
) -> tuple[int, list[dict]]:
    """Return the retrieval row count and all validation issues."""
    corpus_path = Path(corpus).resolve()
    instances_path = Path(instances).resolve()
    retrieval_path = Path(retrieval).resolve()
    corpus_rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    metadata = retrieval.get("metadata", {})
    results = retrieval.get("results", [])
    issues: list[dict] = []

    skill_ids, counts, valid_universe = inspect_skill_universe(corpus_rows)
    if not valid_universe:
        issues.append({"type": "invalid_closed_world_corpus", "counts": counts})
    skill_set = set(skill_ids)

    expected_ids = [str(row.get("instance_id", "")) for row in instances]
    result_ids = [str(row.get("instance_id", "")) for row in results]
    if not all(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        issues.append({"type": "invalid_instance_ids"})
    if Counter(result_ids) != Counter(expected_ids):
        issues.append({
            "type": "retrieval_coverage_mismatch",
            "expected": len(expected_ids), "actual": len(result_ids),
        })

    actual_profile = str(metadata.get("retriever", ""))
    accepted = {
        "bge_base": {"bge", "bge_base"},
        "bge_base_rerank": {"bge_base_rerank"},
        "bge_m3": {"bge_m3"},
        "bge_m3_rerank": {"bge_m3_rerank"},
        "bm25": {"bm25"},
        "bm25_rerank": {"bm25_rerank"},
    }.get(required_profile, {required_profile})
    if actual_profile not in accepted:
        issues.append({
            "type": "retrieval_profile_mismatch",
            "required": required_profile, "actual": actual_profile,
        })
    if metadata.get("dataset") != dataset:
        issues.append({
            "type": "retrieval_dataset_mismatch",
            "required": dataset, "actual": metadata.get("dataset"),
        })
    if int(metadata.get("corpus_size", -1)) != SKILL_UNIVERSE_SIZE:
        issues.append({"type": "retrieval_corpus_size_not_408"})
    if required_profile.endswith("_rerank") and not metadata.get("reranker"):
        issues.append({"type": "missing_reranker_provenance"})

    gold_by_id = {row["instance_id"]: set(row.get("skill_annotations", [])) for row in instances}
    for row in results:
        ranked = row.get("retrieved") or []
        if not ranked and not allow_empty:
            issues.append({"type": "empty_ranking", "instance_id": row.get("instance_id")})
            continue
        ranked_ids = [str(item.get("skill_id", "")) for item in ranked]
        if len(ranked_ids) != len(set(ranked_ids)):
            issues.append({
                "type": "duplicate_retrieved_skill", "instance_id": row.get("instance_id"),
            })
        if required_profile == "random_same_domain":
            gold = gold_by_id.get(row.get("instance_id"), set())
            if (len(ranked_ids) != 1 or ranked_ids[0] in gold
                    or not ranked_ids[0].startswith(dataset + "_")):
                issues.append({"type": "invalid_same_domain_non_gold_route",
                               "instance_id": row.get("instance_id")})
        unknown = sorted(set(ranked_ids) - skill_set)
        if unknown:
            issues.append({
                "type": "retrieved_skill_outside_408",
                "instance_id": row.get("instance_id"), "skills": unknown[:20],
            })

    return len(results), issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_SKILL_COUNTS), required=True)
    parser.add_argument("--required-profile", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--checkpoint-record")
    parser.add_argument("--frozen", action="store_true",
                        help="Validate distributed routes without loading retriever weights")
    args = parser.parse_args()
    rows, issues = validate_retrieval(
        retrieval=args.retrieval, instances=args.instances, corpus=args.corpus,
        dataset=args.dataset, required_profile=args.required_profile,
        allow_empty=args.allow_empty,
    )
    if args.required_profile == "bge_m3_ft_synthetic":
        if args.frozen:
            from msi.retriever.retrieve import query_fingerprint, REPRESENTATION
            payload = json.loads(Path(args.retrieval).read_text())
            meta = payload.get("metadata", {})
            identity = meta.get("identity", {})
            if (identity.get("epoch") != 3 or identity.get("representation") != REPRESENTATION
                    or meta.get("query_fingerprint") != query_fingerprint(
                        json.loads(Path(args.instances).read_text()))):
                issues.append({"type": "frozen_route_query_or_protocol_mismatch"})
        elif not args.checkpoint_record:
            issues.append({"type": "missing_checkpoint_identity"})
        else:
            from msi.retriever.retrieve import checkpoint_identity, check_cache
            try:
                identity, _ = checkpoint_identity(args.checkpoint_record, args.corpus)
                check_cache(json.loads(Path(args.retrieval).read_text()), identity=identity,
                            instances=json.loads(Path(args.instances).read_text()), dataset=args.dataset)
            except (ValueError, KeyError, OSError) as error:
                issues.append({"type": "retriever_identity_mismatch", "message": str(error)})
    if issues:
        print(json.dumps({"status": "FAIL", "issues": issues}, ensure_ascii=False, indent=2))
    else:
        print(f"Validated {rows} retrieval rows")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
