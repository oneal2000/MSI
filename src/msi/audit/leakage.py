#!/usr/bin/env python3
"""Audit training inputs against benchmark test questions.

The check is stateless: success exits zero and failure exits non-zero.  No audit
receipt is persisted.  The 408-skill corpus is an allowed, declared closed-world
candidate universe; per-instance test questions and labels are not.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from msi.protocol import (
    DATASET_SKILL_COUNTS, SKILL_UNIVERSE_SIZE, inspect_skill_universe,
)
from msi.training.provenance import canonical


def normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def grams(text: str) -> set[str]:
    words = text.split()
    if len(words) >= 5:
        return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}
    compact = text.replace(" ", "")
    width = min(8, max(3, len(compact)))
    return {compact[i:i + width] for i in range(max(1, len(compact) - width + 1))}


def read_jsonish(path: str | Path):
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first in "[{":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def records(data):
    if isinstance(data, list):
        return data
    for key in ("instances", "samples", "records", "data"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return []


def question(record: dict) -> str:
    for key in ("question", "query", "source_question", "problem", "prompt"):
        if str(record.get(key, "")).strip():
            return str(record[key])
    return ""


def record_id(record: dict, fallback: str) -> str:
    for key in ("instance_id", "id", "query_id"):
        if str(record.get(key, "")).strip():
            return str(record[key])
    return fallback


def test_questions(paths: list[str]) -> list[dict]:
    rows = []
    for raw in paths:
        path = Path(raw).resolve()
        for index, row in enumerate(records(read_jsonish(path))):
            q = question(row)
            if q:
                rows.append({
                    "id": record_id(row, f"{path.name}:{index}"),
                    "question": q,
                    "normalized": normalize(q),
                    "source": str(path),
                })
    return rows


def build_near_index(tests: list[dict]):
    inverted: dict[str, set[int]] = defaultdict(set)
    test_grams = []
    for index, row in enumerate(tests):
        value = grams(row["normalized"])
        test_grams.append(value)
        for gram in value:
            inverted[gram].add(index)
    return inverted, test_grams


def near_matches(norm: str, tests: list[dict], inverted, test_grams, threshold: float):
    qgrams = grams(norm)
    overlap = Counter()
    for gram in qgrams:
        overlap.update(inverted.get(gram, ()))
    minimum = max(1, math.ceil(threshold * len(qgrams)))
    found = []
    for index, count in overlap.items():
        if count < minimum:
            continue
        union = len(qgrams | test_grams[index])
        score = count / union if union else 0.0
        if score >= threshold:
            found.append((tests[index], score))
    return sorted(found, key=lambda item: item[1], reverse=True)


def corpus_map(path: str) -> tuple[dict[str, dict], list[dict]]:
    rows = records(read_jsonish(path))
    mapping = {str(row.get("skill_id")): row for row in rows if row.get("skill_id")}
    if not mapping:
        raise SystemExit(f"empty or invalid skill corpus: {path}")
    return mapping, rows


def gold_ids(row: dict) -> list[str]:
    value = row.get(
        "gold_skill_ids",
        row.get("skill_annotations", row.get("skill_ids", row.get("skill_id", []))),
    )
    if isinstance(value, str):
        return [value]
    return [str(item) for item in (value or [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", action="append", default=[])
    parser.add_argument("--trajectory-dir", action="append", default=[])
    parser.add_argument("--anchor", action="append", default=[])
    parser.add_argument("--anchor-dir", action="append", default=[])
    parser.add_argument("--retriever-data", action="append", default=[])
    parser.add_argument("--test-instances", action="append", required=True,
                        help="Repeat for all four benchmark test instance files")
    parser.add_argument("--skill-corpus", required=True,
                        help="Declared closed-world skill corpus (the final 408-skill universe)")
    parser.add_argument("--near-threshold", type=float, default=0.92)
    parser.add_argument("--required-anchor-retriever", default="bge_base_rerank")
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="Restrict completeness checks for a hash-isolated pilot",
    )
    parser.add_argument(
        "--skill", action="append", default=[],
        help="Restrict completeness checks to exact declared skills",
    )
    args = parser.parse_args()

    def expand_skill_json(directories):
        expanded = []
        selected_cli = set(args.dataset)
        selected_skills_cli = set(args.skill)
        for directory in directories:
            for path in sorted(Path(directory).rglob("*.json")):
                # Two-stage synthetic roots contain both task and trajectory
                # artifacts.  Training audits intentionally consume only the
                # trajectory stage; tasks are audited at admission time and as
                # the anchor question pool below.
                if path.parent.name == "tasks":
                    continue
                if re.fullmatch(r"(?:theoremqa|medcalcbench|logicbench|toolqa)_\d+\.json",
                                path.name) and (
                                    (not selected_cli
                                     or path.stem.rsplit("_", 1)[0] in selected_cli)
                                    and (not selected_skills_cli
                                         or path.stem in selected_skills_cli)
                                ):
                    expanded.append(str(path))
        return expanded

    expanded_trajectory_dirs = {directory: expand_skill_json([directory])
                                for directory in args.trajectory_dir}
    expanded_anchor_dirs = {directory: expand_skill_json([directory])
                            for directory in args.anchor_dir}
    for paths in expanded_trajectory_dirs.values():
        args.trajectory.extend(paths)
    for paths in expanded_anchor_dirs.values():
        args.anchor.extend(paths)

    if not (args.trajectory or args.anchor or args.retriever_data):
        parser.error("at least one training input is required")

    corpus, corpus_rows = corpus_map(args.skill_corpus)
    tests = test_questions(args.test_instances)
    exact = defaultdict(list)
    for row in tests:
        exact[row["normalized"]].append(row)
    inverted, test_grams = build_near_index(tests)

    issues: list[dict] = []
    warnings: list[dict] = []
    input_counts: dict[str, int] = {}
    _, actual_skill_counts, valid_universe = inspect_skill_universe(corpus_rows)
    if not valid_universe or len(corpus) != SKILL_UNIVERSE_SIZE:
        issues.append({
            "type": "invalid_closed_world_skill_universe",
            "rows": len(corpus_rows), "unique_skill_ids": len(corpus),
            "expected_counts": dict(DATASET_SKILL_COUNTS),
            "actual_counts": actual_skill_counts,
        })
    if len(args.test_instances) != 4:
        issues.append({
            "type": "incomplete_test_suite_for_decontamination",
            "test_files": len(args.test_instances), "expected": 4,
        })
    selected_datasets = set(args.dataset) if args.dataset else set(DATASET_SKILL_COUNTS)
    if not selected_datasets <= set(DATASET_SKILL_COUNTS) or not selected_datasets:
        raise SystemExit(f"invalid dataset restriction: {sorted(selected_datasets)}")
    dataset_skill_ids = {
        skill_id for skill_id in corpus
        if skill_id.rsplit("_", 1)[0] in selected_datasets
    }
    selected_skills = set(args.skill)
    unknown_skills = selected_skills - set(corpus)
    outside_datasets = selected_skills - dataset_skill_ids
    if unknown_skills or outside_datasets:
        raise SystemExit(
            f"invalid skill restriction: unknown={sorted(unknown_skills)}, "
            f"outside selected datasets={sorted(outside_datasets)}"
        )
    expected_skill_ids = selected_skills or dataset_skill_ids
    for kind, expanded in (
        ("trajectory", expanded_trajectory_dirs), ("anchor", expanded_anchor_dirs),
    ):
        for directory, paths in expanded.items():
            present = {Path(path).stem for path in paths}
            if present != expected_skill_ids:
                issues.append({
                    "type": f"incomplete_{kind}_directory", "path": canonical(directory),
                    "expected": len(expected_skill_ids), "present": len(present),
                    "missing": sorted(expected_skill_ids - present)[:30],
                    "extra": sorted(present - expected_skill_ids)[:30],
                })
    trajectory_questions: dict[str, tuple[str, str]] = {}
    trajectory_ids: dict[str, tuple[str, str]] = {}
    cross_skill_duplicate_count = 0
    cross_skill_duplicate_sample = []

    def audit_file(raw: str, kind: str):
        nonlocal cross_skill_duplicate_count
        path = Path(raw).resolve()
        data = read_jsonish(path)
        rows = records(data)
        input_counts[str(path)] = len(rows)

        if kind == "anchor":
            provenance = data.get("provenance", {}) if isinstance(data, dict) else {}
            if provenance.get("source_kind") != "synthetic_train_pool":
                issues.append({"type": "anchor_source_not_synthetic", "path": str(path)})
            required_fields = [
                "model", "retriever", "retriever_model", "near_policy",
            ]
            if args.required_anchor_retriever.endswith("_rerank"):
                required_fields.append("reranker_model")
            for field in required_fields:
                if not provenance.get(field):
                    issues.append({
                        "type": "anchor_missing_provenance_field", "path": str(path),
                        "field": field,
                    })
            if provenance.get("retriever") != args.required_anchor_retriever:
                issues.append({
                    "type": "anchor_retriever_protocol_mismatch", "path": str(path),
                    "retriever": provenance.get("retriever"),
                    "required": args.required_anchor_retriever,
                })
            if provenance.get("near_policy") != "retrieval_top1_then_topk_loose":
                issues.append({
                    "type": "anchor_near_policy_mismatch", "path": str(path),
                    "near_policy": provenance.get("near_policy"),
                })
            target_skill = data.get("skill_id") if isinstance(data, dict) else None
            if target_skill not in corpus:
                issues.append({
                    "type": "anchor_unknown_target_skill", "path": str(path),
                    "skill_id": target_skill,
                })
            def count_value(field):
                try:
                    return int(data.get(field, -1))
                except (TypeError, ValueError):
                    return -1
            requested_values = [count_value(f"n_{name}_requested")
                                for name in ("nearmiss", "random")]
            kept_values = [count_value(f"n_{name}_kept")
                           for name in ("nearmiss", "random")]
            requested = sum(requested_values)
            kept = sum(kept_values)
            if (any(value < 0 for value in requested_values + kept_values)
                    or kept_values != requested_values or len(rows) != kept):
                issues.append({
                    "type": "incomplete_anchor_counts", "path": str(path),
                    "rows": len(rows), "kept": kept, "requested": requested,
                })

        seen_ids: dict[str, str] = {}
        seen_questions: dict[str, str] = {}
        for index, row in enumerate(rows):
            q = question(row)
            rid = record_id(row, f"row:{index}")
            norm = normalize(q)
            if not norm:
                issues.append({"type": "missing_question", "path": str(path), "id": rid})
                continue
            if kind == "anchor" and not row.get("source_question"):
                issues.append({"type": "anchor_missing_source_question", "path": str(path), "id": rid})
            if kind == "anchor":
                target_skill = data.get("skill_id") if isinstance(data, dict) else None
                source_skills = row.get("source_skill_ids")
                if (not isinstance(source_skills, list) or not source_skills
                        or any(skill_id not in corpus for skill_id in source_skills)):
                    issues.append({
                        "type": "anchor_missing_or_unknown_source_skills",
                        "path": str(path), "id": rid, "source_skill_ids": source_skills,
                    })
                if target_skill in (source_skills or []):
                    issues.append({
                        "type": "anchor_is_not_negative", "path": str(path),
                        "id": rid, "skill_id": target_skill,
                    })
            if rid in seen_ids:
                issues.append({
                    "type": "duplicate_id", "path": str(path), "id": rid,
                    "conflicting_question": seen_ids[rid] != norm,
                })
            else:
                seen_ids[rid] = norm
            if norm in seen_questions:
                issues.append({
                    "type": "duplicate_normalized_question", "path": str(path),
                    "id": rid, "first_id": seen_questions[norm],
                })
            else:
                seen_questions[norm] = rid

            if kind == "trajectory":
                previous_id = trajectory_ids.get(rid)
                if previous_id and previous_id[0] != str(path):
                    issues.append({
                        "type": "cross_trajectory_duplicate_id", "path": str(path),
                        "id": rid, "first_path": previous_id[0],
                        "conflicting_question": previous_id[1] != norm,
                    })
                else:
                    trajectory_ids[rid] = (str(path), norm)
                previous_question = trajectory_questions.get(norm)
                if previous_question and previous_question[0] != str(path):
                    cross_skill_duplicate_count += 1
                    if len(cross_skill_duplicate_sample) < 20:
                        cross_skill_duplicate_sample.append({
                            "path": str(path), "id": rid,
                            "first_path": previous_question[0],
                            "first_id": previous_question[1],
                        })
                else:
                    trajectory_questions[norm] = (str(path), rid)

            if norm in exact:
                issues.append({
                    "type": "exact_test_question_overlap", "path": str(path), "id": rid,
                    "test_ids": [item["id"] for item in exact[norm]],
                })
            else:
                matches = near_matches(norm, tests, inverted, test_grams, args.near_threshold)
                if matches:
                    issues.append({
                        "type": "near_test_question_overlap", "path": str(path), "id": rid,
                        "test_id": matches[0][0]["id"], "score": round(matches[0][1], 4),
                    })

            if kind == "retriever":
                for skill_id in gold_ids(row):
                    skill = corpus.get(skill_id, {})
                    name_norm = normalize(skill.get("name", ""))
                    if name_norm and name_norm in norm:
                        issues.append({
                            "type": "gold_skill_name_in_query", "path": str(path),
                            "id": rid, "skill_id": skill_id, "skill_name": skill.get("name"),
                        })

        if not rows:
            issues.append({"type": "empty_training_input", "path": str(path)})

    for path in args.trajectory:
        audit_file(path, "trajectory")
    for path in args.anchor:
        audit_file(path, "anchor")
    for path in args.retriever_data:
        audit_file(path, "retriever")
    if cross_skill_duplicate_count:
        warnings.append({
            "type": "cross_skill_duplicate_questions_retained",
            "count": cross_skill_duplicate_count,
            "sample": cross_skill_duplicate_sample,
            "policy": "allowed_in_per_skill_training_globally_merged_for_anchors",
        })
    if issues:
        print(f"Validation failed: {len(input_counts)} inputs; {len(issues)} issues")
    else:
        print(f"Validated {len(input_counts)} inputs; no issues")
    for issue in issues[:20]:
        print(json.dumps(issue, ensure_ascii=False, sort_keys=True))
    if len(issues) > 20:
        print(f"... {len(issues) - 20} additional issues")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
