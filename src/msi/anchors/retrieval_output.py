"""Validate retrieval shards and atomically publish one ordered result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sragents.corpus import load_corpus
from msi.anchors.artifacts import load_task_pool
from msi.anchors.retrieve import (
    RETRIEVAL_SCHEMA, retrieval_complete, retrieval_payload_complete,
)
from msi.training.provenance import assert_shared_output
from msi.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_shared_output(args.output)

    _, task_rows = load_task_pool(args.task_pool)
    all_ids = [row["instance_id"] for row in task_rows]
    corpus_ids = {row["skill_id"] for row in load_corpus(args.corpus)}
    shard_count = len(args.input)
    base_metadata = None
    rows_by_id: dict[str, dict] = {}
    for expected_index, raw in enumerate(args.input):
        source = Path(raw)
        if not source.is_file():
            raise SystemExit(f"missing anchor retrieval shard: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}))
        shard_index = metadata.pop("shard_index", None)
        recorded_count = metadata.pop("shard_count", None)
        if shard_index != expected_index or recorded_count != shard_count:
            raise SystemExit(f"retrieval shard identity mismatch: {source}")
        if base_metadata is None:
            base_metadata = metadata
        elif metadata != base_metadata:
            raise SystemExit(f"retrieval shard provenance mismatch: {source}")
        expected_ids = all_ids[expected_index::shard_count]
        if not retrieval_payload_complete(
            payload, metadata=metadata, expected_ids=expected_ids,
            corpus_ids=corpus_ids, shard_index=expected_index,
            shard_count=shard_count,
            expected_gold={
                row["instance_id"]: list(row.get("gold_skill_ids") or [])
                for row in task_rows[expected_index::shard_count]
            },
        ):
            raise SystemExit(f"invalid or incomplete anchor retrieval shard: {source}")
        for row in payload["results"]:
            instance_id = row["instance_id"]
            if instance_id in rows_by_id:
                raise SystemExit(f"duplicate retrieval row across shards: {instance_id}")
            rows_by_id[instance_id] = row
    if set(rows_by_id) != set(all_ids):
        raise SystemExit("retrieval shards do not exactly cover the published task pool")
    assert base_metadata is not None
    ordered = [rows_by_id[instance_id] for instance_id in all_ids]
    destination = Path(args.output)
    if retrieval_complete(
        destination, metadata=base_metadata, expected_ids=all_ids,
        corpus_ids=corpus_ids,
        expected_gold={
            row["instance_id"]: list(row.get("gold_skill_ids") or []) for row in task_rows
        },
    ):
        print(f"published anchor retrieval is already complete -> {destination}")
        return
    if destination.exists():
        raise SystemExit(
            f"published anchor retrieval does not match current inputs: {destination}"
        )
    atomic_json(destination, {
        "schema": RETRIEVAL_SCHEMA,
        "metadata": base_metadata,
        "results": ordered,
    })
    print(f"published {len(ordered)} anchor retrieval rows -> {destination}")


if __name__ == "__main__":
    main()
