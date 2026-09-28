#!/usr/bin/env python3
"""Retrieve one deterministic shard of the published synthetic task pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sragents.corpus import load_corpus, skill_text
from sragents.retrieve import get as get_retriever
from sragents.retrieve.dense import DenseRetriever
from sragents.retrieve.query import build_retrieval_query
from msi.anchors.artifacts import (
    TASK_POOL_SCHEMA, load_task_pool, same_skill_ids,
)
from msi.anchors.reranker import CrossEncoderReranker
from msi.training.provenance import assert_shared_output
from msi.utils import atomic_json


RETRIEVAL_SCHEMA = "msi.anchor-task-retrieval"


def retrieval_metadata(
    *, retriever: str, model: str, reranker_model: str | None,
    first_stage_top_k: int, top_k: int, dtype: str, task_pool_id: str,
) -> dict:
    return {
        "source_kind": "synthetic_train_pool",
        "task_pool_schema": TASK_POOL_SCHEMA,
        "task_pool_id": task_pool_id,
        "retriever": retriever + ("_rerank" if reranker_model else ""),
        "model": model if retriever != "bm25" else "bm25",
        "reranker": reranker_model,
        "first_stage_top_k": first_stage_top_k,
        "top_k": top_k,
        "dtype": dtype if retriever != "bm25" else "not_applicable",
    }


def retrieval_payload_complete(
    payload: dict, *, metadata: dict, expected_ids: list[str],
    corpus_ids: set[str], shard_index: int | None = None,
    shard_count: int | None = None,
    expected_gold: dict[str, list[str]] | None = None,
) -> bool:
    actual_metadata = payload.get("metadata", {})
    expected_metadata = dict(metadata)
    if shard_index is not None or shard_count is not None:
        expected_metadata.update({
            "shard_index": shard_index,
            "shard_count": shard_count,
        })
    rows = payload.get("results", [])
    row_ids = [str(row.get("instance_id", "")) for row in rows]
    candidates_valid = all(
        len(row.get("retrieved", [])) == int(metadata["top_k"])
        and len({item.get("skill_id") for item in row.get("retrieved", [])})
        == len(row.get("retrieved", []))
        and all(item.get("skill_id") in corpus_ids for item in row.get("retrieved", []))
        for row in rows
    )
    gold_valid = expected_gold is None or all(
        same_skill_ids(
            row.get("gold_skill_ids"), expected_gold.get(row["instance_id"], [])
        )
        for row in rows if row.get("instance_id") in expected_gold
    )
    return (
        payload.get("schema") == RETRIEVAL_SCHEMA
        and actual_metadata == expected_metadata
        and row_ids == expected_ids
        and candidates_valid
        and gold_valid
    )


def retrieval_complete(
    path: str | Path, *, metadata: dict, expected_ids: list[str],
    corpus_ids: set[str], shard_index: int | None = None,
    shard_count: int | None = None,
    expected_gold: dict[str, list[str]] | None = None,
) -> bool:
    source = Path(path)
    if not source.is_file():
        return False
    payload = json.loads(source.read_text(encoding="utf-8"))
    return retrieval_payload_complete(
        payload, metadata=metadata, expected_ids=expected_ids,
        corpus_ids=corpus_ids, shard_index=shard_index, shard_count=shard_count,
        expected_gold=expected_gold,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--retriever", choices=("bge_base", "bge_m3", "bm25"),
                        default="bge_base")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker-model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--first-stage-top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    assert_shared_output(args.output)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("retrieval shard index/count are invalid")
    if args.first_stage_top_k < args.top_k:
        raise SystemExit("first-stage-top-k must be >= top-k")
    if min(
        args.batch_size, args.query_chunk_size, args.reranker_batch_size,
        args.reranker_max_length, args.top_k,
    ) < 1:
        raise SystemExit("retrieval batch, chunk, length, and top-k values must be positive")

    pool_metadata, all_questions = load_task_pool(args.task_pool)
    task_pool_id = str(pool_metadata.get("task_pool_id", ""))
    if not task_pool_id:
        raise SystemExit("published anchor task pool is missing task_pool_id")
    questions = all_questions[args.shard_index::args.shard_count]
    ids = [row["instance_id"] for row in questions]
    corpus = load_corpus(args.corpus)
    corpus_ids = [row["skill_id"] for row in corpus]
    corpus_set = set(corpus_ids)
    if args.first_stage_top_k > len(corpus_ids):
        raise SystemExit("first-stage-top-k exceeds the skill corpus size")
    invalid_sources = [row.get("instance_id") for row in questions if not (
        str(row.get("instance_id", "")).startswith("syn_")
        and isinstance(row.get("source_instance_ids"), list)
        and row.get("source_instance_ids")
        and isinstance(row.get("origin_skill_ids"), list)
        and row.get("origin_skill_ids")
        and set(row["origin_skill_ids"]) <= corpus_set
        and row.get("gold_skill_ids") == row.get("origin_skill_ids")
    )]
    if invalid_sources:
        raise SystemExit(
            "task pool is not the canonical synthetic training view; "
            f"sample={invalid_sources[:20]}"
        )
    metadata = retrieval_metadata(
        retriever=args.retriever, model=args.model,
        reranker_model=args.reranker_model,
        first_stage_top_k=args.first_stage_top_k, top_k=args.top_k,
        dtype=args.dtype, task_pool_id=task_pool_id,
    )
    output = Path(args.output)
    if retrieval_complete(
        output, metadata=metadata, expected_ids=ids, corpus_ids=corpus_set,
        shard_index=args.shard_index, shard_count=args.shard_count,
        expected_gold={
            row["instance_id"]: list(row.get("gold_skill_ids") or []) for row in questions
        },
    ):
        print(f"anchor retrieval shard is complete -> {output}")
        return
    if output.exists():
        raise SystemExit(
            f"existing anchor retrieval shard does not match current inputs: {output}; "
            "use a new run ID"
        )

    corpus_by_id = {row["skill_id"]: row for row in corpus}
    corpus_texts = [skill_text(row)[:2000] for row in corpus]
    if args.retriever == "bge_base":
        retriever = get_retriever(
            "bge", model_path=args.model, batch_size=args.batch_size,
            device=args.device, dtype=args.dtype,
            query_chunk_size=args.query_chunk_size,
        )
    elif args.retriever == "bge_m3":
        retriever = DenseRetriever(
            args.model, query_prefix="", batch_size=args.batch_size,
            device=args.device, dtype=args.dtype,
            query_chunk_size=args.query_chunk_size,
        )
    else:
        retriever = get_retriever("bm25")
    retriever.build_index(corpus_ids, corpus_texts)
    queries = [build_retrieval_query(row) for row in questions]
    ranked = retriever.retrieve(queries, top_k=args.first_stage_top_k)
    if args.reranker_model:
        reranker = CrossEncoderReranker(
            args.reranker_model, max_length=args.reranker_max_length,
            batch_size=args.reranker_batch_size,
            device=args.device, dtype=args.dtype,
        )
        candidate_lists = [
            [corpus_by_id[skill_id] for skill_id, _ in candidates]
            for candidates in ranked
        ]
        final_rows = reranker.rerank_many(queries, candidate_lists)
    else:
        final_rows = ranked
    results = []
    for row, reranked in zip(questions, final_rows):
        gold = row.get("gold_skill_ids") or []
        results.append({
            "instance_id": row["instance_id"],
            "gold_skill_ids": [gold] if isinstance(gold, str) else list(gold),
            "retrieved": [
                {"skill_id": skill_id, "score": score}
                for skill_id, score in reranked[:args.top_k]
            ],
        })
    atomic_json(output, {
        "schema": RETRIEVAL_SCHEMA,
        "metadata": {
            **metadata, "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
        "results": results,
    })
    print(f"saved retrieval shard {args.shard_index + 1}/{args.shard_count}: "
          f"{len(results)} rows -> {output}")


if __name__ == "__main__":
    main()
