#!/usr/bin/env python3
"""Generate each model's plain answer once for the anchor candidate union."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from msi.models.llm_client import _chat, create_client, last_finish_reason
from sragents.llm import get_extra_body
from sragents.prompts import build_prompt
from msi.anchors.completion import (
    answer_cache_complete, answer_is_retryable,
)
from msi.anchors.artifacts import (
    assigned_instance_ids, load_assignments, load_task_pool,
)
from msi.training.provenance import assert_shared_output


def atomic_write(path: Path, payload: dict):
    # First-run cache paths are model/skill scoped and therefore do not exist
    # yet.  Create the durable parent before the sibling temporary file; the
    # process-wide shared umask has already been applied by the launcher.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pool", required=True)
    parser.add_argument("--assignment", action="append", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--flush-every", type=int, default=100)
    args = parser.parse_args()
    assert_shared_output(args.output)

    _, rows = load_task_pool(args.task_pool)
    by_id = {row["instance_id"]: row for row in rows}
    wanted = assigned_instance_ids(load_assignments(args.assignment))
    if any(iid not in by_id for iid in wanted):
        raise SystemExit("anchor assignment references an ID outside the published task pool")

    output = Path(args.output)
    payload = {
        "schema": "msi.base-answer-cache",
        "provenance": {
            "source_kind": "synthetic_train_pool",
            "model": args.base_model,
            "served_model": args.served_model,
            "max_tokens": args.max_tokens,
        },
        "answers": {},
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("provenance") != payload["provenance"]:
            raise SystemExit("existing answer cache provenance differs; choose a new output path")
        payload = existing
    if answer_cache_complete(
        output, args.task_pool, args.assignment, args.base_model,
        args.served_model, args.max_tokens,
    ):
        print(f"answer cache already complete -> {output}")
        return
    todo = [
        iid for iid in wanted
        if answer_is_retryable(payload["answers"].get(iid), by_id[iid])
    ]
    print(f"answer cache: wanted={len(wanted)} existing={len(payload['answers'])} todo={len(todo)}")
    client = create_client(base_url=args.api_base)
    extra = get_extra_body(args.base_model, thinking=False)

    def work(iid):
        system, user = build_prompt(by_id[iid], skills=None)
        try:
            answer = _chat(
                client, args.served_model, system, [{"role": "user", "content": user}],
                max_tokens=args.max_tokens, extra_body=extra, temperature=0.0,
            )
            return iid, answer, last_finish_reason(), None
        except Exception as error:  # noqa: BLE001
            return iid, "", None, f"{type(error).__name__}: {str(error)[:200]}"

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, iid) for iid in todo]
        for future in as_completed(futures):
            iid, answer, finish_reason, error = future.result()
            payload["answers"][iid] = {
                "question": by_id[iid].get("question"),
                "dataset": by_id[iid].get("dataset"),
                "answer": answer, "finish_reason": finish_reason, "error": error,
            }
            completed += 1
            if completed % args.flush_every == 0:
                atomic_write(output, payload)
                print(f"completed {completed}/{len(todo)}", flush=True)
    atomic_write(output, payload)
    failures = sum(
        answer_is_retryable(payload["answers"].get(iid), by_id[iid])
        for iid in wanted
    )
    skipped = sum(
        bool(payload["answers"].get(iid, {}).get("answer"))
        and not payload["answers"].get(iid, {}).get("error")
        and payload["answers"].get(iid, {}).get("finish_reason") != "stop"
        for iid in wanted
    )
    print(f"saved {len(payload['answers'])} answers "
          f"({failures} retryable, {skipped} non-stop-skipped) -> {output}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
