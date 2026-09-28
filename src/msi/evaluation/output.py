"""Merge work-local inference progress or shards into one final result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msi.training.provenance import assert_shared_output


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_shared_output(args.output)

    instances = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    expected_ids = [str(row["instance_id"]) for row in instances]
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("benchmark instances contain duplicate instance_id values")

    inputs = [Path(value).resolve() for value in args.input]
    if len(inputs) != len(set(inputs)):
        raise SystemExit("duplicate inference input path")
    expected_id_set = set(expected_ids)
    all_rows: dict[str, dict] = {}
    errored: list[str] = []

    for source in inputs:
        if not source.is_file():
            raise SystemExit(f"missing inference progress: {source}")
        source_rows = _read_jsonl(source)
        for row in source_rows:
            instance_id = str(row.get("instance_id", ""))
            if not instance_id or instance_id in all_rows:
                raise SystemExit(f"empty/duplicate inference instance: {instance_id!r}")
            if instance_id not in expected_id_set:
                raise SystemExit(f"unexpected inference instance: {instance_id!r}")
            if row.get("error"):
                # Generation failures (for example ReAct contexts that overflow
                # max_model_len) are carried through as failed instances so the
                # scoring step counts them as incorrect instead of aborting the
                # whole evaluate run.  The row keeps its ``error`` field for
                # auditability.
                errored.append(instance_id)
            elif (not str(row.get("raw_output", "")).strip()
                  and not bool(row.get("meta", {}).get("failed"))):
                raise SystemExit(f"inference output is empty for {instance_id}")
            if row.get("dataset") != args.dataset:
                raise SystemExit(f"inference dataset mismatch for {instance_id}")
            if row.get("method") != args.method:
                raise SystemExit(f"inference method mismatch for {instance_id}")
            all_rows[instance_id] = row
    if errored:
        preview = ", ".join(errored[:5])
        suffix = "" if len(errored) <= 5 else f" (+{len(errored) - 5} more)"
        print(
            f"carrying {len(errored)} errored instances into scoring as "
            f"failures: {preview}{suffix}"
        )

    if set(all_rows) != expected_id_set:
        raise SystemExit(
            f"inference coverage mismatch: rows={len(all_rows)}, "
            f"expected={len(expected_ids)}"
        )
    output = Path(args.output).resolve()
    _atomic_text(
        output,
        "".join(
            json.dumps(all_rows[instance_id], ensure_ascii=False) + "\n"
            for instance_id in expected_ids
        ),
    )
    print(f"Merged {len(all_rows)} inference rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
