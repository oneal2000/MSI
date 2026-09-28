#!/usr/bin/env python3
"""Render frozen two-shot synthetic demos for text-only evaluation.

Selects positives from the configured synthetic training split, renders
every positive as a demo, and selects the two median-length ones. Output is a frozen artifact — inference reads it and
never samples live.
"""

import argparse
import json
from pathlib import Path

from msi import REPO_ROOT

from msi.training.dataset import load_training_data, split_train_val

CORPUS = REPO_ROOT / "data/protocol/skills.json"
N_TRAIN, N_VAL, SEED = 200, 80, 42
PER_DEMO_TOKEN_CAP = 4096


def render_direct(instance: dict) -> str:
    sample = instance["trajectory"]["training_samples"][0]
    roles = [m["role"] for m in sample["messages"]]
    if roles != ["user", "assistant"]:
        raise SystemExit(f"unexpected direct roles {roles} for {instance['instance_id']}")
    assistant = sample["messages"][-1]["content"]
    return f"Problem:\n{instance['question']}\nSolution:\n{assistant}"


def render_toolqa(instance: dict) -> str:
    trajectory = instance["trajectory"]["output"].strip()
    if not trajectory:
        raise ValueError(f"missing full ToolQA trajectory: {instance['instance_id']}")
    return f"Problem:\n{instance['question']}\nTrajectory:\n{trajectory}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render two synthetic training demos per skill")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="Replay the frozen published instance IDs")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--num-train", type=int, default=N_TRAIN)
    parser.add_argument("--num-val", type=int, default=N_VAL)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--fixed-val-offset", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing demos: {args.output}")
    selection = json.loads(args.selection.read_text()) if args.selection else None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    demos = {}
    for row in corpus:
        sid = str(row["skill_id"])
        dataset = sid.rsplit("_", 1)[0]
        source = args.pool / dataset / "trajectories" / f"{sid}.json"
        if not source.is_file():
            raise SystemExit(f"missing synthetic trajectory: {source}")
        data = load_training_data(str(source))
        split_train_val(data, n_train=args.num_train, n_val=args.num_val,
                        seed=args.split_seed, fix_val=True,
                        fixed_val_offset=args.fixed_val_offset)
        positives = data["instances"]
        if len(positives) != args.num_train:
            raise SystemExit(f"{sid}: split yielded {len(positives)} train positives")

        if selection is not None:
            frozen_ids = [demo["instance_id"] for demo in selection[sid]["demos"]]
            training = {inst["instance_id"]: inst for inst in positives}
            if len(frozen_ids) != 2 or len(set(frozen_ids)) != 2:
                raise SystemExit(f"{sid}: expected two distinct frozen IDs")
            if any(iid not in training for iid in frozen_ids):
                raise SystemExit(f"{sid}: frozen demo is not a training positive")
            positives = [training[iid] for iid in frozen_ids]

        render = render_toolqa if dataset == "toolqa" else render_direct
        scored = []
        for order, inst in enumerate(positives):
            text = render(inst)
            tokens = len(tok(text, add_special_tokens=False)["input_ids"])
            scored.append({"order": order, "tokens": tokens, "text": text,
                           "instance_id": inst["instance_id"]})
        # Frozen rule: the two median-length positives; ties by split order;
        # skip any demo over the per-demo cap (take next candidate).
        ranked = sorted(scored, key=lambda d: (d["tokens"], d["order"]))
        picked, skipped = [], 0
        cursor = (len(ranked) - 2) // 2  # 100th/101st of 200 (0-based 99/100)
        while len(picked) < 2 and cursor < len(ranked):
            cand = ranked[cursor]
            if cand["tokens"] <= PER_DEMO_TOKEN_CAP:
                picked.append(cand)
            else:
                skipped += 1
            cursor += 1
        if len(picked) != 2:
            raise SystemExit(f"{sid}: cannot pick 2 demos under cap")
        if selection is not None:
            by_id = {row["instance_id"]: row for row in scored}
            picked = [by_id[row["instance_id"]] for row in selection[sid]["demos"]]
            if len(picked) != 2 or len({row["instance_id"] for row in picked}) != 2:
                raise SystemExit(f"{sid}: expected two distinct frozen training IDs")
            if any(row["tokens"] > PER_DEMO_TOKEN_CAP for row in picked):
                raise SystemExit(f"{sid}: frozen demo exceeds token cap")
            skipped = selection[sid]["capped_skips"]
        demos[sid] = {"demos": [
            {"instance_id": p["instance_id"], "tokens": p["tokens"], "text": p["text"]}
            for p in picked
        ], "capped_skips": skipped}

    args.output.write_text(json.dumps(demos, ensure_ascii=False), encoding="utf-8")
    print(f"Rendered {len(demos)} skills to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
