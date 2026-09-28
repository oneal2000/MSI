"""Multi-GPU contrastive router training with a configurable global batch."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from sentence_transformers import SentenceTransformer
from sragents.corpus import skill_text
from msi.retriever.data import REPRESENTATION
from msi.retriever.loss import distributed_loss
from msi.utils import atomic_json


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return torch.cat([torch.nn.functional.normalize(self.model(block)["sentence_embedding"], dim=-1)
                          for block in features], dim=0)


@torch.no_grad()
def validate(model, rows, corpus, batch_size, rank=0, world=1):
    model.eval()
    skill_ids = [r["skill_id"] for r in corpus]
    lookup = {sid: i for i, sid in enumerate(skill_ids)}
    domains = sorted({sid.rsplit("_", 1)[0] for sid in skill_ids})
    counts = torch.zeros((len(skill_ids), 5), device=model.device, dtype=torch.float64)
    docs = model.encode([skill_text(s) for s in corpus], batch_size=batch_size,
                        normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False)
    local = rows[rank::world]
    totals = torch.zeros(5, device=model.device, dtype=torch.float64)
    domain_counts = torch.zeros((len(domains), 5), device=model.device, dtype=torch.float64)
    for start in range(0, len(local), batch_size):
        batch = local[start:start + batch_size]
        emb = model.encode([r["query"] for r in batch], batch_size=batch_size,
                           normalize_embeddings=True, convert_to_tensor=True,
                           show_progress_bar=False)
        # Match SR-Agents DenseRetriever scoring precision exactly.
        ranking = (emb @ docs.T).topk(10, dim=1).indices.tolist()
        for row, ranked in zip(batch, ranking):
            gold = {lookup[s] for s in row["gold_skill_ids"]}
            ranks = [i + 1 for i, sid in enumerate(ranked) if sid in gold]
            first = min(ranks, default=math.inf)
            stats = torch.tensor([1, first <= 1, first <= 5, first <= 10,
                                  0 if not ranks else 1 / first],
                                 dtype=torch.float64, device=model.device)
            totals += stats
            domain_counts[domains.index(row["dataset"])] += stats
            for sid in gold:
                counts[sid] += stats
    if world > 1:
        for value in (totals, counts, domain_counts):
            dist.all_reduce(value)
    result = {"queries": int(totals[0]), "micro_R@1": float(totals[1] / totals[0]),
              "R@5": float(totals[2] / totals[0]), "R@10": float(totals[3] / totals[0]),
              "MRR@10": float(totals[4] / totals[0])}
    present = counts[:, 0] > 0
    result["skill_macro_R@1"] = float((counts[present, 1] / counts[present, 0]).mean())
    result["datasets"] = {ds: {"queries": int(c[0]), "R@1": float(c[1] / c[0])}
                          for ds, c in zip(domains, domain_counts) if c[0] > 0}
    result["selection_R@1"] = sum(v["R@1"] for v in result["datasets"].values()) / len(result["datasets"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from msi.config import load_config
    from msi.runtime.pipeline import path
    cfg = load_config(args.config)
    params = cfg["defaults"]
    root = path(cfg, "work_dir") / cfg["run_id"]
    prepared = path(cfg, "prepared_data") if cfg["paths"].get("prepared_data") else root / "data.json"
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    seed = int(params["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    batch = int(params["batch_size"])
    if batch % world:
        raise ValueError("global batch must be divisible by the number of GPUs")
    local_batch = batch // world
    data = json.loads(prepared.read_text())
    corpus = data["corpus"]
    if data["representation"] != REPRESENTATION:
        raise ValueError("prepared representation mismatch")
    rows, dev = data["train"], data["dev"]
    smoke_steps = int(params.get("smoke_steps", 0))
    if params.get("dev_limit", 0):
        # Keep every domain represented in a smoke test.
        domains = sorted({r["dataset"] for r in dev})
        per_domain = max(1, int(params["dev_limit"]) // len(domains))
        dev = [r for ds in domains for r in
               sorted((r for r in dev if r["dataset"] == ds), key=lambda r: r["instance_id"])[:per_domain]]
    if (root / "selected.json").exists() or list(root.glob("epoch-*")):
        raise ValueError("training outputs exist; use a new run ID rather than overwrite weights")
    model_path = Path(cfg["model"]["path"])
    if not model_path.is_absolute():
        from msi import REPO_ROOT
        model_path = REPO_ROOT / model_path
    model = SentenceTransformer(str(model_path), device=f"cuda:{local_rank}",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = int(params["max_length"])
    model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder = Encoder(model)
    wrapped = DistributedDataParallel(encoder, device_ids=[local_rank],
                                      find_unused_parameters=True) if world > 1 else encoder
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(params["learning_rate"]), weight_decay=0.01)
    steps_per_epoch = min(math.ceil(len(rows) / batch), smoke_steps) if smoke_steps else math.ceil(len(rows) / batch)
    total_steps = steps_per_epoch * int(params["epochs"])
    warmup = int(total_steps * float(params["warmup_ratio"]))
    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0., (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ids = [s["skill_id"] for s in corpus]
    lookup = {sid: i for i, sid in enumerate(ids)}
    texts = [skill_text(s) for s in corpus]
    history = []
    best = None
    for epoch in range(1, int(params["epochs"]) + 1):
        wrapped.train()
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        if smoke_steps:
            domains = sorted({r['dataset'] for r in rows})
            by_domain = {ds: [i for i in order if rows[i]['dataset'] == ds] for ds in domains}
            # Exercise long ToolQA queries on every rank, not just short-query batches.
            order = [by_domain[ds][i] for i in range(math.ceil(smoke_steps * batch / len(domains)))
                     for ds in domains]
        order += order[:(-len(order)) % batch]
        started = time.monotonic()
        for step in range(steps_per_epoch):
            global_indices = order[step * batch:(step + 1) * batch]
            indices = global_indices[rank * local_batch:(rank + 1) * local_batch]
            items = [rows[i] for i in indices]
            positives = [lookup[r["gold_skill_ids"][(epoch - 1) % len(r["gold_skill_ids"])]] for r in items]
            negatives = [lookup[r["negative_id"]] for r in items]
            candidates = positives + negatives
            # Long task instructions must not pad every short skill document.
            groups = [[r["query"] for r in items], [texts[i] for i in candidates]]
            features = [{k: v.to(model.device) for k, v in model.tokenize(group).items()}
                        for group in groups]
            mask = torch.zeros((local_batch, len(ids)), dtype=torch.bool, device=model.device)
            for i, row in enumerate(items):
                mask[i, [lookup[s] for s in row["gold_skill_ids"]]] = True
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = wrapped(features)
                loss = distributed_loss(embeddings[:local_batch], embeddings[local_batch:],
                                        torch.tensor(candidates, device=model.device), mask,
                                        float(params["temperature"]))
            if not torch.isfinite(loss):
                raise ValueError("non-finite retriever loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if rank == 0 and (step % 100 == 0 or (step < 100 and step % 25 == 0) or step + 1 == steps_per_epoch):
                print(json.dumps({"epoch": epoch, "step": step + 1, "steps": steps_per_epoch,
                                  "loss": float(loss.detach()), "seconds": round(time.monotonic() - started, 1)}), flush=True)
        if world > 1:
            dist.barrier()
        metrics = validate(model, dev, corpus, int(params["eval_batch_size"]), rank, world)
        if rank == 0:
            checkpoint = root / f"epoch-{epoch}"
            model.save_pretrained(str(checkpoint))
            item = {"epoch": epoch, "checkpoint": str(checkpoint.resolve()), "metrics": metrics}
            history.append(item)
            if best is None or metrics["selection_R@1"] > best["metrics"]["selection_R@1"]:
                best = item
            atomic_json(root / "validation.json", history)
            print(json.dumps(item), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        atomic_json(root / "selected.json", {"schema": "msi.retriever-checkpoint",
                    "run_id": cfg["run_id"], **best, "base_model": str(model_path.resolve()),
                    "prepared_data": str(prepared.resolve()), "representation": REPRESENTATION,
                    "training": params, "global_batch_size": batch, "world_size": world,
                    "train_queries": len(rows), "dev_queries": len(dev),
                    "smoke": bool(smoke_steps or params.get("dev_limit", 0))})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
