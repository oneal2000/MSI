"""Prepare an immutable synthetic-only train/dev view."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import random

from sragents.retrieve.query import build_retrieval_query
from msi.audit.leakage import normalize, grams, test_questions
from msi.protocol import inspect_skill_universe
from msi.utils import atomic_json

REPRESENTATION = {"query": "sragents.retrieve.query.build_retrieval_query",
                  "document": "sragents.corpus.skill_text", "query_prefix": ""}


class NearIndex:
    """Exact Jaccard threshold search using rare query grams to prune candidates."""
    def __init__(self, questions, threshold=0.92):
        self.threshold = threshold
        self.exact = set()
        self.values = []
        self.postings = defaultdict(set)
        for question in questions:
            norm = normalize(question)
            self.exact.add(norm)
            value = grams(norm)
            idx = len(self.values)
            self.values.append(value)
            for gram in value:
                self.postings[gram].add(idx)

    def matches(self, question):
        norm = normalize(question)
        if norm in self.exact:
            return True
        value = grams(norm)
        # A qualifying set must share >= ceil(t * |query|) grams. Therefore
        # at least one of any |query|-ceil(t*|query|)+1 grams must be shared.
        count = len(value) - math.ceil(self.threshold * len(value)) + 1
        prefix = sorted(value, key=lambda g: (len(self.postings.get(g, ())), g))[:count]
        candidates = set()
        for gram in prefix:
            candidates.update(self.postings.get(gram, ()))
        for idx in candidates:
            other = self.values[idx]
            if min(len(value), len(other)) < self.threshold * max(len(value), len(other)):
                continue
            intersection = len(value & other)
            if intersection >= self.threshold * (len(value) + len(other) - intersection):
                return True
        return False


def merge_rows(rows, skill_ids):
    merged = {}
    for row in rows:
        norm = normalize(row.get("question"))
        gold = set(row.get("gold_skill_ids", []))
        if not norm or not gold or not gold <= skill_ids:
            raise ValueError(f"invalid synthetic task: {row.get('instance_id')}")
        if not row.get("source_instance_ids") or gold != set(row.get("origin_skill_ids", [])):
            raise ValueError("synthetic task is missing its origin")
        if norm not in merged:
            merged[norm] = {"instance_id": row["instance_id"], "question": row["question"],
                            "source_instance_ids": [], "gold_skill_ids": [],
                            "pool_instance_ids": []}
        target = merged[norm]
        for key, values in [("source_instance_ids", row["source_instance_ids"]),
                            ("gold_skill_ids", gold), ("pool_instance_ids", [row["instance_id"]])]:
            target[key] = sorted(set(target[key]) | set(values))
    return list(merged.values())


def split_rows(rows, *, seed=42, dev_ratio=0.1, threshold=0.92):
    groups = defaultdict(list)
    for row in rows:
        groups[min(row["gold_skill_ids"])].append(row)
    train, dev = [], []
    rng = random.Random(seed)
    for skill in sorted(groups):
        values = sorted(groups[skill], key=lambda r: r["instance_id"])
        rng.shuffle(values)
        n = max(1, round(len(values) * dev_ratio))
        dev.extend(values[:n])
        train.extend(values[n:])
    index = NearIndex([r["question"] for r in dev], threshold)
    clean = [r for r in train if not index.matches(r["question"])]
    return clean, dev, len(train) - len(clean)


def prepare(config, root, paths):
    target = root / "data.json"
    if target.exists():
        raise ValueError(f"prepared inputs already exist; use a new run ID: {target}")
    pool = json.loads(paths["task_pool"].read_text())
    retrieval = json.loads(paths["anchor_retrieval"].read_text())
    corpus = json.loads(paths["corpus"].read_text())
    ids, _, valid = inspect_skill_universe(corpus)
    if not valid:
        raise ValueError("retriever requires the final 408-skill corpus")
    meta, retmeta = pool["metadata"], retrieval["metadata"]
    if (meta.get("source_kind") not in {"admitted_synthetic_task_directory",
                                       "synthetic_training_trajectories"}
            or retmeta.get("source_kind") != "synthetic_train_pool"
            or not meta.get("task_pool_id")
            or meta["task_pool_id"] != retmeta.get("task_pool_id")):
        raise ValueError("anchor retrieval does not belong to the admitted synthetic pool")
    ranking = {r["instance_id"]: r for r in retrieval["results"]}
    source = pool["instances"]
    if (len(ranking) != len(retrieval["results"]) or len(source) != len(ranking)
            or set(ranking) != {r["instance_id"] for r in source}):
        raise ValueError("synthetic retrieval coverage is not exact")
    for row in source:
        ret = ranking[row["instance_id"]]
        ranked = [v["skill_id"] for v in ret["retrieved"]]
        if (set(ret["gold_skill_ids"]) != set(row["gold_skill_ids"])
                or len(ranked) != len(set(ranked)) or not set(ranked) <= set(ids)):
            raise ValueError("invalid synthetic ranking or origin")
    rows = merge_rows(source, set(ids))
    settings = config["defaults"]
    test_paths = [str(paths["instances_dir"] / f"{ds}.json")
                  for ds in ("theoremqa", "logicbench", "toolqa", "medcalcbench")]
    index = NearIndex([r["question"] for r in test_questions(test_paths)],
                      settings["near_threshold"])
    clean, rejected = [], []
    for i, row in enumerate(rows):
        if index.matches(row["question"]):
            rejected.append(row["instance_id"])
        else:
            clean.append(row)
        if i % 20000 == 0:
            print(f"audit {i}/{len(rows)}; excluded={len(rejected)}", flush=True)
    train, dev, near_removed = split_rows(clean, seed=settings["seed"],
                                         dev_ratio=settings["dev_ratio"],
                                         threshold=settings["near_threshold"])
    rng = random.Random(settings["seed"])
    for split in (train, dev):
        counts = Counter(s for row in split for s in row["gold_skill_ids"])
        if set(counts) != set(ids):
            raise ValueError(f"split missing skills: {set(ids) - set(counts)}")
        for row in split:
            row["dataset"] = min(row["gold_skill_ids"]).rsplit("_", 1)[0]
            row["query"] = build_retrieval_query({"dataset": row["dataset"],
                                                   "question": row["question"]})
            candidates = list(dict.fromkeys(
                item["skill_id"] for rid in row["pool_instance_ids"]
                for item in ranking[rid]["retrieved"]
                if item["skill_id"] not in row["gold_skill_ids"]))
            if not candidates:
                raise ValueError(f"no hard negative for {row['instance_id']}")
            row["negative_id"] = rng.choice(candidates)
    summary = {"source_questions": len(source), "train": len(train), "dev": len(dev),
               "benchmark_excluded": len(rejected), "train_dev_near_excluded": near_removed,
               "train_per_skill": dict(Counter(s for r in train for s in r["gold_skill_ids"])),
               "dev_per_skill": dict(Counter(s for r in dev for s in r["gold_skill_ids"]))}
    atomic_json(target, {"schema": "msi.retriever-data", "corpus": corpus,
                        "representation": REPRESENTATION,
                        "sources": {"task_pool": str(paths["task_pool"]),
                                    "task_pool_id": meta["task_pool_id"],
                                    "anchor_retrieval": retmeta},
                        "settings": settings, "summary": summary,
                        "excluded_benchmark_ids": rejected, "train": train, "dev": dev})
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("per_skill")}), flush=True)
