#!/usr/bin/env python3
"""Cross-encoder reranking shared by anchor construction and evaluation.

The paper protocol pairs unmodified BGE-base-en-v1.5 with
BGE-reranker-v2-m3. Candidate text is rendered by the same ``skill_text``
function as the first-stage index.
"""
import argparse
import json
import time
from pathlib import Path

from sragents.corpus import skill_text
from sragents.retrieve.query import build_retrieval_query
from sragents.retrieve.metrics import compute_retrieval_metrics
from msi.training.provenance import assert_shared_output
from msi.protocol import SKILL_UNIVERSE_SIZE, inspect_skill_universe


class CrossEncoderReranker:
    """Score (query, skill_text) pairs with a cross-encoder; rerank by score desc.

    Mirrors LLMReranker.rerank(query, candidates) -> [(skill_id, score), ...]."""
    def __init__(
        self, model_path, max_length=512, batch_size=32,
        device=None, dtype="float32",
    ):
        from sentence_transformers import CrossEncoder
        import torch
        print(f"  Loading cross-encoder: {model_path}", flush=True)
        self._model = CrossEncoder(model_path, max_length=max_length, device=device)
        self._batch_size = batch_size
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtypes:
            raise ValueError(f"unsupported reranker dtype: {dtype}")
        model_device = next(self._model.model.parameters()).device
        if dtype != "float32":
            if not str(model_device).startswith("cuda"):
                raise ValueError(f"{dtype} reranking requires a CUDA device")
            self._model.model.to(dtype=dtypes[dtype])
        self._device_type = "cuda" if str(model_device).startswith("cuda") else "cpu"
        self._torch_dtype = dtypes[dtype]

    def _predict(self, pairs):
        import torch
        enabled = self._device_type == "cuda" and self._torch_dtype != torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self._device_type, dtype=self._torch_dtype, enabled=enabled,
        ):
            scores = self._model.predict(
                pairs, batch_size=self._batch_size,
                convert_to_numpy=False, convert_to_tensor=True,
            )
        return scores.float().cpu().numpy()

    def rerank(self, query, candidates):
        if not candidates:
            return []
        pairs = [(query, skill_text(c)) for c in candidates]
        scores = self._predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [(candidates[i]["skill_id"], float(scores[i])) for i in order]

    def rerank_many(self, queries, candidate_lists, query_chunk_size=256):
        """Rerank many independent queries without one ``predict`` call each.

        Pair scores and per-query sorting are identical to :meth:`rerank`; only
        Python/model dispatch is batched. Query chunking bounds host memory for
        the 408-skill synthetic pool.
        """
        if len(queries) != len(candidate_lists):
            raise ValueError("queries and candidate_lists must have equal length")
        outputs = []
        for start in range(0, len(queries), query_chunk_size):
            chunk_queries = queries[start:start + query_chunk_size]
            chunk_candidates = candidate_lists[start:start + query_chunk_size]
            pairs = []
            offsets = [0]
            for query, candidates in zip(chunk_queries, chunk_candidates):
                pairs.extend((query, skill_text(candidate)) for candidate in candidates)
                offsets.append(len(pairs))
            if pairs:
                scores = self._predict(pairs)
            else:
                scores = []
            for index, candidates in enumerate(chunk_candidates):
                begin, end = offsets[index], offsets[index + 1]
                local_scores = scores[begin:end]
                order = sorted(
                    range(len(candidates)), key=lambda item: local_scores[item], reverse=True
                )
                outputs.append([
                    (candidates[item]["skill_id"], float(local_scores[item]))
                    for item in order
                ])
        return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="first-stage BGE-base retrieval JSON")
    ap.add_argument("--output", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--corpus", required=True,
                    help="declared closed-world 408-skill corpus")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="bge_base_rerank")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    assert_shared_output(args.output)

    data = json.loads(Path(args.input).read_text())
    src = {r["instance_id"]: r for r in data["results"]}
    instances = {i["instance_id"]: i for i in json.load(open(args.instances))}
    corpus_rows = json.load(open(args.corpus))
    _, _, valid_universe = inspect_skill_universe(corpus_rows)
    corpus = {s["skill_id"]: s for s in corpus_rows}
    if not valid_universe:
        raise SystemExit(
            f"closed-world corpus must contain the declared "
            f"{SKILL_UNIVERSE_SIZE}-skill universe, got {len(corpus)}"
        )
    if set(src) != set(instances):
        raise SystemExit(
            f"first-stage retrieval coverage differs from benchmark: "
            f"retrieval={len(src)}, instances={len(instances)}"
        )

    reranker = CrossEncoderReranker(args.model, max_length=args.max_length,
                                    batch_size=args.batch_size)

    out = {}
    t0 = time.time()
    rerank_ids = []
    rerank_queries = []
    rerank_candidates = []
    for iid in src:
        inst = instances.get(iid)
        if inst is None:
            continue
        orig = src.get(iid, {}).get("retrieved", [])[:args.top_k]
        cskills = [corpus[c["skill_id"]] for c in orig if c["skill_id"] in corpus]
        if not cskills:
            out[iid] = orig
            continue
        query = build_retrieval_query(inst)
        rerank_ids.append(iid)
        rerank_queries.append(query)
        rerank_candidates.append(cskills)
    reranked_many = reranker.rerank_many(rerank_queries, rerank_candidates)
    for n, (iid, reranked) in enumerate(zip(rerank_ids, reranked_many), start=1):
        out[iid] = [{"skill_id": s, "score": sc} for s, sc in reranked]
        if n % 100 == 0:
            print(f"  {n}/{len(src)} ({(time.time()-t0)/n:.2f}s/inst)", flush=True)

    results = []
    for iid, rec in src.items():
        results.append({
            "instance_id": iid,
            "gold_skill_ids": rec.get("gold_skill_ids", []),
            "retrieved": out.get(iid, rec.get("retrieved", []))[:args.top_k],
        })
    metrics = compute_retrieval_metrics(results, top_k=args.top_k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    meta = {**data.get("metadata", {}), "retriever": args.name,
            "source": str(args.input),
            "reranker": args.model, "top_k": args.top_k}
    Path(args.output).write_text(json.dumps(
        {"metadata": meta, "metrics": metrics, "results": results},
        ensure_ascii=False, indent=2))
    print(f"  Saved {args.output}  Recall@1={metrics.get('Recall@1'):.4f} "
          f"(source {data['metadata'].get('retriever','?')} "
          f"Recall@1={data.get('metrics',{}).get('Recall@1'):.4f})", flush=True)


if __name__ == "__main__":
    main()
