"""Publish SR-Agents-compatible routes from one frozen checkpoint."""
import json
import hashlib
from pathlib import Path

from sragents.corpus import skill_text
from sragents.retrieve.dense import DenseRetriever
from sragents.retrieve.query import build_retrieval_query
from sragents.retrieve import compute_retrieval_metrics
from msi.retriever.data import REPRESENTATION
from msi.utils import atomic_json

PROFILE = "bge_m3_ft_synthetic"


def query_fingerprint(instances):
    # Cache identity must change when question text or prompt rendering changes,
    # even if the instance IDs stay the same.
    rendered = sorted((r['instance_id'], build_retrieval_query(r)) for r in instances)
    return hashlib.sha256(json.dumps(rendered, ensure_ascii=False).encode()).hexdigest()


def checkpoint_identity(selected_file, corpus_file, *, allow_smoke=False):
    selected = json.loads(Path(selected_file).read_text())
    if selected.get("smoke") and not allow_smoke:
        raise ValueError("smoke checkpoint cannot produce benchmark routes")
    checkpoint = Path(selected["checkpoint"])
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"missing checkpoint weights: {checkpoint}")
    prepared = json.loads(Path(selected["prepared_data"]).read_text())
    corpus = json.loads(Path(corpus_file).read_text())
    if prepared["corpus"] != corpus or selected["representation"] != REPRESENTATION:
        raise ValueError("checkpoint corpus/representation differs from inference")
    return {"checkpoint": str(checkpoint.resolve()), "run_id": selected["run_id"],
            "epoch": selected["epoch"], "representation": REPRESENTATION,
            "max_length": selected["training"]["max_length"],
            "task_pool_id": prepared["sources"]["task_pool_id"]}, corpus


def check_cache(payload, *, identity, instances, dataset):
    metadata = payload.get("metadata", {})
    expected = [r["instance_id"] for r in instances]
    actual = [r.get("instance_id") for r in payload.get("results", [])]
    if (metadata.get("identity") != identity or metadata.get("retriever") != PROFILE
            or metadata.get("dataset") != dataset or metadata.get("corpus_size") != 408
            or len(expected) != len(set(expected)) or len(actual) != len(set(actual))
            or set(expected) != set(actual)):
        raise ValueError("retrieval cache checkpoint/profile/coverage mismatch")
    if metadata.get('query_fingerprint') != query_fingerprint(instances):
        raise ValueError('retrieval cache query text/rendering mismatch')


def retrieve(*, selected_file, corpus_file, instances_file, output, batch_size=16, allow_smoke=False):
    identity, corpus = checkpoint_identity(selected_file, corpus_file, allow_smoke=allow_smoke)
    instances = json.loads(Path(instances_file).read_text())
    dataset = instances[0]["dataset"]
    output = Path(output)
    if output.exists():
        check_cache(json.loads(output.read_text()), identity=identity, instances=instances, dataset=dataset)
        print(f"reuse frozen routes: {output}", flush=True)
        return
    engine = DenseRetriever(identity["checkpoint"], query_prefix="", batch_size=batch_size,
                            device="cuda:0", dtype="bfloat16")
    engine.build_index([r["skill_id"] for r in corpus], [skill_text(r) for r in corpus])
    if engine._model.max_seq_length != identity["max_length"]:
        raise ValueError("reloaded checkpoint changed maximum sequence length")
    ranked = engine.retrieve([build_retrieval_query(r) for r in instances], top_k=10)
    rows = [{"instance_id": row["instance_id"],
             "gold_skill_ids": row.get("skill_annotations", []),
             "retrieved": [{"skill_id": sid, "score": score} for sid, score in result]}
            for row, result in zip(instances, ranked)]
    payload = {"metadata": {"retriever": PROFILE, "dataset": dataset, "corpus_size": len(corpus),
                            "top_k": 10, "identity": identity,
                            "query_fingerprint": query_fingerprint(instances)},
               "results": rows, "metrics": compute_retrieval_metrics(rows, top_k=10)}
    check_cache(payload, identity=identity, instances=instances, dataset=dataset)
    atomic_json(output, payload)
    print(json.dumps({"dataset": dataset, "metrics": payload["metrics"]}), flush=True)
