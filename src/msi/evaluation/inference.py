#!/usr/bin/env python3
"""Run LLM inference on benchmark datasets.

Delegates to sragents' Provider × Engine pipeline (run_many).  LoRA adapter
selection is handled by grouping instances by adapter and switching the model
name for each group.

Final commands use an explicit work-local ``--output-dir``. Missing adapters or
retrieval rows fail rather than silently changing the requested method.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sragents.infer import get_engine, get_provider
from sragents.infer.runner import run_many

from msi.models.config import CORPUS_PATH, EXTERNAL_DIR, INSTANCES_DIR, RESULTS_DIR
from msi.models.qwen_thinking import apply_for as apply_qwen_thinking
from msi.models.inprocess_peft import InProcessClient, load_base_only, load_peft_with_adapters
from msi.models.vllm_lora import register_lora, unregister_lora
from msi.training.provenance import assert_shared_output
from msi.config import load_config

ALL_DATASETS = ["theoremqa", "medcalcbench", "logicbench", "toolqa"]
GOLDEN_LORA_METHODS = {
    "golden_lora", "golden_lora_notext_text", "golden_lora_text",
}
RETRIEVED_LORA_METHODS = {
    "retrieved_lora", "retrieved_lora_notext_text", "retrieved_lora_text",
}
LORA_METHODS = GOLDEN_LORA_METHODS | RETRIEVED_LORA_METHODS
# Mismatched adapter ablation: adapter follows a supplied route file's
# top-1 (a fixed same-domain non-gold sample) while tools/text stay with the
# instance's GOLD skill — decoupling adapter routing from tool supply.
MISMATCHED_LORA_METHODS = {"mismatched_lora"}
LORA_METHODS |= MISMATCHED_LORA_METHODS
# Synthetic demonstration baseline: NO adapter — the routed skill's text is
# augmented with its two frozen synthetic training-positive demos (pure ICL
# alternative to MSI). golden_* routes by the oracle, retrieved_* by the
# same frozen retriever file as the main table.
DEMOS_METHODS = {"golden_skill_2shot", "retrieved_skill_2shot"}
NO_TEXT_LORA_METHODS = {"golden_lora", "retrieved_lora"}
NO_TEXT_LORA_METHODS.add("mismatched_lora")
NO_RETRIEVAL_METHODS = (
    {"naive", "golden_skill", "golden_skill_2shot"} | GOLDEN_LORA_METHODS
)


class _TextFallbackProvider:
    """Strip skill text on adapter routes and retain it on fallback routes."""

    def __init__(self, inner, fallback_ids):
        self._inner = inner
        self._fallback = set(fallback_ids)

    def provide(self, instance: dict) -> list[dict]:
        skills = self._inner.provide(instance)
        if instance["instance_id"] in self._fallback:
            return skills
        return [{**skill, "content": ""} for skill in skills]


class _DemoProvider:
    """Append the routed skill's frozen 2-shot demos to its text (Experiment B).

    The demo payload comes from the frozen offline artifact (evaluation.demos);
    a routed skill missing from it is fatal — closed-world inference must not
    silently degrade to text-only. Injection rides the same provider seam the
    engines already consume ("Relevant Skill" block), so no engine changes.
    """

    def __init__(self, inner, demos: dict):
        self._inner = inner
        self._demos = demos

    def provide(self, instance: dict) -> list[dict]:
        out = []
        for skill in self._inner.provide(instance):
            payload = self._demos.get(skill["skill_id"])
            if not payload:
                raise SystemExit(
                    f"no frozen demos for skill {skill['skill_id']} "
                    f"(instance {instance['instance_id']})"
                )
            block = "\n\n---\n\n".join(d["text"] for d in payload["demos"])
            out.append({**skill, "content": (
                f"{skill.get('content', '')}\n\n"
                f"Worked examples of this skill:\n\n{block}"
            )})
        return out


# ---------------------------------------------------------------------------
# Corpus loading (cached)
# ---------------------------------------------------------------------------

_corpus_cache: dict[str, dict] | None = None


def _load_corpus() -> dict[str, dict]:
    global _corpus_cache
    if _corpus_cache is None:
        with open(CORPUS_PATH) as f:
            skills = json.load(f)
        _corpus_cache = {s["skill_id"]: s for s in skills}
    return _corpus_cache


# ---------------------------------------------------------------------------
# LoRA adapter discovery
# ---------------------------------------------------------------------------

def _discover_lora_adapter(lora_dir: Path, skill_id: str) -> Path | None:
    meta = lora_dir / skill_id / "metadata.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text())
        if d.get("skill_id") == skill_id:
            p = meta.parent / "adapter_model.safetensors"
            return p if p.exists() else None
    except (json.JSONDecodeError, KeyError):
        pass
    return None



def _group_by_adapter(
    instances: list[dict], lora_dir: Path, corpus: dict,
    base_model: str | None = None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Group by the first gold adapter and return explicit fallback rows."""
    groups: dict[str, list[dict]] = defaultdict(list)
    fallback: list[dict] = []
    for inst in instances:
        matched = False
        for sid in inst.get("skill_annotations", []):
            if sid in corpus and _discover_lora_adapter(lora_dir, sid):
                # Group key == adapter name == vLLM-registered lora_name
                # (basename of the adapter dir, e.g. "medcalcbench_052").
                # Requesting "skill_<sid>" here would not match any
                # registered adapter and every LoRA call would 404.
                groups[sid].append(inst)
                matched = True
                break
        if not matched:
            fallback.append(inst)
    return dict(groups), fallback


def _group_by_retrieved_adapter(
    instances: list[dict], retrieval_map: dict, lora_dir: Path,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Group instances by their RETRIEVED top-1 skill's adapter (not the oracle).

    Counterpart to :func:`_group_by_adapter` for the retrieved regime: each instance
    routes to the adapter of whatever skill the retriever returned first, not its own
    ``skill_annotations``. The closed-world 408 protocol has a complete adapter
    pool, so a missing row, empty ranking, or missing top-1 adapter is fatal.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    fallback: list[dict] = []
    for inst in instances:
        ret = retrieval_map.get(inst["instance_id"], [])
        sid = ret[0]["skill_id"] if ret else None
        if sid and _discover_lora_adapter(lora_dir, sid):
            groups[sid].append(inst)
        else:
            fallback.append(inst)
    return dict(groups), fallback


def _shard_adapter_groups(
    groups: dict[str, list[dict]],
    fallback: list[dict],
    shard_idx: int,
    num_shards: int,
) -> tuple[dict[str, list[dict]], list[dict], list[int]]:
    """Assign each adapter wholly to one shard while balancing row counts."""
    shard_groups: list[dict[str, list[dict]]] = [dict() for _ in range(num_shards)]
    shard_fallback: list[list[dict]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards

    for sid, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        shard_groups[target][sid] = group
        loads[target] += len(group)

    for instance in fallback:
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        shard_fallback[target].append(instance)
        loads[target] += 1

    return shard_groups[shard_idx], shard_fallback[shard_idx], loads


def _drop_unassigned_resume_rows(output_path: Path, assigned_ids: set[str]) -> int:
    """Remove rows left by an older shard assignment before resuming."""
    if not output_path.exists():
        return 0
    data = output_path.read_bytes()
    kept: list[bytes] = []
    removed = 0
    for raw in data.splitlines(keepends=True):
        try:
            instance_id = str(json.loads(raw.decode("utf-8"))["instance_id"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
            kept.append(raw)
            continue
        if instance_id in assigned_ids:
            kept.append(raw)
        else:
            removed += 1
    if removed:
        temporary = output_path.with_suffix(output_path.suffix + ".shard.tmp")
        temporary.write_bytes(b"".join(kept))
        temporary.replace(output_path)
    return removed


class _PerInstanceModelEngine:
    """Wraps an engine so each instance is sent to its own model name.

    LoRA inference must route every instance to its registered adapter name.
    Running every adapter group through ONE worker pool (instead of one ``run_many``
    per adapter, serially) keeps the vLLM server's batch deep and the GPU
    fed — each skill often covers only a handful of instances, so per-adapter
    pools would crater in-flight concurrency to single digits. This wrapper
    restores the per-instance model selection that a flat pool would lose.

    ``InferenceEngine`` is a structural Protocol and ``run_many`` only calls
    ``engine.run``, so duck-typing suffices (no vendor change needed).
    """

    def __init__(self, inner, model_by_instance: dict[str, str]):
        self._inner = inner
        self._map = model_by_instance

    def run(self, instance, skills, client, model, **kwargs):
        return self._inner.run(
            instance, skills, client,
            self._map.get(instance["instance_id"], model),
            **kwargs,
        )


def _build_inprocess_client(base_model_path, lora_dir, adapter_sids, default_max_tokens,
                            allow_base_model=False):
    """Load base + the covered skills' adapters into one PeftModel (PRAG-style).

    Returns an :class:`InProcessClient` whose ``model=`` selects
    ``set_adapter(sid)`` for a known skill id or ``disable_adapter()`` for the
    base name. Bypasses vLLM's LoRA engine entirely (a no-op on Qwen3.5).
    """
    adapter_paths = {}
    for sid in adapter_sids:
        d = Path(lora_dir) / sid
        if (d / "adapter_model.safetensors").exists():
            adapter_paths[sid] = str(d)
    if not adapter_paths:
        raise SystemExit(
            f"inprocess: no adapters found under {lora_dir} for {list(adapter_sids)}")
    print(f"  [inprocess] loading base {base_model_path} + "
          f"{len(adapter_paths)} adapters ...", flush=True)
    peft_model, tokenizer = load_peft_with_adapters(base_model_path, adapter_paths)
    request_to_adapter = {sid: sid for sid in adapter_paths}
    return InProcessClient(
        peft_model, tokenizer, request_to_adapter,
        default_max_tokens=default_max_tokens,
        allow_base_model=allow_base_model,
    )


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

def _make_pooled_client(api_base: str, workers: int):
    """OpenAI client with a connection pool sized for high-concurrency vLLM.

    The default openai/httpx pool caps well below a 2B server's --max-num-seqs
    (512), so a large --workers would be throttled client-side. Size the pool to
    the worker count so the server actually saturates.
    """
    import httpx
    from openai import OpenAI
    from msi.models.llm_client import request_timeout
    limits = httpx.Limits(max_connections=max(workers + 32, 128),
                          max_keepalive_connections=max(workers, 64))
    # Keep request timeout configurable for queued, long-form generations.
    return OpenAI(
        base_url=api_base, api_key="EMPTY",
        http_client=httpx.Client(
            limits=limits, timeout=httpx.Timeout(request_timeout(), connect=10.0)
        ),
    )


def _make_engine(dataset: str, engine_kwargs: dict):
    """Select the right sragents engine for a dataset."""
    if dataset == "toolqa":
        # toolqa_data_dir MUST be passed explicitly: react.py imports EXTERNAL_DIR
        # from sragents.config (the upstream default points to its external data,
        # which doesn't exist here), not lora.config, so the DB path would 404
        # and every Retrieve* call would silently fail → garbage toolqa results.
        kw = {"toolqa_data_dir": str(EXTERNAL_DIR / "toolqa"), **engine_kwargs}
        return get_engine("react", **kw)
    return get_engine("direct", **engine_kwargs)


# ---------------------------------------------------------------------------
# ToolQA environment pre-warming
# ---------------------------------------------------------------------------

def _prewarm_toolqa():
    from sragents.toolqa import ToolEnvironment
    from sragents.toolqa.tools.table import TableToolkit
    from sragents.toolqa.tools.graph import GraphToolkit

    corpus_dir = EXTERNAL_DIR / "toolqa"
    env = ToolEnvironment(str(corpus_dir))
    env._get_agenda_retriever()
    env._get_scirex_retriever()

    table = TableToolkit(corpus_dir)
    graph = GraphToolkit(corpus_dir)
    for db_name in ["flights", "coffee", "airbnb", "yelp"]:
        try:
            table.load_db(db_name)
            print(f"    {db_name} database ready")
        except Exception as e:
            print(f"    {db_name} skipped: {e}")
    try:
        graph.load_graph("dblp")
        print("    dblp graph ready")
    except Exception as e:
        print(f"    dblp skipped: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on benchmark")
    parser.add_argument("--name", type=str, choices=ALL_DATASETS, default=None,
                        help="Dataset name (default: all)")
    parser.add_argument("--method", type=str, default="naive",
                        help="Method: naive, golden_skill, golden_lora, "
                             "golden_lora_notext_text, golden_lora_text, "
                             "retrieved_lora, retrieved_lora_notext_text, retrieved_lora_text "
                             "(LoRA routed by the retriever's top-1 skill; needs "
                             "--retrieval-results), or a retrieval-based label "
                             "(text-only, e.g. bge_base_rerank_top1; needs "
                             "--retrieval-results).")
    parser.add_argument("--model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=20480,
                        help="Maximum output tokens; for ReAct this is the "
                             "cumulative trajectory budget")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for hybrid-thinking models")

    # Retrieval
    parser.add_argument("--retrieval-results", type=str, default=None)
    parser.add_argument("--demos-file", type=str, default=None,
                        help="Frozen 2-shot demo payload (data/demos/two_shot.json) for the "
                             "*_skill_2shot Experiment-B methods")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--lora-dir", type=str, default=None,
                        help="LoRA adapter directory (default: results/lora)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override results/ directory (e.g. when default is read-only)")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split each dataset across N processes; LoRA methods keep each "
                             "adapter wholly on one balanced shard")
    parser.add_argument("--base-model-path", type=str, default=None,
                        help="Filesystem path to the base model (required for --lora-backend inprocess)")
    parser.add_argument("--lora-backend", choices=["vllm", "inprocess"], default="inprocess",
                        help="vllm: serve via a student vLLM server (--api-base, dynamic LoRA, "
                             "high-concurrency batching); inprocess: PeftModel in-process "
                             "(sharded).")
    parser.add_argument("--max-loaded-loras", type=int, default=16,
                        help="Maximum adapter groups registered in one vLLM wave")
    parser.add_argument("--vl-rename", action="store_true",
                        help="Qwen3.5-specific: rename adapter keys to the VL nested-LM path "
                             "(language_model.*) before registering. Off by default (general for any "
                             "model); architecture detection enables it automatically.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[3] / "configs" / "eval" / "qwen3.5-2b.yaml"),
        help="Protocol config; retrieval.allow_fallback is authoritative by default",
    )
    parser.add_argument(
        "--allow-fallback", action=argparse.BooleanOptionalAction, default=None,
        help="Override retrieval.allow_fallback for diagnostic runs. The final "
             "closed_world_408 config keeps it false.",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        parser.error("--shard-idx must be in [0, --num-shards)")
    apply_qwen_thinking(args.model)
    if args.base_model_path:
        from msi.models.arch_profile import detect
        args.vl_rename = args.vl_rename or detect(args.base_model_path).vl_rename
    protocol_config = load_config(args.config)
    configured_fallback = (
        protocol_config.get("defaults", {}).get("fallback", "error") == "allow"
    )
    allow_fallback = configured_fallback if args.allow_fallback is None else args.allow_fallback

    # Validate
    if args.method not in NO_RETRIEVAL_METHODS and not args.retrieval_results:
        parser.error(f"--retrieval-results required for method '{args.method}'")
    if args.method in DEMOS_METHODS and not args.demos_file:
        parser.error(f"--demos-file required for method '{args.method}'")
    if args.lora_backend == "vllm":
        if not args.api_base:
            parser.error("--lora-backend vllm requires --api-base (student vLLM server URL)")
    elif not args.base_model_path:
        parser.error("--base-model-path is required for --lora-backend inprocess")

    datasets = [args.name] if args.name else ALL_DATASETS
    # vllm: ALL methods talk to the student server (model=<adapter> or <base>).
    # inprocess: non-LoRA loads base-only; LoRA loads base+adapters (per-dataset).
    is_lora = args.method in LORA_METHODS
    if args.lora_backend == "vllm":
        client = _make_pooled_client(args.api_base, args.workers)
    elif not is_lora:
        _bm, _bt = load_base_only(args.base_model_path)
        client = InProcessClient(_bm, _bt, {}, default_max_tokens=args.max_tokens)
    else:
        client = None  # built per-dataset inside the loop
    corpus = _load_corpus() if args.method != "naive" else {}
    lora_dir = Path(args.lora_dir) if args.lora_dir else RESULTS_DIR / "lora"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    assert_shared_output(output_dir)

    # Engine kwargs (temperature, max_tokens, thinking)
    engine_kwargs = {}
    if args.temperature != 0.7:
        engine_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        engine_kwargs["max_tokens"] = args.max_tokens
    if args.thinking:
        engine_kwargs["thinking"] = True

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset} | method={args.method}")
        print(f"{'='*60}")

        instances_path = INSTANCES_DIR / f"{dataset}.json"
        if not instances_path.exists():
            print(f"  SKIPPED: {instances_path} not found")
            continue

        instances = json.load(open(instances_path))

        if not instances:
            print("  No instances to run")
            continue

        # Standard methods have no adapter ownership constraint.
        if args.num_shards > 1 and not is_lora:
            instances = instances[args.shard_idx::args.num_shards]
            print(f"  Shard {args.shard_idx}/{args.num_shards}: {len(instances)} instances")

        output_path = output_dir / dataset / Path(args.model).name / f"{args.method}.jsonl"
        if args.num_shards > 1:
            output_path = Path(str(output_path) + f".shard{args.shard_idx}")
        # --- LoRA methods: route each instance to its adapter (or base) ---
        if is_lora:
            is_retrieved = args.method in RETRIEVED_LORA_METHODS
            if args.method in MISMATCHED_LORA_METHODS:
                # Same routing mechanics as the retrieved regime (groups and
                # adapters follow the supplied route file's top-1), but the
                # provider stays ORACLE: tools and any text remain the
                # instance's gold skill, so only the adapter changes.
                _ret_data = json.loads(Path(args.retrieval_results).read_text())
                retrieval_map = {
                    r["instance_id"]: r["retrieved"] for r in _ret_data["results"]
                }
                _matched = sum(1 for i in instances if i["instance_id"] in retrieval_map)
                if _matched != len(instances) and not allow_fallback:
                    raise SystemExit(
                        f"mismatch route covers {_matched}/{len(instances)} "
                        f"instances for {dataset}; closed-world inference requires exact coverage"
                    )
                groups, fallback = _group_by_retrieved_adapter(
                    instances, retrieval_map, lora_dir,
                )
                if fallback and not allow_fallback:
                    raise SystemExit(
                        f"{len(fallback)} {dataset} instances lack the "
                        "mismatched adapter; fallback is disabled by protocol"
                    )
                _oracle = get_provider("oracle", corpus_path=str(CORPUS_PATH))
                provider = _TextFallbackProvider(
                    _oracle, [row["instance_id"] for row in fallback]
                )
            elif is_retrieved:
                # Retrieved regime: route by the retriever's top-1 skill, not the
                # oracle. Build the per-instance retrieval map and require one
                # complete top-1 adapter route per instance.
                _ret_data = json.loads(Path(args.retrieval_results).read_text())
                retrieval_map = {
                    r["instance_id"]: r["retrieved"] for r in _ret_data["results"]
                }
                # A retrieval file from another dataset must not silently route
                # every instance to the base model.
                _matched = sum(1 for i in instances if i["instance_id"] in retrieval_map)
                if _matched != len(instances) and not allow_fallback:
                    raise SystemExit(
                        f"retrieval covers {_matched}/{len(instances)} "
                        f"instances for {dataset}; closed-world inference requires exact coverage"
                    )
                if _matched:
                    _ret_ds = _ret_data.get("metadata", {}).get("dataset")
                    if _ret_ds and _ret_ds != dataset:
                        print(f"  *** WARNING: retrieval file dataset='{_ret_ds}' "
                              f"!= current '{dataset}' ***")
                    print(f"  retrieval match: {_matched}/{len(instances)} instances")
                groups, fallback = _group_by_retrieved_adapter(
                    instances, retrieval_map, lora_dir,
                )
                if fallback and not allow_fallback:
                    raise SystemExit(
                        f"{len(fallback)} {dataset} instances lack the "
                        "required retrieved adapter; fallback is disabled by protocol"
                    )
                # Skill text comes from the retriever (top-1), not the oracle.
                _topk = get_provider(
                    "topk", source=args.retrieval_results,
                    k=1, corpus_path=str(CORPUS_PATH),
                )
                for sid, group in groups.items():
                    for instance in group:
                        supplied = _topk.provide(instance)
                        if [skill["skill_id"] for skill in supplied] != [sid]:
                            raise SystemExit("SR-Agents skill text and adapter route disagree")
                if args.method in NO_TEXT_LORA_METHODS:
                    provider = _TextFallbackProvider(
                        _topk, [row["instance_id"] for row in fallback]
                    )
                else:
                    provider = _topk
            else:
                groups, fallback = _group_by_adapter(
                    instances, lora_dir, corpus, base_model=args.model,
                )
                if fallback and not allow_fallback:
                    raise SystemExit(
                        f"{len(fallback)} {dataset} instances lack their "
                        "gold adapter; fallback is disabled by protocol"
                    )

                if args.method in NO_TEXT_LORA_METHODS:
                    _oracle = get_provider("oracle", corpus_path=str(CORPUS_PATH))
                    provider = _TextFallbackProvider(
                        _oracle, [row["instance_id"] for row in fallback]
                    )
                else:
                    provider = get_provider("oracle", corpus_path=str(CORPUS_PATH))

            if args.num_shards > 1:
                groups, fallback, shard_loads = _shard_adapter_groups(
                    groups, fallback, args.shard_idx, args.num_shards,
                )
                print(
                    f"  Adapter-affine shard {args.shard_idx}/{args.num_shards}: "
                    f"{shard_loads[args.shard_idx]} instances "
                    f"(all shard loads: {shard_loads})"
                )

            # Batch adapter groups into bounded vLLM waves.  Each wave still
            # shares one worker pool, while the registration count stays within
            # the server's configured LoRA capacity.
            flat = [inst for group in groups.values() for inst in group] + fallback

            if args.num_shards > 1:
                removed = _drop_unassigned_resume_rows(
                    output_path, {str(inst["instance_id"]) for inst in flat},
                )
                if removed:
                    print(f"  Removed {removed} rows from the previous shard assignment")

            if flat:
                covered = sum(len(v) for v in groups.values())
                print(f"  {len(groups)} adapters ({covered} LoRA, {len(fallback)} fallback) "
                      f"in one pool (workers={args.workers})")
                if args.lora_backend == "vllm":
                    if args.max_loaded_loras < 1:
                        raise SystemExit("--max-loaded-loras must be positive")
                    group_items = list(groups.items())
                    waves = [
                        group_items[index:index + args.max_loaded_loras]
                        for index in range(0, len(group_items), args.max_loaded_loras)
                    ]
                    if fallback:
                        waves.append([])
                    for wave_index, wave in enumerate(waves, start=1):
                        registered = {}
                        model_by_instance = {}
                        wave_instances = []
                        for sid, group in wave:
                            reg = register_lora(
                                args.api_base, sid,
                                str(lora_dir / sid),
                                vl_rename=args.vl_rename,
                            )
                            if not reg:
                                for loaded_sid in registered:
                                    unregister_lora(args.api_base, loaded_sid)
                                raise SystemExit(
                                    f"adapter registration failed: {sid}"
                                )
                            registered[sid] = reg
                            for inst in group:
                                model_by_instance[inst["instance_id"]] = sid
                                wave_instances.append(inst)
                        if not wave and fallback:
                            for inst in fallback:
                                model_by_instance[inst["instance_id"]] = args.model
                                wave_instances.append(inst)
                        print(
                            f"  vLLM wave {wave_index}/{len(waves)}: "
                            f"{len(registered)} adapters, {len(wave_instances)} instances",
                            flush=True,
                        )
                        engine = _PerInstanceModelEngine(
                            _make_engine(dataset, engine_kwargs), model_by_instance,
                        )
                        try:
                            run_many(
                                instances=wave_instances, provider=provider, engine=engine,
                                client=client, model=args.model,
                                output_path=output_path,
                                label=args.method,
                                workers=args.workers,
                                engine_kwargs={**engine_kwargs, "base_model": args.model},
                            )
                        finally:
                            import shutil as _shutil
                            for sid, reg in registered.items():
                                unregister_lora(args.api_base, sid)
                                if args.vl_rename:
                                    _shutil.rmtree(reg, ignore_errors=True)
                else:
                    client = _build_inprocess_client(
                        args.base_model_path, lora_dir, groups.keys(),
                        args.max_tokens, allow_base_model=bool(fallback),
                    )
                    model_by_instance = {
                        inst["instance_id"]: sid
                        for sid, group in groups.items() for inst in group
                    }
                    model_by_instance.update({
                        inst["instance_id"]: args.model for inst in fallback
                    })
                    engine = _PerInstanceModelEngine(
                        _make_engine(dataset, engine_kwargs), model_by_instance,
                    )
                    run_many(
                        instances=flat, provider=provider, engine=engine,
                        client=client, model=args.model,
                        output_path=output_path, label=args.method,
                        workers=args.workers,
                        engine_kwargs={**engine_kwargs, "base_model": args.model},
                    )
            else:
                print("  No instances to run")
            continue

        # --- Pre-warm ToolQA ---
        if dataset == "toolqa":
            print("  Pre-warming ToolQA environment...")
            _prewarm_toolqa()

        # --- Standard methods ---
        if args.method == "naive":
            provider = get_provider("none")
        elif args.method == "golden_skill":
            provider = get_provider("oracle", corpus_path=str(CORPUS_PATH))
        elif args.method in DEMOS_METHODS:
            # Experiment B: routed skill text + frozen 2-shot demos, NO adapter.
            if args.method == "golden_skill_2shot":
                base = get_provider("oracle", corpus_path=str(CORPUS_PATH))
            else:  # retrieved_skill_2shot: same frozen top-1 route as main table
                base = get_provider(
                    "topk", source=args.retrieval_results,
                    k=1, corpus_path=str(CORPUS_PATH),
                )
            provider = _DemoProvider(
                base, json.loads(Path(args.demos_file).read_text(encoding="utf-8")))
        elif args.retrieval_results:
            provider = get_provider(
                "topk", source=args.retrieval_results,
                k=args.top_k, corpus_path=str(CORPUS_PATH),
            )
        else:
            provider = get_provider("none")

        engine = _make_engine(dataset, engine_kwargs)

        run_many(
            instances=instances, provider=provider, engine=engine,
            client=client, model=args.model,
            output_path=output_path, label=args.method,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
