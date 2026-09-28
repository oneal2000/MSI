"""Public stage interface for the MSI paper release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from msi import REPO_ROOT
from msi.config import (
    ConfigError, apply_assignments, apply_scoped, expand, load_config,
    set_value, validate_stage, write_resolved,
)
from msi.runtime.gpus import GPUSelectionError, resolve as resolve_gpus


def _common(
    parser: argparse.ArgumentParser, default_config: str, *, output_dir: bool = True,
) -> None:
    parser.add_argument("-c", "--config", default=str(REPO_ROOT / default_config))
    parser.add_argument("--cluster", help="Untracked environment paths YAML")
    parser.add_argument("--run-id")
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--skill", "--skills", action="append", metavar="ID[,ID...]",
        help="Select one skill, a comma-separated list, or repeat this option",
    )
    parser.add_argument("--work-dir")
    parser.add_argument("--cache-dir")
    if output_dir:
        parser.add_argument("--output-dir")
    parser.add_argument("--set", dest="set_values", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--dataset-param", action="append", default=[],
                        metavar="DATASET.KEY=VALUE")
    parser.add_argument("--model-param", action="append", default=[],
                        metavar="MODEL.KEY=VALUE")
    parser.add_argument("--skill-param", action="append", default=[],
                        metavar="SKILL.KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")


def _skill_selection(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated public skill selections."""
    if values is None:
        return []
    selected = [
        skill.strip()
        for value in values
        for skill in value.split(",")
        if skill.strip()
    ]
    if not selected:
        raise ConfigError("--skill must contain at least one non-empty skill ID")
    return list(dict.fromkeys(selected))


def _add_generate(sub) -> None:
    p = sub.add_parser("generate", help="Generate and audit synthetic trajectories")
    _common(p, "configs/generate.yaml")
    p.add_argument(
        "--gpu", help="GPUs for a pipeline-managed teacher; omit with --api-base",
    )
    p.add_argument("--teacher-model")
    p.add_argument("--teacher-model-path")
    p.add_argument("--api-base")
    p.add_argument("--task-model")
    p.add_argument("--task-api-base")
    p.add_argument("--trajectory-model")
    p.add_argument("--trajectory-api-base")
    p.add_argument("--evaluator-model")
    p.add_argument("--evaluator-api-base")
    p.add_argument("--teacher-tokenizer")
    p.add_argument("--server-port", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--server-max-model-len", type=int)
    p.add_argument("--phase", choices=("all", "tasks", "trajectories"))
    p.add_argument(
        "--evaluator", action=argparse.BooleanOptionalAction, default=None,
        help="Enable the LLM trajectory judge; --no-evaluator uses format-only checks",
    )
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--task-temperature", type=float)
    p.add_argument("--delay", type=float)
    p.add_argument("--min-acceptable", type=int)
    p.add_argument("--task-oversample", type=float)
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--dedup-threshold", type=float)
    p.add_argument(
        "--retry-failed", action=argparse.BooleanOptionalAction, default=None,
        help="Retry one attempt for failures already recorded in generation progress",
    )


def _add_anchors(sub) -> None:
    p = sub.add_parser("anchors", help="Build synthetic-only retrieval anchors")
    actions = p.add_subparsers(dest="anchor_action", required=True)
    prepare = actions.add_parser(
        "prepare", help="Publish task-pool retrieval and per-skill task assignments",
    )
    _common(prepare, "configs/anchors.yaml", output_dir=False)
    prepare.add_argument(
        "--gpu", help="Comma-separated physical GPU indices or UUIDs (for example 0,1,2)"
    )
    prepare.add_argument("--source-file", help="Pre-aggregated synthetic anchor source JSON")
    prepare.add_argument("--retriever-model")
    prepare.add_argument("--retriever", choices=("bge_base", "bge_m3", "bm25"))
    prepare.add_argument("--reranker-model")
    prepare.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=None,
        help="Enable the configured reranker or use first-stage retrieval only",
    )
    prepare.add_argument("--retrieval-batch-size", type=int)
    prepare.add_argument("--retrieval-query-chunk-size", type=int)
    prepare.add_argument(
        "--retrieval-dtype", choices=("float32", "float16", "bfloat16")
    )
    prepare.add_argument("--reranker-batch-size", type=int)
    prepare.add_argument("--reranker-max-length", type=int)
    prepare.add_argument("--first-stage-top-k", type=int)
    prepare.add_argument("--top-k", type=int)
    prepare.add_argument("--near-count", type=int)
    prepare.add_argument("--random-count", type=int)
    prepare.add_argument("--oversample", type=float)

    answer = actions.add_parser(
        "answer", help="Answer a selected subset of published anchor tasks",
    )
    _common(answer, "configs/anchors.yaml")
    answer.add_argument(
        "--gpu", help="GPU for a pipeline-managed answer model server",
    )
    answer.add_argument("--base-model", dest="model", required=True)
    answer.add_argument("--served-model")
    answer.add_argument("--api-base")
    answer.add_argument("--base-answer-max-tokens", type=int)
    answer.add_argument("--workers", type=int)
    answer.add_argument("--server-port", type=int)
    answer.add_argument("--gpu-memory-utilization", type=float)
    answer.add_argument("--server-max-model-len", type=int)


def _add_train(sub) -> None:
    p = sub.add_parser("train", help="Train one model's per-skill LoRA adapters")
    _common(p, "configs/train/qwen3.5-2b.yaml")
    p.add_argument(
        "--gpu", help="Comma-separated physical GPU indices or UUIDs (for example 0,1,2)"
    )
    p.add_argument("--base-model")
    p.add_argument(
        "--anchors", dest="use_anchors",
        action=argparse.BooleanOptionalAction, default=None,
        help="Include synthetic hard/random negatives (paper default)",
    )
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--rank", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--max-length", type=int)
    p.add_argument("--num-train", type=int)
    p.add_argument("--num-val", type=int)
    p.add_argument("--val-offset", type=int)
    p.add_argument("--split-seed", type=int)
    p.add_argument("--val-freq", type=int)
    p.add_argument("--val-workers", type=int)
    p.add_argument("--val-concurrency", type=int)
    p.add_argument("--val-max-tokens", type=int)
    p.add_argument("--val-backend", choices=("vllm", "inprocess", "none"))
    p.add_argument(
        "--compute-val-loss", action=argparse.BooleanOptionalAction, default=None,
    )
    p.add_argument(
        "--base-baseline", action=argparse.BooleanOptionalAction, default=None,
        help="Run the optional synthetic validation base-model diagnostic",
    )
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--server-max-model-len", type=int)
    p.add_argument("--val-gpu", help="Dedicated physical validation GPU or UUID")
    p.add_argument("--val-port", type=int)
    p.add_argument("--no-validation", action="store_true",
                   help="Exploratory single-GPU mode; not equivalent to the paper protocol")
    p.add_argument("--val-async", dest="val_async", action="store_true", default=None)
    p.add_argument("--no-val-async", dest="val_async", action="store_false")
    p.add_argument("--regime", action="append", choices=("notext", "withtext"))
    p.add_argument("--api-base")


def _add_retriever(sub) -> None:
    p = sub.add_parser("retriever", help="Train and freeze a synthetic task-to-skill retriever")
    actions = p.add_subparsers(dest="retriever_action", required=True)
    for action in ("prepare", "train", "retrieve"):
        child = actions.add_parser(action)
        _common(child, "configs/retriever.yaml")
        child.add_argument("--gpu")


def _add_evaluate(sub) -> None:
    p = sub.add_parser("evaluate", help="Run SR-Agents evaluation with optional LoRA routing")
    _common(p, "configs/eval/qwen3.5-2b.yaml")
    p.add_argument(
        "--gpu", help="Comma-separated physical GPU indices or UUIDs (for example 0,1,2)"
    )
    p.add_argument("--base-model")
    p.add_argument("--api-base")
    p.add_argument("--workers", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--method", dest="methods", action="append")
    p.add_argument("--retrieval-file")
    p.add_argument("--adapter-dir")
    p.add_argument("--lora-backend", choices=("vllm", "inprocess"))
    p.add_argument("--fallback", choices=("error", "allow"))
    p.add_argument("--eval-workers", type=int)
    p.add_argument("--max-loaded-loras", type=int)
    p.add_argument("--server-port", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--server-max-model-len", type=int)


DIRECT: dict[str, dict[str, str]] = {
    "retriever": {},
    "generate": {
        "teacher_model": "teacher.model",
        "teacher_model_path": "teacher.path",
        "api_base": "teacher.api_base",
        "task_model": "teacher.task_model",
        "task_api_base": "teacher.task_api_base",
        "trajectory_model": "teacher.trajectory_model",
        "trajectory_api_base": "teacher.trajectory_api_base",
        "evaluator_model": "teacher.evaluator_model",
        "evaluator_api_base": "teacher.evaluator_api_base",
        "teacher_tokenizer": "teacher.tokenizer",
        "server_port": "runtime.port",
        "gpu_memory_utilization": "runtime.gpu_memory_utilization",
        "server_max_model_len": "runtime.max_model_len",
        "phase": "defaults.phase", "evaluator": "defaults.evaluator",
        "max_tokens": "defaults.max_tokens",
        "workers": "defaults.workers", "min_acceptable": "defaults.min_acceptable",
        "task_oversample": "defaults.task_oversample", "max_rounds": "defaults.max_rounds",
        "dedup_threshold": "defaults.dedup_threshold",
        "retry_failed": "defaults.retry_failed",
        "temperature": "defaults.temperature",
        "task_temperature": "defaults.task_temperature", "delay": "defaults.delay",
    },
    "anchors": {
        "source_file": "paths.source_file",
        "api_base": "runtime.api_base", "retriever_model": "retrieval.model",
        "retriever": "retrieval.first_stage",
        "reranker_model": "retrieval.reranker_model",
        "retrieval_batch_size": "retrieval.batch_size",
        "retrieval_query_chunk_size": "retrieval.query_chunk_size",
        "retrieval_dtype": "retrieval.dtype",
        "reranker_batch_size": "retrieval.reranker_batch_size",
        "reranker_max_length": "retrieval.reranker_max_length",
        "first_stage_top_k": "retrieval.first_stage_top_k",
        "top_k": "retrieval.top_k", "near_count": "defaults.near_count",
        "random_count": "defaults.random_count",
        "oversample": "defaults.oversample",
        "base_answer_max_tokens": "defaults.base_answer_max_tokens",
        "workers": "defaults.workers",
        "server_port": "runtime.port",
        "gpu_memory_utilization": "runtime.gpu_memory_utilization",
        "server_max_model_len": "runtime.max_model_len",
    },
    "train": {
        "base_model": "model.path", "batch_size": "defaults.batch_size",
        "use_anchors": "defaults.use_anchors",
        "gradient_accumulation_steps": "defaults.gradient_accumulation_steps",
        "rank": "defaults.rank", "epochs": "defaults.epochs",
        "learning_rate": "defaults.learning_rate", "max_length": "defaults.max_length",
        "num_train": "defaults.num_train", "num_val": "defaults.num_val",
        "split_seed": "defaults.split_seed", "val_backend": "defaults.val_backend",
        "val_offset": "defaults.fixed_val_offset",
        "val_freq": "defaults.val_freq", "val_workers": "defaults.val_workers",
        "val_concurrency": "defaults.val_concurrency",
        "val_max_tokens": "defaults.val_max_tokens",
        "compute_val_loss": "defaults.compute_val_loss",
        "val_async": "defaults.val_async", "api_base": "runtime.api_base",
        "val_port": "runtime.port",
        "gpu_memory_utilization": "runtime.gpu_memory_utilization",
        "server_max_model_len": "runtime.max_model_len",
    },
    "evaluate": {
        "base_model": "model.path", "api_base": "runtime.api_base",
        "workers": "defaults.workers", "max_tokens": "defaults.max_tokens",
        "temperature": "defaults.temperature", "retrieval_file": "paths.retrieval_file",
        "adapter_dir": "paths.adapter_dir", "lora_backend": "defaults.lora_backend",
        "fallback": "defaults.fallback",
        "eval_workers": "defaults.eval_workers",
        "max_loaded_loras": "defaults.max_loaded_loras",
        "server_port": "runtime.port",
        "gpu_memory_utilization": "runtime.gpu_memory_utilization",
        "server_max_model_len": "runtime.max_model_len",
    },
}


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.command
    config = load_config(args.config)
    validate_stage(config, stage)
    cluster: dict[str, Any] = {}
    if args.cluster:
        cluster = load_config(args.cluster)
        if cluster.get("schema") != "msi.cluster":
            raise ConfigError("cluster config schema must be 'msi.cluster'")
        config["_shared_umask"] = cluster.get("shared_umask")
        workspace = cluster.get("workspace")
        if workspace:
            set_value(config, "paths.work_dir", workspace)
            set_value(config, "paths.cache_dir", f"{workspace}/cache")
        if cluster.get("shared_root"):
            set_value(config, "paths.shared_root", cluster["shared_root"])
        for name, value in cluster.get("paths", {}).items():
            set_value(config, f"paths.{name}", value)
        config["_python"] = cluster.get("python")
        model_name = config.get("model", {}).get("name")
        if model_name in cluster.get("models", {}):
            set_value(config, "model.path", cluster["models"][model_name])
        if stage == "anchors":
            for name, model_path in cluster.get("models", {}).items():
                if name in config.get("models", {}):
                    # Model identifiers contain dots (for example Qwen3.5-2B),
                    # so a dotted-path setter would split the identifier.
                    config["models"][name]["path"] = model_path
        if stage == "evaluate":
            retrievers = cluster.get("retrievers", {})
            for profile, values in config.get("retrieval_profiles", {}).items():
                if not isinstance(values, dict):
                    continue
                if profile in {"bge_base", "bge_base_rerank"} and retrievers.get("bge_base"):
                    if values.get("model"):
                        set_value(config, f"retrieval_profiles.{profile}.model",
                                  retrievers["bge_base"])
                if profile in {"bge_m3", "bge_m3_rerank"} and retrievers.get("bge_m3"):
                    if values.get("model"):
                        set_value(config, f"retrieval_profiles.{profile}.model",
                                  retrievers["bge_m3"])
                if values.get("reranker_model") and retrievers.get("bge_reranker"):
                    set_value(config, f"retrieval_profiles.{profile}.reranker_model",
                              retrievers["bge_reranker"])
    if stage == "retriever" and cluster.get("retrievers", {}).get("bge_m3"):
        set_value(config, "model.path", cluster["retrievers"]["bge_m3"])
    apply_assignments(config, args.set_values)
    apply_scoped(config, "datasets", args.dataset_param)
    apply_scoped(config, "models", args.model_param)
    apply_scoped(config, "skills", args.skill_param)
    direct_defaults: dict[str, Any] = {}
    for attribute, key in DIRECT[stage].items():
        value = getattr(args, attribute, None)
        if value is not None:
            if key.startswith("defaults."):
                direct_defaults[key.removeprefix("defaults.")] = value
            else:
                set_value(config, key, value)
    if stage == "generate" and getattr(args, "teacher_model_path", None) is None:
        teacher_name = config.get("teacher", {}).get("model")
        if teacher_name in cluster.get("models", {}):
            set_value(config, "teacher.path", cluster["models"][teacher_name])
    if stage == "train" and getattr(args, "no_validation", False):
        direct_defaults["val_backend"] = "none"
    if stage == "train" and getattr(args, "base_baseline", None) is not None:
        direct_defaults["skip_base_baseline"] = not args.base_baseline
    config["_direct_defaults"] = direct_defaults
    if stage == "anchors":
        # The protocol label is derived from the implementation actually run;
        # it must never remain stale after a CLI override.
        rerank = getattr(args, "rerank", None)
        if rerank is False:
            set_value(config, "retrieval.reranker_model", None)
        retriever = str(config.get("retrieval", {}).get("first_stage", "bge_base"))
        retrievers = cluster.get("retrievers", {})
        if getattr(args, "retriever_model", None) is None and retrievers.get(retriever):
            set_value(config, "retrieval.model", retrievers[retriever])
        if (rerank is not False and getattr(args, "reranker_model", None) is None
                and retrievers.get("bge_reranker")):
            set_value(config, "retrieval.reranker_model", retrievers["bge_reranker"])
        has_reranker = bool(config.get("retrieval", {}).get("reranker_model"))
        set_value(config, "retrieval.name", retriever + ("_rerank" if has_reranker else ""))
    if args.run_id:
        config["run_id"] = args.run_id
    for attribute, key in (("work_dir", "work_dir"), ("cache_dir", "cache_dir")):
        value = getattr(args, attribute)
        if value:
            set_value(config, f"paths.{key}", value)
    output_value = getattr(args, "output_dir", None)
    if output_value:
        output_key = {
            "generate": "synthetic_dir",
            "anchors": "anchor_dir",
            "train": "adapter_dir",
            "evaluate": "output_dir",
            "retriever": "output_dir",
        }[stage]
        set_value(config, f"paths.{output_key}", output_value)
    run_id = str(config.get("run_id", "paper"))
    config = expand(config, {"run_id": run_id, "repo": str(REPO_ROOT)})
    requested_gpus = getattr(args, "gpu", None)
    selected_gpus = (
        [] if (stage == "generate" or (stage == "retriever" and args.retriever_action == "prepare")) and requested_gpus is None else
        resolve_gpus(requested_gpus, allow_unresolved=args.dry_run)
    )
    val_gpus = resolve_gpus(args.val_gpu, allow_unresolved=args.dry_run) \
        if getattr(args, "val_gpu", None) else []
    if len(val_gpus) > 1:
        raise ConfigError("--val-gpu accepts exactly one device")
    selection = {
        "datasets": args.dataset or [], "skills": _skill_selection(args.skill),
        "model": getattr(args, "model", None),
        "anchor_action": getattr(args, "anchor_action", None),
        "retriever_action": getattr(args, "retriever_action", None),
        "regimes": getattr(args, "regime", None),
        "served_model": getattr(args, "served_model", None),
        "methods": getattr(args, "methods", None),
        "gpus": selected_gpus,
        "val_gpu": val_gpus[0] if val_gpus else None,
    }
    config["_selection"] = selection
    config["invocation"] = {
        **selection,
        "direct_defaults": direct_defaults,
    }
    return config


def _apply_shared_umask(config: dict[str, Any]) -> None:
    """Apply the cluster umask before creating any shared run artifact.

    A shared filesystem may map the SSH account on each host to a different
    numeric UID/GID.  In that setup a conventional 0022 umask makes lock files
    created by the first host unwritable by every other host.  The cluster
    administrator can opt into a cooperative umask (normally 0002 with a
    shared group, or 0000 inside an access-controlled private workspace).
    """
    value = config.get("_shared_umask")
    if value is None:
        return
    try:
        mode = value if isinstance(value, int) else int(str(value), 8)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            "cluster shared_umask must be an octal value such as '0002' or '0000'"
        ) from error
    if isinstance(mode, bool) or not 0 <= mode <= 0o777:
        raise ConfigError(
            "cluster shared_umask must be between '0000' and '0777'"
        )
    os.umask(mode)


def _record_resolved(config: dict[str, Any], stage: str, dry_run: bool) -> None:
    work = Path(config["paths"].get("work_dir", REPO_ROOT / "work"))
    if not work.is_absolute():
        work = REPO_ROOT / work
    run_id = str(config.get("run_id", "paper"))
    selection = config.get("_selection", {})
    label = selection.get("model") or config.get("model", {}).get("name") or "all"
    if stage == "anchors":
        label = f"{selection.get('anchor_action')}-{label}"
    if stage == "retriever":
        label = f"{selection.get('retriever_action')}-{label}"
    destination = work / run_id / "resolved" / f"{stage}-{label}.yaml"
    config["_resolved_config_file"] = str(destination)
    print(f"Resolved config: {destination}")
    if not dry_run and stage != "retriever":
        write_resolved(config, destination)


def _configure_environment(config: dict[str, Any], cluster_path: str | None) -> None:
    # The checked-in launcher is intentionally zero-install. Isolated stage
    # processes use the cluster's admitted Python, so explicitly carry both
    # source roots instead of assuming an editable install on every host.
    import_roots = [
        str((REPO_ROOT / "src").resolve()),
        str((REPO_ROOT / "SR-Agents" / "src").resolve()),
    ]
    inherited_pythonpath = [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        [*import_roots, *inherited_pythonpath]
    ))
    # Output flags never grant themselves trust. Only the checkout and the
    # explicitly declared shared root may contain durable artifacts.
    roots = {str(REPO_ROOT.resolve())}
    shared_root = config.get("paths", {}).get("shared_root")
    if shared_root:
        root_path = Path(str(shared_root)).expanduser()
        if not root_path.is_absolute():
            root_path = REPO_ROOT / root_path
        roots.add(str(root_path.resolve()))
    os.environ["MSI_SHARED_ROOTS"] = os.pathsep.join(sorted(roots))
    path_env = {
        "corpus": "MSI_SKILL_CORPUS",
        "instances_dir": "MSI_INSTANCES_DIR",
        "output_dir": "MSI_RESULTS_DIR",
        "external_dir": "MSI_EXTERNAL_DIR",
        "toolqa_embedding_model": "SRAGENTS_TOOLQA_EMBED_MODEL",
    }
    for key, variable in path_env.items():
        value = config.get("paths", {}).get(key)
        if value:
            resolved = Path(str(value)).expanduser()
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            os.environ[variable] = str(resolved.resolve())
    if os.environ.get("MSI_EXTERNAL_DIR"):
        os.environ["SRAGENTS_EXTERNAL_DIR"] = os.environ[
            "MSI_EXTERNAL_DIR"
        ]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    selected_gpus = config.get("_selection", {}).get("gpus")
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    cache = config.get("paths", {}).get("cache_dir")
    if cache:
        cache_path = Path(str(cache)).expanduser()
        if not cache_path.is_absolute():
            cache_path = REPO_ROOT / cache_path
        cache_path = cache_path.resolve()
        os.environ.setdefault("HF_HOME", str(cache_path / "huggingface"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_path))
        # ToolQA text corpora are identical across GPU shards.  Give SR-Agents
        # one explicit cross-process cache so only one shard embeds them.
        os.environ.setdefault(
            "SRAGENTS_TOOLQA_EMBED_CACHE",
            str(cache_path / "toolqa-embeddings"),
        )
        os.environ.setdefault("TORCH_HOME", str(cache_path / "torch"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache_path / "triton"))
    if cluster_path:
        os.environ["MSI_CLUSTER_CONFIG"] = str(Path(cluster_path).resolve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="msi",
        description="Synthetic skill data, anchors, LoRA training and evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate(sub); _add_anchors(sub); _add_train(sub); _add_evaluate(sub); _add_retriever(sub)
    args = parser.parse_args(argv)
    try:
        config = _prepare_config(args)
        _apply_shared_umask(config)
        _record_resolved(config, args.command, args.dry_run)
        _configure_environment(config, args.cluster)
        package = {
            "generate": "generation", "train": "training", "evaluate": "evaluation"
        }.get(
            args.command, args.command
        )
        module = __import__(f"msi.{package}.pipeline", fromlist=["run"])
        code = module.run(config, dry_run=args.dry_run)
    except (ConfigError, GPUSelectionError) as error:
        parser.error(str(error))
    raise SystemExit(int(code or 0))


if __name__ == "__main__":
    main()
