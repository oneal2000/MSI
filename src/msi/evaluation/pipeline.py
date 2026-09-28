"""SR-Agents-compatible evaluation with MSI adapter routing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from hashlib import sha256

from msi import REPO_ROOT
from msi.retriever.ownership import claim

from msi.runtime.gpus import child_environment
from msi.runtime.invoke import StageRunner
from pathlib import Path

from msi.runtime.pipeline import datasets, path, repeated, settings
from msi.runtime.validation_server import (
    managed_server_pool, managed_validation_server,
)


LORA_METHODS = {
    "golden_lora", "golden_lora_notext_text", "golden_lora_text",
    "retrieved_lora", "retrieved_lora_notext_text", "retrieved_lora_text",
    # Ablation: same-domain non-gold adapter routing with gold tools.
    "mismatched_lora",
}


def _inference_work_files(work_file: Path, process_count: int) -> list[Path]:
    if process_count < 1:
        raise ValueError("inference requires at least one process")
    if process_count == 1:
        return [work_file]
    return [
        Path(str(work_file) + f".shard{index}")
        for index in range(process_count)
    ]


def _retrieval_profile(config: dict, method: str) -> str | None:
    explicit = config.get("method_retrieval", {}).get(method)
    if explicit:
        return str(explicit)
    if method.endswith("_top1"):
        return method[:-5]
    if method.startswith("retrieved_"):
        return "bge_base_rerank"
    return None


def _profile_path(mapping: dict, profile: str, dataset: str):
    value = mapping.get(profile, {})
    if isinstance(value, dict):
        return value.get(dataset)
    return None


def _ensure_retrieval(
    runner: StageRunner, config: dict, profile: str, dataset: str,
    instances_dir, corpus, dry_run: bool, prepared: dict,
):
    key = (profile, dataset)
    if key in prepared:
        return prepared[key]
    paths = config.get("paths", {}).get("retrieval", {})
    output_value = _profile_path(paths, profile, dataset)
    if not output_value:
        raise ValueError(f"missing paths.retrieval.{profile}.{dataset}")
    output = Path(output_value)
    if not output.is_absolute():
        output = path({"paths": {"value": output_value}}, "value")
    profile_config = config.get("retrieval_profiles", {}).get(profile)
    if not isinstance(profile_config, dict):
        raise ValueError(f"missing retrieval_profiles.{profile}")
    source_profile = profile_config.get("source_profile")
    if profile_config.get("frozen"):
        if not output.is_file():
            raise FileNotFoundError(f"missing bundled frozen routes: {output}")
    elif profile == "bge_m3_ft_synthetic":
        selected = profile_config["checkpoint_record"]
        if dry_run:
            print(f"Frozen FT routes: {selected} -> {output}")
        else:
            from msi.retriever.retrieve import retrieve
            retrieve(selected_file=selected, corpus_file=corpus,
                     instances_file=instances_dir / f"{dataset}.json", output=output,
                     batch_size=profile_config.get("batch_size", 16))
    elif source_profile:
        source = _ensure_retrieval(
            runner, config, str(source_profile), dataset,
            instances_dir, corpus, dry_run, prepared,
        )
        if dry_run or not output.is_file():
            runner.run("msi.anchors.reranker", [
                "--input", source, "--output", output,
                "--instances", instances_dir / f"{dataset}.json",
                "--corpus", corpus,
                "--model", profile_config["reranker_model"],
                "--name", profile,
                "--top-k", profile_config.get("top_k", 10),
                "--max-length", profile_config.get("max_length", 512),
                "--batch-size", profile_config.get("batch_size", 32),
            ], isolated=True)
    elif dry_run or not output.is_file():
        retriever = str(profile_config["retriever"])
        arguments: list[object] = [
            "retrieve", "--retriever", retriever,
            "--corpus", corpus,
            "--instances", instances_dir / f"{dataset}.json",
            "--output", output,
            "--top-k", profile_config.get("top_k", 10),
        ]
        if profile_config.get("model"):
            arguments += ["--retriever-arg", f"model_path={profile_config['model']}"]
        if profile_config.get("batch_size"):
            arguments += [
                "--retriever-arg", f"batch_size={profile_config['batch_size']}"
            ]
        runner.run("sragents.cli.main", arguments, isolated=True)
    fallback = settings(config, dataset=dataset).get("fallback", "error")
    runner.run("msi.evaluation.retrieval", [
        "--retrieval", output,
        "--instances", instances_dir / f"{dataset}.json",
        "--corpus", corpus, "--dataset", dataset,
        "--required-profile", profile_config.get("required_profile", profile),
        *(["--frozen"] if profile_config.get("frozen") else []),
        *(["--checkpoint-record", profile_config["checkpoint_record"]]
          if profile == "bge_m3_ft_synthetic" else []),
        *(["--allow-empty"] if fallback == "allow" else []),
    ], isolated=True)
    prepared[key] = output
    return output


def run(config: dict, *, dry_run: bool = False) -> int:
    runner = StageRunner(dry_run=dry_run, python=config.get("_python"))
    model = config["model"]
    selected = config.get("_selection", {})
    methods = selected.get("methods") or config.get("methods", [])
    selected_datasets = datasets(config)
    if not methods:
        raise ValueError("evaluation method selection is empty")
    output = path(config, "output_dir", "results/inference")
    eval_output = path(config, "eval_dir", "results/eval")
    adapters = path(config, "adapter_dir", "work/adapters")
    instances_dir = path(config, "instances_dir", "SR-Agents/data/bench/instances")
    corpus = path(config, "corpus", "data/protocol/skills.json")
    work = path(config, "work_dir", "work") / str(config.get("run_id", "paper"))
    inference_work = work / "inference"
    runtime = config.get("runtime", {})
    external_api_base = runtime.get("api_base")
    prepared: dict = {}

    # ToolQA shards share the same two corpus embedding matrices.  Build or
    # validate them once before the GPU queues fan out, so a first run on an
    # NFS-backed shared directory cannot make four processes race on the cache.
    if "toolqa" in selected_datasets:
        runner.run("msi.evaluation.toolqa_prewarm", [], isolated=True)

    retrieval_for: dict[tuple[str, str], Path | str] = {}
    for dataset in selected_datasets:
        values = settings(config, dataset=dataset)
        for method in methods:
            profile = _retrieval_profile(config, method)
            if not profile or (dataset, profile) in retrieval_for:
                continue
            explicit = config.get("paths", {}).get("retrieval_file")
            if explicit:
                retrieval = explicit
                runner.run("msi.evaluation.retrieval", [
                    "--retrieval", retrieval,
                    "--instances", instances_dir / f"{dataset}.json",
                    "--corpus", corpus, "--dataset", dataset,
                    "--required-profile", config["retrieval_profiles"][profile].get("required_profile", profile),
                    *(["--frozen"] if config["retrieval_profiles"][profile].get("frozen") else []),
                    *(["--checkpoint-record", config["retrieval_profiles"][profile]["checkpoint_record"]]
                      if profile == "bge_m3_ft_synthetic" else []),
                    *(["--allow-empty"] if values.get("fallback", "error") == "allow" else []),
                ], isolated=True)
            else:
                retrieval = _ensure_retrieval(
                    runner, config, profile, dataset,
                    instances_dir, corpus, dry_run, prepared,
                )
            retrieval_for[(dataset, profile)] = retrieval

    def run_job(
        dataset: str, method: str, api_base: str | None, gpu: str | None,
        shard_index: int, shard_count: int,
    ) -> None:
        values = settings(config, dataset=dataset)
        arguments: list[object] = [
            "--name", dataset, "--method", method,
            "--model", model["name"], "--base-model-path", model["path"],
            "--lora-backend", values.get("lora_backend", "vllm"),
            "--workers", values.get("workers", 128),
            "--max-tokens", values["max_tokens"],
            "--temperature", values.get("temperature", 0),
            "--max-loaded-loras", values.get("max_loaded_loras", 16),
            "--lora-dir", adapters / model["name"],
            "--output-dir", inference_work,
            "--config", config.get("_resolved_config_file", config["_config_file"]),
            "--shard-idx", shard_index, "--num-shards", shard_count,
        ]
        if api_base:
            arguments += ["--api-base", api_base]
        # Experiment B (*_skill_2shot): forward the frozen demo payload.
        demos_file = config.get("demos_file")
        if demos_file:
            arguments += ["--demos-file", str(demos_file)]
        profile = _retrieval_profile(config, method)
        if profile:
            retrieval = retrieval_for[(dataset, profile)]
            arguments += [
                "--retrieval-results", retrieval,
            ]
        if method in LORA_METHODS:
            regime = config.get("method_adapters", {}).get(
                method,
                "notext" if method.endswith("_notext_text")
                else "withtext" if method.endswith("_text") else "notext",
            )
            arguments[arguments.index("--lora-dir") + 1] = adapters / model["name"] / regime
        arguments.append(
            "--no-allow-fallback"
            if values.get("fallback", "error") == "error" else "--allow-fallback"
        )
        environment = None
        if gpu:
            cache = path(config, "cache_dir", f"work/{config.get('run_id', 'paper')}/cache")
            gpu_cache = cache / f"gpu-{gpu}"
            gpu_temp = work / "tmp" / f"gpu-{gpu}"
            if not dry_run:
                gpu_cache.mkdir(parents=True, exist_ok=True)
                gpu_temp.mkdir(parents=True, exist_ok=True)
            environment = child_environment(gpu, gpu_cache, gpu_temp)
        runner.run(
            "msi.evaluation.inference", arguments, isolated=True,
            environment=environment,
        )

    jobs = [(dataset, method) for dataset in selected_datasets for method in methods]
    backends = {
        settings(config, dataset=dataset).get("lora_backend", "vllm")
        for dataset in selected_datasets
    }
    if len(backends) != 1:
        raise ValueError("one evaluate invocation requires one lora_backend")
    backend = next(iter(backends))
    selected_gpus = list(selected.get("gpus") or [])

    if external_api_base:
        endpoints = [external_api_base]
        queue_gpus: list[str | None] = [None]
        server_context = nullcontext(endpoints)
    else:
        if not selected_gpus:
            raise ValueError(
                "evaluation requires --gpu (or visible GPUs) when --api-base is omitted"
            )
        queue_count = len(selected_gpus)
        queue_gpus = selected_gpus[:queue_count]
        if backend == "vllm":
            shared_root = path(config, "shared_root", ".")
            cache = path(config, "cache_dir", f"work/{config.get('run_id', 'paper')}/cache")
            base_port = int(runtime.get("port", 8100))
            managers = [
                managed_validation_server(
                    python=config.get("_python") or runner.python,
                    model_path=model["path"], served_name=model["name"],
                    gpu=gpu, port=base_port + index, shared_root=shared_root,
                    cache_dir=cache / f"gpu-{gpu}",
                    temp_dir=work / "tmp" / f"gpu-{gpu}",
                    log_path=work / "logs" / model["name"] /
                    f"evaluation-server-{index}.log",
                    gpu_memory_utilization=float(
                        runtime.get("gpu_memory_utilization", 0.92)
                    ),
                    max_model_len=(
                        int(runtime["max_model_len"])
                        if runtime.get("max_model_len") is not None else None
                    ),
                    dry_run=dry_run,
                )
                for index, gpu in enumerate(queue_gpus)
            ]
            server_context = managed_server_pool(managers)
        else:
            server_context = nullcontext([None] * queue_count)

    def finalize_job(dataset: str, method: str, shard_count: int) -> None:
        values = settings(config, dataset=dataset)
        model_label = Path(model["name"]).name
        inference_file = output / dataset / model_label / f"{method}.jsonl"
        work_file = inference_work / dataset / model_label / f"{method}.jsonl"
        shard_files = _inference_work_files(work_file, shard_count)
        runner.run("msi.evaluation.output", [
            "--instances", instances_dir / f"{dataset}.json",
            "--dataset", dataset, "--method", method,
            *repeated("--input", shard_files),
            "--output", inference_file,
        ])
        runner.run("sragents.cli.main", [
            "evaluate", "--input", inference_file,
            "--instances", instances_dir / f"{dataset}.json",
            "--output", eval_output / dataset / model_label / f"{method}.json",
            "--workers", values.get("eval_workers", 32),
            "--force",
        ], isolated=True)

    # Claims follow final output identity, not run ID: two separate launchers
    # must not overwrite the same published condition. Reuse atomic mkdir claims
    # because flock is not propagated across hosts on the shared filesystem.
    with ExitStack() as writers:
        if not dry_run:
            claims = REPO_ROOT / "work" / ".evaluation_claims"
            claims.mkdir(parents=True, exist_ok=True)
            targets = {str((root / dataset / Path(model["name"]).name /
                            f"{method}{suffix}").resolve())
                       for dataset, method in jobs
                       for root, suffix in ((output, ".jsonl"), (eval_output, ".json"))}
            for target in sorted(targets):
                key = sha256(target.encode()).hexdigest()
                writers.enter_context(claim(claims / f"{key}.claim"))
        with server_context as endpoints:
            shard_count = len(endpoints)
            with ThreadPoolExecutor(max_workers=shard_count) as pool:
                for dataset, method in jobs:
                    futures = [
                        pool.submit(
                            run_job, dataset, method, endpoints[index],
                            queue_gpus[index], index, shard_count,
                        )
                        for index in range(shard_count)
                    ]
                    for future in futures:
                        future.result()
                    # Finalize each complete condition immediately. A failure in a
                    # later dataset/setting cannot strand earlier valid work shards
                    # without their merged output and evaluation.
                    finalize_job(dataset, method, shard_count)
    return 0
