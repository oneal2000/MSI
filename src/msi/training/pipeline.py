"""Configuration-driven, multi-GPU per-skill LoRA training."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, Queue

from msi.runtime.gpus import child_environment
from msi.runtime.invoke import StageRunner
from msi.runtime.pipeline import (
    path, repeated, selected_skill_ids, settings,
)
from msi.runtime.validation_server import managed_validation_server
from msi.training.provenance import synthetic_stage_path
from msi.training.provenance import file_sha256


def _already_complete(directory, job_id: str, anchor_file: Path | None = None) -> bool:
    marker = directory / "TRAINING_DONE"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    adapter = directory / "adapter_model.safetensors"
    valid = (
        adapter.is_file()
        and payload.get("training_job_id") == job_id
    )
    if not valid:
        raise ValueError(f"stale or inconsistent completion marker: {marker}")
    if anchor_file is not None:
        current_hash = file_sha256(anchor_file)
        recorded_hash = payload.get("anchor_sha256")
        if recorded_hash is not None:
            if recorded_hash != current_hash:
                print(f"RETRAIN adapter with changed anchor: {directory}", flush=True)
                return False
        else:
            # Legacy completion markers predate content identities.  Their
            # metadata still proves whether the current anchor was published
            # after training and therefore cannot be the input that was used.
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                raise ValueError(f"legacy completion marker has no metadata: {marker}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded_path = metadata.get("training_inputs", {}).get("anchor")
            if recorded_path != str(anchor_file.resolve()):
                raise ValueError(f"legacy adapter references a different anchor: {marker}")
            if metadata_path.stat().st_mtime_ns < anchor_file.stat().st_mtime_ns:
                print(f"RETRAIN legacy adapter with newer anchor: {directory}", flush=True)
                return False
    print(f"SKIP complete adapter: {directory}", flush=True)
    return True


def _training_job_id(
    run_id: str, model: dict, skill_id: str, regime: str,
) -> str:
    """Return a readable identity for the run-local adapter directory."""
    return f"{run_id}/{model['name']}/{regime}/{skill_id}"


def _gpu_roles(config: dict, validation_enabled: bool) -> tuple[list[str], str | None]:
    selected = list(config.get("_selection", {}).get("gpus") or [])
    explicit_val = config.get("_selection", {}).get("val_gpu")
    if not selected:
        raise ValueError("no GPUs discovered; pass --gpu with physical indices or UUIDs")
    if not validation_enabled:
        return selected, None
    val_gpu = explicit_val or selected[-1]
    train_gpus = [gpu for gpu in selected if gpu != val_gpu]
    all_gpus = set(selected) | {val_gpu}
    if len(all_gpus) < 2 or not train_gpus:
        raise ValueError(
            "validation requires a dedicated GPU and at least one different training GPU; "
            "pass --gpu 0,1 (last GPU is reserved), or use --gpu 0 --no-validation "
            "for non-paper exploratory runs"
        )
    return train_gpus, val_gpu


def _dispatch_training_jobs(job_specs, train_gpus, train_one) -> None:
    """Drain training jobs without letting one failed job retire a GPU worker."""
    pending: Queue[dict] = Queue()
    for spec in job_specs:
        pending.put(spec)
    failures: Queue[tuple[dict, str, BaseException]] = Queue()

    def gpu_queue(index: int) -> None:
        gpu = train_gpus[index]
        while True:
            try:
                spec = pending.get_nowait()
            except Empty:
                return
            try:
                train_one(spec, gpu)
            except (Exception, SystemExit) as error:
                failures.put((spec, gpu, error))
                print(
                    f"TRAINING JOB FAILED: {spec['job_id']} on GPU {gpu}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=len(train_gpus)) as pool:
        futures = [pool.submit(gpu_queue, index) for index in range(len(train_gpus))]
        for future in futures:
            future.result()

    failed = []
    while not failures.empty():
        spec, gpu, error = failures.get_nowait()
        failed.append(
            f"{spec['job_id']} on GPU {gpu}: {type(error).__name__}: {error}"
        )
    if failed:
        raise RuntimeError(
            f"{len(failed)} training job(s) failed; incomplete jobs remain "
            "restartable:\n" + "\n".join(failed)
        )


def run(config: dict, *, dry_run: bool = False) -> int:
    runner = StageRunner(dry_run=dry_run, python=config.get("_python"))
    model = config["model"]
    run_id = str(config.get("run_id", "paper"))
    regimes = config.get("_selection", {}).get("regimes") \
        or config.get("regimes", ["notext", "withtext"])
    trajectories = path(config, "synthetic_dir", "data/synthetic")
    anchors = path(config, "anchor_dir", "data/anchors") / model["name"]
    adapters = path(config, "adapter_dir", f"work/{run_id}/adapters") / model["name"]
    work = path(config, "work_dir", "work") / run_id
    cache = path(config, "cache_dir", f"work/{run_id}/cache")
    shared_root = path(config, "shared_root", ".")
    skill_ids = selected_skill_ids(config)
    runtime = config.get("runtime", {})
    external_api_base = runtime.get("api_base")

    tests = config.get("paths", {}).get("test_instances", {})
    if len(tests) != 4:
        raise ValueError("training validation requires all four benchmark files")

    # Validate the exact inputs selected by this invocation. The check reads
    # current contents directly and writes no audit sidecar.
    audit_trajectories = []
    audit_anchors = []
    for skill_id in skill_ids:
        dataset = skill_id.rsplit("_", 1)[0]
        values = settings(config, dataset=dataset, skill=skill_id)
        audit_trajectories.append(
            path({"paths": {"input": values["trajectory_file"]}}, "input")
            if values.get("trajectory_file") else synthetic_stage_path(trajectories, skill_id, "trajectories")
        )
        if values.get("use_anchors", True):
            audit_anchors.append(
                path({"paths": {"input": values["anchor_file"]}}, "input")
                if values.get("anchor_file") else anchors / dataset / f"{skill_id}.json"
            )
    runner.run("msi.audit.leakage", [
        *repeated("--trajectory", audit_trajectories),
        *repeated("--anchor", audit_anchors),
        *repeated("--test-instances", tests.values()),
        *repeated("--skill", skill_ids),
        "--required-anchor-retriever", "bge_base_rerank",
        "--skill-corpus", path(config, "corpus"),
    ])

    # Resolve every job before paying the validation-server startup cost. This
    # makes a completed retry a metadata-only resume.
    job_specs: list[dict] = []
    for skill_id in skill_ids:
        dataset = skill_id.rsplit("_", 1)[0]
        for regime in regimes:
            values = settings(config, dataset=dataset, skill=skill_id)
            destination = adapters / regime / skill_id
            trajectory_file = (
                path({"paths": {"input": values["trajectory_file"]}}, "input")
                if values.get("trajectory_file") else synthetic_stage_path(
                    trajectories, skill_id, "trajectories"
                )
            )
            anchor_file = (
                (path({"paths": {"input": values["anchor_file"]}}, "input")
                 if values.get("anchor_file") else anchors / dataset / f"{skill_id}.json")
                if values.get("use_anchors", True) else None
            )
            job_id = (
                "DRY-RUN"
                if dry_run else _training_job_id(run_id, model, skill_id, regime)
            )
            if not dry_run and _already_complete(destination, job_id, anchor_file):
                continue
            job_specs.append({
                "skill_id": skill_id,
                "dataset": dataset,
                "regime": regime,
                "values": values,
                "destination": destination,
                "trajectory_file": trajectory_file,
                "anchor_file": anchor_file,
                "job_id": job_id,
            })

    pending_validation = any(
        spec["values"].get("val_backend", "vllm") != "none"
        for spec in job_specs
    )
    train_gpus, val_gpu = _gpu_roles(config, pending_validation)

    if pending_validation and not external_api_base:
        server = managed_validation_server(
            python=config.get("_python") or runner.python,
            model_path=model["path"], served_name=model["name"], gpu=str(val_gpu),
            port=int(runtime.get("port", 8003)), shared_root=shared_root,
            cache_dir=cache / f"gpu-{val_gpu}",
            temp_dir=work / "tmp" / f"gpu-{val_gpu}",
            log_path=work / "logs" / model["name"] / "validation-server.log",
            gpu_memory_utilization=float(runtime.get("gpu_memory_utilization", 0.92)),
            max_model_len=(
                int(runtime["max_model_len"])
                if runtime.get("max_model_len") is not None else None
            ),
            dry_run=dry_run,
        )
    else:
        server = nullcontext(external_api_base)

    with server as api_base:
        def train_one(spec: dict, gpu: str) -> None:
            skill_id = spec["skill_id"]
            values = spec["values"]
            arguments: list[object] = [
                "--base-model", model["path"], "--base-model-name", model["name"],
                "--skill-id", skill_id,
                "--trajectory-file", spec["trajectory_file"],
                "--output-dir", adapters / spec["regime"],
                "--training-job-id", spec["job_id"],
                "--rank", values["rank"], "--epochs", values["epochs"],
                "--batch-size", values["batch_size"],
                "--gradient-accumulation-steps", values["gradient_accumulation_steps"],
                "--learning-rate", values["learning_rate"],
                "--max-length", values["max_length"],
                "--num-train", values["num_train"], "--num-val", values["num_val"],
                "--split-seed", values.get("split_seed", 42),
                "--val-backend", values.get("val_backend", "vllm"),
                "--val-max-tokens", values["val_max_tokens"],
                # Regime-scoped lora_name: the queue hands the same skill's
                # notext/withtext jobs to different workers concurrently, and a
                # shared name lets one job hot-swap/unregister the adapter the
                # other is validating against (404s or silently wrong val).
                "--val-lora-name-base", f"val_{skill_id}_{spec['regime']}",
                "--val-workers", values.get("val_workers", 512),
                "--val-freq", values.get("val_freq", 2),
                "--val-concurrency", values.get("val_concurrency", 4),
            ]
            if values.get("val_fix_val", False):
                arguments.append("--fix-val")
                if values.get("fixed_val_offset") is not None:
                    arguments += ["--val-offset", values["fixed_val_offset"]]
            if spec["anchor_file"]:
                arguments += ["--anchor-file", spec["anchor_file"]]
                if values.get("anchor_near_count") is not None:
                    arguments += ["--anchor-near-count", values["anchor_near_count"]]
                if values.get("anchor_random_count") is not None:
                    arguments += ["--anchor-random-count", values["anchor_random_count"]]
            if api_base:
                arguments += ["--val-api-base", api_base]
            if values.get("val_async", True):
                arguments.append("--val-async")
            if not values.get("compute_val_loss", False):
                arguments.append("--skip-val-loss")
            if values.get("skip_base_baseline", True):
                arguments.append("--skip-base-baseline")
            if spec["regime"] == "notext":
                arguments.append("--strip-skill")
            gpu_cache = cache / f"gpu-{gpu}"
            gpu_temp = work / "tmp" / f"gpu-{gpu}"
            if not dry_run:
                gpu_cache.mkdir(parents=True, exist_ok=True)
                gpu_temp.mkdir(parents=True, exist_ok=True)
            environment = child_environment(gpu, gpu_cache, gpu_temp)
            runner.run(
                "msi.training.command", arguments, isolated=True,
                environment=environment,
            )

        _dispatch_training_jobs(job_specs, train_gpus, train_one)

    return 0
