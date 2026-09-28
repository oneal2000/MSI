"""Two-stage anchor preparation and answer generation pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path

from msi.anchors.artifacts import (
    assigned_instance_ids, assignment_path, load_assignments, load_task_pool,
    published_json_matches,
)
from msi.anchors.completion import answer_cache_complete
from msi.anchors.retrieve import retrieval_complete, retrieval_metadata
from msi.runtime.gpus import child_environment
from msi.runtime.invoke import StageRunner
from msi.runtime.pipeline import (
    datasets, path, repeated, selected_skill_ids, selection_scope, settings, skills,
)
from msi.runtime.validation_server import managed_validation_server
from msi.utils import atomic_json


def _count_overrides(config: dict, skill_ids: list[str]) -> dict:
    """Resolve model-independent sampling counts for published assignments."""
    result = {}
    for skill_id in skill_ids:
        dataset = skill_id.rsplit("_", 1)[0]
        values = settings(config, dataset=dataset, skill=skill_id)
        result[skill_id] = {
            "n_nearmiss": int(values.get("near_count", 80)),
            "n_random": int(values.get("random_count", 80)),
        }
    return result


def _retrieval_arguments(config: dict, task_pool: Path, corpus: Path) -> list[object]:
    retrieval = config.get("retrieval", {})
    return [
        "--task-pool", task_pool, "--corpus", corpus,
        "--retriever", retrieval.get("first_stage", "bge_base"),
        "--model", retrieval.get("model", "BAAI/bge-base-en-v1.5"),
        "--first-stage-top-k", retrieval.get("first_stage_top_k", 10),
        "--top-k", retrieval.get("top_k", 10),
        "--batch-size", retrieval.get("batch_size", 256),
        "--query-chunk-size", retrieval.get("query_chunk_size", 4096),
        "--reranker-batch-size", retrieval.get("reranker_batch_size", 32),
        "--reranker-max-length", retrieval.get("reranker_max_length", 512),
        "--dtype", retrieval.get("dtype", "bfloat16"),
        *(["--reranker-model", retrieval["reranker_model"]]
          if retrieval.get("reranker_model") else []),
    ]


def _prepare(config: dict, runner: StageRunner, *, dry_run: bool) -> int:
    run_id = str(config.get("run_id", "paper"))
    selected = config.get("_selection", {})
    selected_datasets = datasets(config)
    selected_skills = selected_skill_ids(config)
    explicit_skills = skills(config)
    corpus = path(config, "corpus")
    synthetic = path(config, "synthetic_dir", "data/synthetic")
    task_pool = path(config, "anchor_task_pool", "data/anchor_tasks/task-pool.json")
    retrieval_dir = path(config, "anchor_retrieval_dir", "data/anchor_retrieval")
    assignment_dir = path(
        config, "anchor_assignment_dir", "data/anchor_tasks/assignments"
    )
    retrieval_cfg = config.get("retrieval", {})
    profile = str(retrieval_cfg.get("name", "bge_base_rerank"))
    retrieval_output = retrieval_dir / f"{profile}.json"
    assignment_output = assignment_dir / profile
    work = path(config, "work_dir", "work") / run_id / "anchors" / "prepare"
    prepared_pool = work / "task-pool.json"
    counts_file = work / "counts.json"
    tests = config.get("paths", {}).get("test_instances", {})
    source_file = (
        path(config, "source_file") if config.get("paths", {}).get("source_file") else None
    )

    values = settings(config)
    source_arguments = (["--source-file", source_file] if source_file else [
        "--trajectory-dir", synthetic,
        "--num-train", values["num_train"], "--num-val", values["num_val"],
        "--split-seed", values["split_seed"],
        "--fixed-val-offset", values["fixed_val_offset"],
    ])
    runner.run("msi.anchors.source", [
        *source_arguments, "--skill-corpus", corpus, "--output", prepared_pool,
    ])
    runner.run("msi.audit.leakage", [
        "--trajectory", prepared_pool,
        *repeated("--test-instances", tests.values()),
        "--skill-corpus", corpus,
    ])
    if not dry_run:
        atomic_json(counts_file, _count_overrides(config, selected_skills))

    complete = False
    published_pool_matches = False
    if not dry_run:
        pool_metadata, task_rows = load_task_pool(prepared_pool)
        prepared_payload = json.loads(prepared_pool.read_text(encoding="utf-8"))
        published_pool_matches = published_json_matches(
            task_pool, prepared_payload, "anchor task pool"
        )
        task_pool_id = str(pool_metadata.get("task_pool_id", ""))
        if not task_pool_id:
            raise SystemExit("prepared anchor task pool is missing task_pool_id")
        expected_metadata = retrieval_metadata(
            retriever=str(retrieval_cfg.get("first_stage", "bge_base")),
            model=str(retrieval_cfg.get("model", "BAAI/bge-base-en-v1.5")),
            reranker_model=retrieval_cfg.get("reranker_model"),
            first_stage_top_k=int(retrieval_cfg.get("first_stage_top_k", 10)),
            top_k=int(retrieval_cfg.get("top_k", 10)),
            dtype=str(retrieval_cfg.get("dtype", "bfloat16")),
            task_pool_id=task_pool_id,
        )
        task_ids = [row["instance_id"] for row in task_rows]
        corpus_ids = {
            row["skill_id"] for row in json.loads(corpus.read_text(encoding="utf-8"))
        }
        complete = retrieval_complete(
            retrieval_output, metadata=expected_metadata,
            expected_ids=task_ids, corpus_ids=corpus_ids,
            expected_gold={
                row["instance_id"]: list(row.get("gold_skill_ids") or [])
                for row in task_rows
            },
        )
        if retrieval_output.exists() and not complete:
            raise SystemExit(
                f"published retrieval conflicts with current task pool/settings: "
                f"{retrieval_output}"
            )
    if complete:
        print(f"reusing complete published anchor retrieval -> {retrieval_output}")
    else:
        selected_gpus = list(selected.get("gpus") or [])
        dense = retrieval_cfg.get("first_stage", "bge_base") != "bm25"
        if dense and not selected_gpus and not dry_run:
            raise ValueError("dense anchor preparation requires --gpu or visible GPUs")
        worker_gpus: list[str | None] = selected_gpus if dense else [None]
        if dry_run and dense and not worker_gpus:
            worker_gpus = [None]
        shard_count = len(worker_gpus)
        shard_dir = work / "retrieval-shards" / profile
        shard_files = [
            shard_dir / f"shard-{index:05d}-of-{shard_count:05d}.json"
            for index in range(shard_count)
        ]
        base_arguments = _retrieval_arguments(config, prepared_pool, corpus)

        def run_shard(index: int, gpu: str | None) -> None:
            environment = None
            arguments = [
                *base_arguments, "--output", shard_files[index],
                "--shard-index", index, "--shard-count", shard_count,
            ]
            if gpu:
                cache = path(config, "cache_dir", f"work/{run_id}/cache")
                gpu_cache = cache / f"gpu-{gpu}"
                gpu_temp = work / "tmp" / f"gpu-{gpu}"
                if not dry_run:
                    gpu_cache.mkdir(parents=True, exist_ok=True)
                    gpu_temp.mkdir(parents=True, exist_ok=True)
                environment = child_environment(gpu, gpu_cache, gpu_temp)
                arguments += ["--device", "cuda:0"]
            runner.run(
                "msi.anchors.retrieve", arguments,
                isolated=True, environment=environment,
            )

        with ThreadPoolExecutor(max_workers=shard_count) as pool:
            futures = [
                pool.submit(run_shard, index, gpu)
                for index, gpu in enumerate(worker_gpus)
            ]
            for future in futures:
                future.result()
        runner.run("msi.anchors.retrieval_output", [
            "--task-pool", prepared_pool, "--corpus", corpus,
            *repeated("--input", shard_files), "--output", retrieval_output,
        ])

    if not dry_run:
        if published_pool_matches:
            print(f"reusing identical published anchor task pool -> {task_pool}")
        else:
            atomic_json(task_pool, prepared_payload)
    values = settings(config)
    runner.run("msi.anchors.assign", [
        "--task-pool", task_pool, "--retrieval", retrieval_output,
        "--skill-corpus", corpus, "--output-dir", assignment_output,
        "--n-nearmiss", values.get("near_count", 80),
        "--n-random", values.get("random_count", 80),
        "--oversample", values.get("oversample", 1.6),
        "--counts-config", counts_file,
        "--required-retriever", profile,
        *(repeated("--target-skill", selected_skills) if explicit_skills
          else repeated("--target-prefix", selected_datasets)),
    ])
    return 0


def _answer(config: dict, runner: StageRunner, *, dry_run: bool) -> int:
    run_id = str(config.get("run_id", "paper"))
    selected = config.get("_selection", {})
    selected_datasets = datasets(config)
    selected_skills = selected_skill_ids(config)
    explicit_skills = skills(config)
    scope = selection_scope(config)
    model_name = selected.get("model")
    if not model_name:
        raise ValueError("anchors answer requires --base-model")
    model = config.get("models", {}).get(model_name)
    if not model:
        raise ValueError(f"anchors config has no model {model_name!r}")
    corpus = path(config, "corpus")
    synthetic = path(config, "synthetic_dir", "data/synthetic")
    task_pool = path(config, "anchor_task_pool", "data/anchor_tasks/task-pool.json")
    assignment_root = path(
        config, "anchor_assignment_dir", "data/anchor_tasks/assignments"
    ) / str(config.get("retrieval", {}).get("name", "bge_base_rerank"))
    assignment_files = [assignment_path(assignment_root, skill) for skill in selected_skills]
    if not dry_run:
        missing = [str(source) for source in assignment_files if not source.is_file()]
        if missing:
            raise SystemExit(
                "published anchor task assignments are missing; run `anchors prepare` "
                f"first: {missing[:20]}"
            )
        pool_metadata, task_rows = load_task_pool(task_pool)
        loaded_assignments = load_assignments(assignment_files)
        if set(loaded_assignments) != set(selected_skills):
            raise SystemExit("published assignments do not match the requested skills")
        actual_profile = next(iter(loaded_assignments.values()))["retrieval"]["profile"]
        if actual_profile != config.get("retrieval", {}).get("name", "bge_base_rerank"):
            raise SystemExit("published assignments use a different retrieval profile")
        assignment_pool_ids = {
            row["retrieval"].get("task_pool_id") for row in loaded_assignments.values()
        }
        if assignment_pool_ids != {pool_metadata.get("task_pool_id")}:
            raise SystemExit("published assignments belong to a different task pool")
        task_ids = {row["instance_id"] for row in task_rows}
        if not set(assigned_instance_ids(loaded_assignments)) <= task_ids:
            raise SystemExit("published assignments reference tasks outside the task pool")
    work = path(config, "work_dir", "work") / run_id / "anchors" / "answer"
    target_work = work / model_name / scope
    answers = target_work / "answers.json"
    output = path(config, "anchor_dir", "data/anchors") / model_name
    values = settings(config, model=model_name)
    served_model = selected.get("served_model") or model_name
    max_answer_tokens = int(values.get("base_answer_max_tokens", 4096))
    answers_complete = False if dry_run else answer_cache_complete(
        answers, task_pool, assignment_files, model_name,
        served_model, max_answer_tokens,
    )
    runtime = config.get("runtime", {})
    external_api_base = runtime.get("api_base")
    selected_gpus = list(selected.get("gpus") or [])
    if not answers_complete and not external_api_base and not selected_gpus:
        raise ValueError(
            "anchor answer generation requires --gpu, or --api-base to reuse an endpoint"
        )
    if answers_complete:
        print(f"answer cache already complete; server startup skipped -> {answers}")
        server = nullcontext(None)
    elif external_api_base:
        server = nullcontext(external_api_base)
    else:
        anchor_gpu = str(selected_gpus[0])
        server = managed_validation_server(
            python=config.get("_python") or runner.python,
            model_path=model["path"], served_name=served_model, gpu=anchor_gpu,
            port=int(runtime.get("port", 8000)),
            shared_root=path(config, "shared_root", "."),
            cache_dir=path(config, "cache_dir", f"work/{run_id}/cache") / f"gpu-{anchor_gpu}",
            temp_dir=work / "tmp" / f"gpu-{anchor_gpu}",
            log_path=work / "logs" / model_name / "answer-server.log",
            gpu_memory_utilization=float(runtime.get("gpu_memory_utilization", 0.92)),
            max_model_len=(int(runtime["max_model_len"])
                           if runtime.get("max_model_len") is not None else None),
            dry_run=dry_run,
        )
    with server as api_base:
        if not answers_complete:
            runner.run("msi.anchors.answer_cache", [
                "--task-pool", task_pool,
                *repeated("--assignment", assignment_files),
                "--api-base", api_base, "--base-model", model_name,
                "--served-model", served_model,
                "--max-tokens", max_answer_tokens,
                "--workers", values.get("workers", 128), "--output", answers,
            ], isolated=True)
        runner.run("msi.anchors.assemble", [
            "--task-pool", task_pool,
            *repeated("--assignment", assignment_files),
            "--answer-cache", answers, "--skill-corpus", corpus,
            "--output-dir", output, "--base-model", model_name,
        ])
    result = 0
    tests = config.get("paths", {}).get("test_instances", {})
    audit_groups = [None] if explicit_skills else selected_datasets
    for dataset in audit_groups:
        result = runner.run("msi.audit.leakage", [
            "--trajectory-dir", synthetic, "--anchor-dir", output,
            *repeated("--test-instances", tests.values()),
            *(repeated("--skill", selected_skills) if explicit_skills
              else ["--dataset", dataset]),
            "--required-anchor-retriever",
            config.get("retrieval", {}).get("name", "bge_base_rerank"),
            "--skill-corpus", corpus,
        ])
    return result


def run(config: dict, *, dry_run: bool = False) -> int:
    runner = StageRunner(dry_run=dry_run, python=config.get("_python"))
    action = config.get("_selection", {}).get("anchor_action")
    if action == "prepare":
        return _prepare(config, runner, dry_run=dry_run)
    if action == "answer":
        return _answer(config, runner, dry_run=dry_run)
    raise ValueError("anchors requires either the prepare or answer subcommand")
