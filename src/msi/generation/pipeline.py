"""Configuration-driven synthetic task and trajectory generation."""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from msi.config import resolve_path
from msi.generation.artifacts import read_tasks
from msi.generation.progress import latest_trajectories
from msi.runtime.invoke import StageRunner
from msi.runtime.pipeline import path, repeated, selected_skill_ids, settings
from msi.runtime.validation_server import managed_validation_server
from msi.protocol import GENERATION_OUTPUT_MAX_TOKENS
from msi.training.provenance import synthetic_stage_path


@dataclass(frozen=True)
class _SkillRun:
    skill_id: str
    phase: str
    min_acceptable: int
    task_oversample: float
    state_dir: Path
    tasks_output: Path
    arguments: list[object]
    audit_arguments: list[object]
    audit_cycles: int


def run(config: dict, *, dry_run: bool = False) -> int:
    runner = StageRunner(dry_run=dry_run, python=config.get("_python"))
    run_id = str(config.get("run_id", "paper"))
    skill_ids = selected_skill_ids(config)
    if not skill_ids:
        raise ValueError("generation selection is empty")

    output_root = path(config, "synthetic_dir", "data/synthetic")
    work_root = path(config, "work_dir", "work") / run_id / "generation"
    tests = config.get("paths", {}).get("test_instances", {})
    if len(tests) != 4:
        raise ValueError(
            "independent post-generation audit requires all four benchmark files"
        )

    teacher = config.get("teacher", {})
    default_model = teacher.get("served_name") or teacher.get("model")
    external_api = teacher.get("api_base")
    selected_gpus = list(config.get("_selection", {}).get("gpus") or [])
    if external_api:
        server = nullcontext(external_api)
    elif selected_gpus:
        model_path = teacher.get("path")
        if not model_path or not default_model:
            raise ValueError(
                "managed teacher requires teacher.path and teacher.model"
            )
        runtime = config.get("runtime", {})
        server = managed_validation_server(
            python=config.get("_python") or runner.python,
            model_path=str(resolve_path(model_path)),
            served_name=str(default_model),
            gpu=selected_gpus,
            port=int(runtime.get("port", 8000)),
            shared_root=path(config, "shared_root", "."),
            cache_dir=path(config, "cache_dir", f"work/{run_id}/cache") / "teacher",
            temp_dir=work_root / "tmp" / "teacher",
            log_path=work_root / "logs" / "teacher-server.log",
            gpu_memory_utilization=float(
                runtime.get("gpu_memory_utilization", 0.92)
            ),
            max_model_len=(
                int(runtime["max_model_len"])
                if runtime.get("max_model_len") is not None else None
            ),
            lora=False,
            dry_run=dry_run,
        )
    else:
        server = nullcontext(None)

    with server as default_api:
        return _run_skills(
            config=config, runner=runner, skill_ids=skill_ids,
            output_root=output_root, work_root=work_root, tests=tests,
            teacher=teacher, default_model=default_model,
            default_api=default_api,
        )


def _run_skills(
    *, config: dict, runner: StageRunner, skill_ids: list[str],
    output_root, work_root, tests: dict, teacher: dict,
    default_model: str | None, default_api: str | None,
) -> int:
    task_model = teacher.get("task_model") or default_model
    task_api = teacher.get("task_api_base") or default_api
    trajectory_model = teacher.get("trajectory_model") or default_model
    trajectory_api = teacher.get("trajectory_api_base") or default_api
    evaluator_model = teacher.get("evaluator_model") or trajectory_model
    evaluator_api = teacher.get("evaluator_api_base") or trajectory_api
    trajectory_tokenizer = (
        teacher.get("tokenizer") or teacher.get("path") or teacher.get("model")
    )

    runs: list[_SkillRun] = []
    for skill_id in skill_ids:
        dataset = skill_id.rsplit("_", 1)[0]
        values = settings(config, dataset=dataset, skill=skill_id)
        phase = str(values.get("phase", "all"))
        evaluator = bool(values.get("evaluator", True))
        if phase not in {"all", "tasks", "trajectories"}:
            raise ValueError(f"unsupported generation phase: {phase}")
        if phase in {"all", "tasks"} and not (task_model and task_api):
            raise ValueError("task generation requires a teacher model and API base")
        if phase in {"all", "trajectories"} and not (
            trajectory_model and trajectory_api
        ):
            raise ValueError(
                "trajectory generation requires a teacher model and API base"
            )
        if evaluator and phase in {"all", "trajectories"} and not (
            evaluator_model and evaluator_api
        ):
            raise ValueError("the trajectory evaluator requires a model and API base")
        state_dir = work_root / dataset / skill_id
        tasks_output = synthetic_stage_path(output_root, skill_id, "tasks")
        trajectories_output = synthetic_stage_path(
            output_root, skill_id, "trajectories"
        )
        print(f"{skill_id} progress: {state_dir}", flush=True)
        print(
            f"{skill_id} final outputs: {tasks_output}, {trajectories_output}",
            flush=True,
        )

        arguments: list[object] = [
            "--skill-id", skill_id,
            "--state-dir", state_dir,
            "--tasks-output", tasks_output,
            "--trajectories-output", trajectories_output,
            "--phase", phase,
            "--min-acceptable", values.get("min_acceptable", 280),
            "--task-oversample", values.get("task_oversample", 1.1),
            "--max-rounds", values.get("max_rounds", 16),
            "--trajectory-max-tokens", values["max_tokens"],
            "--trajectory-output-max-tokens", values.get(
                "output_max_tokens", GENERATION_OUTPUT_MAX_TOKENS[dataset],
            ),
            "--trajectory-tokenizer", trajectory_tokenizer,
            "--task-model", task_model,
            "--task-api-base", task_api,
            "--traj-model", trajectory_model,
            "--traj-api-base", trajectory_api,
            "--eval-model", evaluator_model,
            "--eval-api-base", evaluator_api,
            "--num-workers", values.get("workers", 10),
            "--temperature", values.get("temperature", 0.7),
            "--task-temperature", values.get("task_temperature", 1.0),
            "--delay", values.get("delay", 0),
            "--dedup-threshold", values.get("dedup_threshold", 0.88),
            "--evaluator" if evaluator else "--no-evaluator",
            "--retry-failed" if values.get("retry_failed", False)
            else "--no-retry-failed",
        ]
        audit_arguments: list[object] = [
            "--skill-id", skill_id,
            "--state-dir", state_dir,
            "--tasks", tasks_output,
            "--trajectories", trajectories_output,
            "--near-threshold", values.get("near_threshold", 0.92),
            *repeated("--test-instances", tests.values()),
        ]
        runs.append(_SkillRun(
            skill_id=skill_id,
            phase=phase,
            min_acceptable=int(values.get("min_acceptable", 280)),
            task_oversample=float(values.get("task_oversample", 1.1)),
            state_dir=state_dir,
            tasks_output=tasks_output,
            arguments=arguments,
            audit_arguments=audit_arguments,
            audit_cycles=int(values.get("max_rounds", 16)),
        ))

    if runner.dry_run or not bool(config.get("runtime", {}).get("skill_lookahead", False)):
        for skill_run in runs:
            _execute_skill(runner, skill_run)
        return 0

    poll_seconds = float(
        config.get("runtime", {}).get("skill_lookahead_poll_seconds", 5)
    )
    threshold = int(
        config.get("runtime", {}).get("skill_lookahead_threshold", 32)
    )
    if poll_seconds <= 0:
        raise ValueError("runtime.skill_lookahead_poll_seconds must be positive")
    if threshold <= 0:
        raise ValueError("runtime.skill_lookahead_threshold must be positive")
    _run_with_lookahead(
        runner, runs, poll_seconds=poll_seconds, threshold=threshold,
    )
    return 0


def _execute_skill(runner: StageRunner, skill_run: _SkillRun) -> None:
    for cycle in range(skill_run.audit_cycles + 1):
        # The generator subprocess receives no benchmark path or content.
        runner.run(
            "msi.generation.command", skill_run.arguments, isolated=True,
        )
        audit_code = runner.run(
            "msi.audit.decontaminate", skill_run.audit_arguments,
            isolated=True, allowed_codes=(0, 3),
        )
        if audit_code == 0:
            return
        if skill_run.phase == "trajectories":
            raise SystemExit(
                "independent audit removed contaminated rows for "
                f"{skill_run.skill_id}; rerun with phase=all so replacement "
                "tasks can be generated"
            )
        if cycle >= skill_run.audit_cycles:
            raise RuntimeError(
                f"{skill_run.skill_id} still produces benchmark-overlapping "
                f"questions after {skill_run.audit_cycles + 1} independent "
                "audit cycles"
            )


def _next_round_size(skill_run: _SkillRun) -> int | None:
    """Return the next top-up round size once the current round is complete."""
    if skill_run.phase != "all":
        return None
    try:
        tasks = read_tasks(skill_run.tasks_output, skill_run.skill_id)
        if not tasks:
            return None
        task_ids = {str(task["task_id"]) for task in tasks}
        outcomes = latest_trajectories(
            skill_run.state_dir / "trajectories.progress.jsonl"
        )
    except (OSError, RuntimeError, ValueError):
        # A later poll will observe the complete atomically-published state.
        return None
    if not task_ids.issubset(outcomes):
        return None
    accepted = sum(
        outcomes[task_id]["status"] == "accepted" for task_id in task_ids
    )
    deficit = skill_run.min_acceptable - accepted
    if deficit <= 0:
        return None
    return math.ceil(deficit * skill_run.task_oversample)


def _run_with_lookahead(
    runner: StageRunner, runs: list[_SkillRun], *, poll_seconds: float,
    threshold: int,
) -> None:
    """Overlap only a deficient skill's top-up tail with its successor."""
    if not runs:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        current_run = runs[0]
        current: Future[None] = pool.submit(_execute_skill, runner, current_run)
        for next_run in runs[1:]:
            next_round = _next_round_size(current_run)
            while (
                not current.done()
                and (next_round is None or next_round >= threshold)
            ):
                time.sleep(poll_seconds)
                next_round = _next_round_size(current_run)
            if current.done():
                current.result()
            else:
                print(
                    f"{current_run.skill_id} next round has {next_round} "
                    f"requests (< {threshold}); starting lookahead skill "
                    f"{next_run.skill_id}",
                    flush=True,
                )
            successor = pool.submit(_execute_skill, runner, next_run)
            if not current.done():
                current.result()
            current_run = next_run
            current = successor
        current.result()
