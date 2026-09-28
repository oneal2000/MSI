"""Internal per-skill generator with work-local progress recovery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from msi.generation.admission import (
    QuestionDeduplicator,
    validate_admitted_instance,
)
from msi.generation.artifacts import (
    TASK_PROGRESS_SCHEMA,
    TASK_SCHEMA,
    TRAJECTORY_PROGRESS_SCHEMA,
    TRAJECTORY_SCHEMA,
    canonical_task,
    read_instances,
    read_tasks,
)
from msi.generation.progress import (
    append as append_progress,
    latest_tasks,
    latest_trajectories,
    next_attempt,
    prepare as prepare_progress,
    rows as progress_rows,
)
from msi.generation.schemas import DATASET_SCHEMAS
from msi.generation.tasks import generate_tasks
from msi.generation.teacher import teacher_extra_body
from msi.generation.trajectory import generate_trajectories_parallel
from msi.models.corpus import get_dataset_from_skill_id, load_skill
from msi.models.llm_client import create_client
from msi.protocol import (
    TEACHER_ENABLE_THINKING,
    TOOLQA_REACT_ENABLE_THINKING,
)
from msi.utils import atomic_json


FINAL_DATASETS = {"theoremqa", "medcalcbench", "logicbench", "toolqa"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _questions(rows: list[dict]) -> list[str]:
    return [
        str(row["question"]) for row in rows
        if str(row.get("question", "")).strip()
    ]


def _migrate_legacy_progress(
    state_dir: Path, skill_id: str, task_progress: Path, trajectory_progress: Path,
) -> None:
    """Import previous round-local journals once, without changing them."""
    if not task_progress.exists():
        legacy_tasks = sorted(
            path for path in state_dir.glob(f"**/{skill_id}_tasks_progress.jsonl")
            if path != task_progress
        )
        request_id = 0
        for source in legacy_tasks:
            for old in progress_rows(source, TASK_PROGRESS_SCHEMA):
                row = {
                    "schema": TASK_PROGRESS_SCHEMA,
                    "request_id": request_id,
                    "attempt": 1,
                    "status": old["status"],
                }
                if old["status"] == "success":
                    row["task"] = old["task"]
                else:
                    row["reason"] = str(old.get("reason", "failed"))
                append_progress(task_progress, row)
                request_id += 1
        if legacy_tasks:
            print(
                f"Imported {request_id} legacy task progress rows -> {task_progress}",
                flush=True,
            )

    if not trajectory_progress.exists():
        legacy_trajectories = sorted(
            path for path in state_dir.glob(f"**/{skill_id}_progress.jsonl")
            if path != trajectory_progress
        )
        attempts: dict[str, int] = {}
        imported = 0
        for source in legacy_trajectories:
            for old in progress_rows(source, TRAJECTORY_PROGRESS_SCHEMA):
                instance = old.get("instance") or {}
                task_id = str(old.get("task_id") or instance.get("task_id") or "")
                if not task_id:
                    continue
                attempts[task_id] = attempts.get(task_id, 0) + 1
                append_progress(trajectory_progress, {
                    "schema": TRAJECTORY_PROGRESS_SCHEMA,
                    "task_id": task_id,
                    "attempt": attempts[task_id],
                    "status": old["status"],
                    "instance": instance,
                })
                imported += 1
        if legacy_trajectories:
            print(
                f"Imported {imported} legacy trajectory progress rows -> "
                f"{trajectory_progress}",
                flush=True,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal per-skill generator; use `msi generate`.",
    )
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tasks-output", required=True)
    parser.add_argument("--trajectories-output", required=True)
    parser.add_argument("--min-acceptable", type=int, required=True)
    parser.add_argument("--task-oversample", type=float, default=1.1)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--phase", choices=("all", "tasks", "trajectories"), default="all")
    parser.add_argument(
        "--retry-failed", action=argparse.BooleanOptionalAction, default=False,
        help="Run one new attempt for failures already recorded in progress",
    )
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trajectory-max-tokens", type=int, required=True)
    parser.add_argument("--trajectory-output-max-tokens", type=int, required=True)
    parser.add_argument("--trajectory-tokenizer", required=True)
    parser.add_argument("--task-temperature", type=float, default=1.0)
    parser.add_argument("--dedup-threshold", type=float, default=0.88)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--task-model")
    parser.add_argument("--task-api-key")
    parser.add_argument("--task-api-base")
    parser.add_argument("--traj-model")
    parser.add_argument("--traj-api-key")
    parser.add_argument("--traj-api-base")
    parser.add_argument("--eval-model")
    parser.add_argument("--eval-api-key")
    parser.add_argument("--eval-api-base")
    parser.add_argument("--evaluator", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.min_acceptable < 1:
        raise SystemExit("--min-acceptable must be positive")
    if args.task_oversample < 1 or args.max_rounds < 1:
        raise SystemExit("--task-oversample must be >= 1 and --max-rounds positive")
    if not 512 <= args.trajectory_max_tokens <= 65536:
        raise SystemExit("--trajectory-max-tokens must be in [512, 65536]")
    if not 1 <= args.trajectory_output_max_tokens <= args.trajectory_max_tokens:
        raise SystemExit(
            "--trajectory-output-max-tokens must be positive and no greater "
            "than --trajectory-max-tokens"
        )
    if args.phase in {"all", "tasks"} and not (args.task_model and args.task_api_base):
        raise SystemExit("task generation requires --task-model and --task-api-base")
    if args.phase in {"all", "trajectories"} and not (
        args.traj_model and args.traj_api_base
    ):
        raise SystemExit("trajectory generation requires --traj-model and --traj-api-base")
    skill = load_skill(args.skill_id)
    if not skill:
        raise SystemExit(f"Skill {args.skill_id} not found in the declared corpus")
    skill_content = skill.get("content", "")
    skill_tools = skill.get("tools", [])
    dataset = get_dataset_from_skill_id(args.skill_id)
    if dataset not in FINAL_DATASETS or dataset not in DATASET_SCHEMAS:
        raise SystemExit(f"unsupported dataset: {dataset!r}")

    state_dir = Path(args.state_dir)
    decontamination_file = state_dir / "decontaminated_task_ids.json"
    blocked_task_ids: set[str] = set()
    if decontamination_file.is_file():
        payload = _read_json(decontamination_file)
        if (
            payload.get("schema") != "msi.decontaminated-task-ids"
            or payload.get("skill_id") != args.skill_id
            or not isinstance(payload.get("task_ids"), list)
        ):
            raise RuntimeError(
                f"invalid independent decontamination state: {decontamination_file}"
            )
        blocked_task_ids = {
            str(task_id) for task_id in payload.get("task_ids", []) if str(task_id)
        }
    task_progress = state_dir / "tasks.progress.jsonl"
    trajectory_progress = state_dir / "trajectories.progress.jsonl"
    tasks_output = Path(args.tasks_output)
    trajectories_output = Path(args.trajectories_output)
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"Progress: {state_dir.resolve()}", flush=True)
    print(f"Task output: {tasks_output.resolve()}", flush=True)
    print(f"Trajectory output: {trajectories_output.resolve()}", flush=True)

    prepare_progress(task_progress, TASK_PROGRESS_SCHEMA)
    prepare_progress(trajectory_progress, TRAJECTORY_PROGRESS_SCHEMA)
    atomic_json(state_dir / "request.json", {
        "schema": "msi.generation-request",
        "skill_id": args.skill_id,
        "phase": args.phase,
        "min_acceptable": args.min_acceptable,
        "task_oversample": args.task_oversample,
        "max_rounds": args.max_rounds,
        "task_model": args.task_model,
        "trajectory_model": args.traj_model,
        "evaluator_model": args.eval_model if args.evaluator else None,
        "teacher_thinking": TEACHER_ENABLE_THINKING,
        "toolqa_react_thinking": TOOLQA_REACT_ENABLE_THINKING,
        "teacher_request_max_tokens": args.trajectory_max_tokens,
        "trajectory_output_max_tokens": args.trajectory_output_max_tokens,
        "retry_failed": args.retry_failed,
    })
    _migrate_legacy_progress(
        state_dir, args.skill_id, task_progress, trajectory_progress,
    )

    task_payload = _read_json(tasks_output) if tasks_output.is_file() else {}
    published_tasks = read_tasks(tasks_output, args.skill_id)
    published_instances = read_instances(trajectories_output, args.skill_id)
    def current_tasks() -> list[dict]:
        merged = {row["task_id"]: row for row in published_tasks}
        for outcome in latest_tasks(task_progress).values():
            if outcome["status"] == "success":
                row = canonical_task(args.skill_id, outcome["task"])
                merged[row["task_id"]] = row
        for task_id in blocked_task_ids:
            merged.pop(task_id, None)
        return [merged[key] for key in sorted(merged)]

    def write_tasks() -> list[dict]:
        rows = current_tasks()
        if rows:
            atomic_json(tasks_output, {
                "schema": TASK_SCHEMA,
                "skill_id": args.skill_id,
                "dataset": dataset,
                "task_model": args.task_model or task_payload.get("task_model"),
                "tasks": rows,
                "stats": {"task_count": len(rows)},
            })
        return rows

    def current_instances() -> list[dict]:
        merged = {row["task_id"]: row for row in published_instances}
        for task_id, outcome in latest_trajectories(trajectory_progress).items():
            if outcome["status"] == "accepted":
                row = dict(outcome["instance"])
                row["task_id"] = task_id
                row["instance_id"] = "syn_" + task_id.removeprefix("task_")
                merged[task_id] = row
        for task_id in blocked_task_ids:
            merged.pop(task_id, None)
        return [merged[key] for key in sorted(merged)]

    def write_trajectories() -> list[dict]:
        rows = current_instances()
        if rows:
            atomic_json(trajectories_output, {
                "schema": TRAJECTORY_SCHEMA,
                "skill_id": args.skill_id,
                "dataset": dataset,
                "skill_content": skill_content,
                "task_model": args.task_model or task_payload.get("task_model"),
                "trajectory_model": args.traj_model,
                "evaluator_model": args.eval_model if args.evaluator else None,
                "trajectory_max_tokens": args.trajectory_max_tokens,
                "trajectory_output_max_tokens": args.trajectory_output_max_tokens,
                "evaluator_enabled": args.evaluator,
                "instances": rows,
                "stats": {"accepted_count": len(rows)},
            })
        return rows

    task_client = None

    def generate_task_requests(requests: list[tuple[int, int]]) -> list[dict]:
        nonlocal task_client
        if not requests:
            return current_tasks()
        existing = current_tasks()
        gate = QuestionDeduplicator(
            dedup_threshold=args.dedup_threshold,
            existing_questions=_questions(existing),
        )

        def admit_task(task: dict) -> tuple[bool, str]:
            clean = canonical_task(args.skill_id, task)
            if clean["task_id"] in blocked_task_ids:
                return False, "rejected by independent decontamination audit"
            accepted, reason = gate.admit(task.get("question"))
            if accepted:
                task.pop("task_id", None)
                task.clear()
                task.update(clean)
            return accepted, reason

        task_client = task_client or create_client(args.task_api_key, args.task_api_base)
        generate_tasks(
            task_client, args.task_model, skill_content, len(requests), dataset,
            skill=skill, delay=args.delay, num_workers=args.num_workers,
            use_parallel=True, temperature=args.task_temperature,
            progress_path=str(task_progress), avoid_questions=_questions(existing),
            admission=admit_task, requests=requests,
        )
        return write_tasks()

    if current_tasks():
        write_tasks()

    if args.retry_failed and args.phase in {"all", "tasks"}:
        prior = latest_tasks(task_progress)
        failed = [
            (request_id, next_attempt(row))
            for request_id, row in sorted(prior.items())
            if row["status"] == "fail"
        ]
        generate_task_requests(failed)

    if args.phase == "tasks":
        target = math.ceil(args.min_acceptable * args.task_oversample)
        requested = max(0, target - len(current_tasks()))
        prior = latest_tasks(task_progress)
        start = max([*prior, len(current_tasks()) - 1], default=-1) + 1
        generate_task_requests([(start + offset, 1) for offset in range(requested)])
        write_tasks()
        print(f"Task generation complete -> {tasks_output}", flush=True)
        return 0

    if not current_tasks():
        if args.phase == "trajectories":
            raise SystemExit(f"trajectory phase requires completed tasks: {tasks_output}")
        prior = latest_tasks(task_progress)
        start = max(prior, default=-1) + 1
        requested = math.ceil(args.min_acceptable * args.task_oversample)
        generate_task_requests([(start + offset, 1) for offset in range(requested)])

    traj_model = args.traj_model or args.task_model
    traj_api_key = args.traj_api_key or args.task_api_key
    traj_api_base = args.traj_api_base or args.task_api_base
    eval_model = args.eval_model or traj_model
    eval_api_key = args.eval_api_key or traj_api_key
    eval_api_base = args.eval_api_base or traj_api_base
    retry_trajectory_failures = args.retry_failed

    def run_trajectories() -> list[dict]:
        nonlocal retry_trajectory_failures
        tasks = current_tasks()
        traj_client = create_client(traj_api_key, traj_api_base)
        eval_client = create_client(eval_api_key, eval_api_base) if args.evaluator else None
        extra_body = teacher_extra_body(traj_model)
        from transformers import AutoTokenizer
        output_tokenizer = AutoTokenizer.from_pretrained(
            args.trajectory_tokenizer,
            local_files_only=True,
            trust_remote_code=True,
        )
        gate = QuestionDeduplicator(dedup_threshold=args.dedup_threshold)

        def admit_trajectory(instance: dict) -> tuple[bool, str]:
            accepted, reason = validate_admitted_instance(instance)
            return gate.admit(instance.get("question")) if accepted else (accepted, reason)

        generate_trajectories_parallel(
            traj_client, skill_content, tasks, args.skill_id, dataset, traj_model,
            tools=skill_tools, extra_body=extra_body,
            evaluator_enabled=args.evaluator, num_workers=args.num_workers,
            delay=args.delay, temperature=args.temperature,
            max_tokens=args.trajectory_max_tokens,
            output_max_tokens=args.trajectory_output_max_tokens,
            output_tokenizer=output_tokenizer,
            eval_client=eval_client, eval_model=eval_model,
            progress_path=str(trajectory_progress), admission=admit_trajectory,
            retry_failed=retry_trajectory_failures,
            completed_task_ids={row["task_id"] for row in published_instances},
        )
        retry_trajectory_failures = False
        return write_trajectories()

    if args.phase == "trajectories":
        instances = run_trajectories()
        print(
            f"Trajectory generation complete; accepted={len(instances)}, "
            f"deficit={max(0, args.min_acceptable-len(instances))} -> "
            f"{trajectories_output}",
            flush=True,
        )
        return 0

    rounds = 0
    while True:
        instances = run_trajectories()
        if len(instances) >= args.min_acceptable:
            print(
                f"Generation complete: accepted={len(instances)}/"
                f"{args.min_acceptable} -> {trajectories_output}",
                flush=True,
            )
            return 0
        if rounds >= args.max_rounds:
            raise RuntimeError(
                f"{args.skill_id} remains below target after {rounds} top-up rounds: "
                f"{len(instances)}/{args.min_acceptable}"
            )
        deficit = args.min_acceptable - len(instances)
        requested = math.ceil(deficit * args.task_oversample)
        prior = latest_tasks(task_progress)
        start = max([*prior, len(current_tasks()) - 1], default=-1) + 1
        generate_task_requests([(start + offset, 1) for offset in range(requested)])
        rounds += 1


if __name__ == "__main__":
    raise SystemExit(main())
