"""Teacher trajectory generation (toolqa ReAct, tool-augmented, single-turn)."""
from openai import OpenAI
import json
import os
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from msi.generation.artifacts import TRAJECTORY_PROGRESS_SCHEMA
from msi.generation.progress import latest_trajectories, next_attempt
from msi.models.config import EXTERNAL_DIR
from msi.models.llm_client import _chat, create_client, last_finish_reason
from msi.protocol import TOOLQA_REACT_ENABLE_THINKING
from msi.generation.teacher import teacher_extra_body
from msi.generation.prompts import TRAJECTORY_QUALITY_SYSTEM
from msi.generation.evaluate import (
    llm_validate_trajectory,
    validate_trajectory_format,
)
from sragents.prompts import build_prompt
from sragents.infer.engines.tool_loop import execute_tool, parse_tool_call
from sragents.infer.engines.react import ReActAgent
from sragents.toolqa import parse_action

_MAX_TOOL_ROUNDS = 5


def run_with_tools(
    client: OpenAI, model: str, system: str, user: str,
    tools: list[dict], *, max_tokens: int,
    extra_body: dict | None = None, temperature: float = 0.7,
) -> dict:
    """Run LLM with tool interception loop. Returns {output, turns}."""
    tool_index = {t["name"]: t for t in tools}
    messages = [{"role": "user", "content": user}]
    turns: list[dict] = []
    full_output = ""
    truncated = False

    for _ in range(_MAX_TOOL_ROUNDS):
        response_text = _chat(client, model, system, messages, max_tokens=max_tokens,
                              extra_body=extra_body, temperature=temperature)
        if last_finish_reason() == "length":
            truncated = True
        parsed = parse_tool_call(response_text, tool_index)
        if parsed is None:
            full_output += response_text
            turns.append({"role": "assistant", "content": response_text})
            break

        text_before, tool_name, args = parsed
        turns.append({"role": "assistant", "content": text_before})
        full_output += text_before

        try:
            result = execute_tool(tool_index[tool_name], args)
        except Exception as e:
            result = f"Error: {e}"

        full_output += f"\nTOOL_RESULT: {result}\n"
        turns.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
        messages.append({"role": "assistant", "content": text_before})
        messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})

    return {"output": full_output, "turns": turns, "truncated": truncated}
class _QualityGuidedReActAgent(ReActAgent):
    """ReActAgent with quality guidance and raw per-step response capture.

    Overrides ``_step`` to capture the actual model output (after think-tag
    stripping but before any normalization by ``_parse_response``). This raw
    output is what the student model should learn to produce.

    Also replicates the parent's scratchpad/model_scratchpad bookkeeping so
    that ``agent.scratchpad`` (used by the evaluator) remains correct.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_records: list[tuple[str, str]] = []
        self._step_truncated = False

    def _build_prompt(self) -> tuple[str, str]:
        system, user = super()._build_prompt()
        return TRAJECTORY_QUALITY_SYSTEM + "\n\n" + system, user

    def _parse_response(self, response: str) -> tuple[str, str]:
        thought, action = super()._parse_response(response)
        # Strip a model-emitted label before the engine adds its canonical
        # label; duplicated labels produce invalid tool actions.
        action = re.sub(r'^Action\s*\d+\s*:\s*', '', action.strip()).strip()
        return thought, action

    def _step(self):
        # Mirrors sragents.llm.chat but captures finish_reason: a step cut at
        # max_tokens means the whole trajectory is truncated mid-reasoning and
        # must be rejected downstream, not accepted.
        from sragents.llm import strip_think_tags

        system, user = self._build_prompt()
        extra = teacher_extra_body(self.model, thinking=self.thinking)
        basename = self.model.lower().rsplit("/", 1)[-1]
        stop = None if "gpt-5" in basename else [f"\nObservation {self.step_n}:"]

        pre_scratch = self.scratchpad

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        if stop:
            kwargs["stop"] = stop
        if extra:
            kwargs["extra_body"] = extra
        resp = self.client.chat.completions.create(**kwargs)
        if (resp.choices[0].finish_reason or "stop") == "length":
            self._step_truncated = True
        response = resp.choices[0].message.content or ""

        # Capture raw output for training (strip think tags only)
        raw_output = strip_think_tags(response).strip()
        self._step_records.append((pre_scratch, raw_output))

        # --- Replicate parent's _step bookkeeping below ---
        thought, action = self._parse_response(response)

        step_text = (
            f"\nThought {self.step_n}: {thought}"
            f"\nAction {self.step_n}: {action}"
        )
        self.model_scratchpad += step_text
        self.scratchpad += step_text
        self.scratchpad += f"\nObservation {self.step_n}: "

        if not action.strip():
            if not thought.strip():
                self.scratchpad += (
                    "Your response was empty. This can happen when "
                    "reasoning consumed the full output budget. "
                    "Please provide a concise Thought and a specific Action."
                )
            elif len(thought) >= 4000:
                self.scratchpad += (
                    "Your thought was truncated and the action was missing. "
                    "Please keep thoughts concise and provide a specific Action."
                )
            else:
                self.scratchpad += (
                    "Your action could not be parsed. "
                    "Please provide a valid Action on the next line."
                )
        else:
            action_type, argument = parse_action(action)
            if action_type == "Finish":
                self.answer = argument or ""
                self.scratchpad += f"Answer: {self.answer}"
                self.model_scratchpad += f"\nAnswer: {self.answer}"
                self.finished = True
            elif action_type == "LoadSkill":
                obs = self._handle_load_skill(argument)
                self.scratchpad += self._truncate_obs(obs)
            else:
                obs = self.tools.execute(action)
                self.scratchpad += self._truncate_obs(obs)

        self.step_n += 1
def generate_trajectory(
    client: OpenAI, skill_content: str, instance: dict,
    model: str, tools: list[dict] | None = None,
    extra_body: dict | None = None, temperature: float = 0.7,
    toolqa_env=None, toolqa_examples: str | None = None,
    *, max_tokens: int,
) -> dict:
    """Generate trajectory using benchmark's standard prompt format.

    The quality system prompt is injected into the teacher model's input
    to shape its reasoning style, but is NOT stored in the returned
    trajectory — the stored data uses the original system prompt to match
    the distribution the student model will see at inference time.
    """
    dataset = instance.get("dataset", "")

    # ToolQA: per-step training samples matching the inference format.
    # At inference, ReActAgent packs ALL history into the user prompt each
    # step — there is no assistant role. Each step is a separate API call:
    #   system + user(examples + scratchpad + "Thought N:") → model output
    # We reconstruct exactly this for each step.
    if dataset == "toolqa" and toolqa_env is not None:
        agent = _QualityGuidedReActAgent(
            question=instance["question"],
            tools=toolqa_env, client=client, model=model,
            examples=toolqa_examples or "",
            max_steps=20, max_tokens=max_tokens,
            temperature=temperature,
            skills=[skill_content],
            thinking=TOOLQA_REACT_ENABLE_THINKING,
        )
        agent.run()

        base_system, base_user = build_prompt(
            {"dataset": "toolqa", "question": instance["question"]},
            skills=[skill_content],
        )

        examples_header = ""
        if toolqa_examples:
            examples_header = (
                f"Here are some examples:\n{toolqa_examples}\n"
                f"(END OF EXAMPLES)\n"
            )

        step_samples = []
        for i, (pre_scratch, raw_output) in enumerate(agent._step_records):
            step_n = i + 1
            user_prompt = (
                f"{examples_header}"
                f"{base_user}"
                f"{pre_scratch}\n"
                f"Thought {step_n}:"
            )
            # Constrain the student target to correct step numbering: the
            # parser fix cleans the live scratchpad (context), but raw_output
            # (the training target) still carries the model's mis-numbered
            # "Action M:". Without normalizing it, the student would learn to
            # emit wrong step numbers. Align the target with the context.
            target = re.sub(
                r'\bAction\s*\d+\s*:', f'Action {step_n}:', raw_output, count=1
            )
            step_samples.append({
                "system": base_system,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target},
                ],
            })

        return {
            "output": agent.scratchpad,
            "truncated": agent._step_truncated,
            "training_samples": step_samples,
        }

    # Non-ToolQA datasets
    system, user = build_prompt(instance, skills=[skill_content])
    teacher_system = TRAJECTORY_QUALITY_SYSTEM + "\n\n" + system

    # Tool-augmented (e.g. medcalcbench): multi-turn conversation.
    # At inference, each turn sends the full conversation so far as context
    # and the model generates one assistant response. We split into per-turn
    # training samples — each contains the context up to that point plus the
    # new assistant response. Only the last assistant turn should have loss.
    if tools:
        result = run_with_tools(client, model, teacher_system, user, tools,
                                max_tokens=max_tokens, extra_body=extra_body,
                                temperature=temperature)
        all_messages = [{"role": "user", "content": user}] + result["turns"]

        training_samples = []
        accumulated: list[dict] = []
        for msg in all_messages:
            accumulated.append(msg)
            if msg["role"] == "assistant":
                training_samples.append({
                    "system": system,
                    "messages": list(accumulated),
                })

        return {
            "output": result["output"],
            "truncated": result["truncated"],
            "training_samples": training_samples,
        }

    # Simple single-turn: one API call → one training sample.
    # The per-call output cap is supplied by the resolved generation config.
    # It leaves prompt headroom inside the 20,480-token training budget and
    # prevents a tiny long tail from forcing every job for a skill to bs=1.
    # A generation that hits the cap is rejected below rather than silently
    # entering the dataset as a truncated/untrainable trajectory.
    output = _chat(client, model, teacher_system, [{"role": "user", "content": user}],
                   max_tokens=max_tokens, extra_body=extra_body, temperature=temperature)
    return {
        "output": output,
        "truncated": last_finish_reason() == "length",
        "training_samples": [{
            "system": system,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": output},
            ],
        }],
    }
def _prewarm_toolqa_env():
    """Pre-warm ToolQA caches (DataFrames, graphs, retrievers, SQLite).

    Loads pandas DataFrames, networkx graphs, and text retrievers into
    process-level caches so disk I/O happens sequentially before worker
    threads start.  Also populates the shared in-memory SQLite database
    so that every worker's ToolEnvironment (which connects to the same
    shared-cache URI) sees the tables already loaded and skips to_sql().
    """
    from sragents.toolqa import ToolEnvironment
    from sragents.toolqa.tools.table import TableToolkit
    from sragents.toolqa.tools.graph import GraphToolkit

    corpus_dir = EXTERNAL_DIR / "toolqa"
    env = ToolEnvironment(corpus_dir)
    env._get_agenda_retriever()
    env._get_scirex_retriever()
    table = TableToolkit(corpus_dir)
    graph = GraphToolkit(corpus_dir)
    for db_name in ["flights", "coffee", "airbnb", "yelp"]:
        try:
            table.load_db(db_name)
        except Exception:
            pass
    try:
        graph.load_graph("dblp")
    except Exception:
        pass

    # Load databases into the shared in-memory SQLite via the URI
    # shared-cache.  Worker threads open their own connections to the
    # same shared DB and skip to_sql() because _sql_loaded_tables is a
    # process-level set.
    for db_name in ["flights", "coffee", "airbnb", "yelp"]:
        try:
            env.execute(f"LoadDB[{db_name}]")
        except Exception:
            pass
def generate_trajectories_parallel(
    client: OpenAI, skill_content: str, tasks: list[dict],
    skill_id: str, dataset: str, model: str,
    tools: list[dict] | None = None, extra_body: dict | None = None,
    evaluator_enabled: bool = True, num_workers: int = 4, delay: float = 0,
    temperature: float = 0.7,
    eval_client: OpenAI | None = None, eval_model: str | None = None,
    progress_path: str | None = None,
    admission=None,
    retry_failed: bool = False,
    completed_task_ids: set[str] | None = None,
    *, max_tokens: int, output_max_tokens: int, output_tokenizer,
) -> tuple[list[dict], list[dict]]:
    """Generate trajectories using multiple parallel workers.

    Each worker thread gets its own OpenAI client (avoiding httpx connection
    pool contention) but shares process-level caches (DataFrames, retrievers,
    SQLite).  This mirrors sragents' own ``_BaseReActEngine._get_tools()``
    pattern: ``threading.local()`` for ToolEnvironment, shared module-level
    caches for heavy data.

    Args:
        progress_path: Append every accepted/failed attempt to this journal.
        retry_failed: Schedule one new attempt for previously failed tasks.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Resume by stable task ID. Published instances supplied by the caller are
    # complete even if their old run-local progress journal is unavailable.
    completed_task_ids = set(completed_task_ids or ())
    existing_instances: list[dict] = []
    existing_failures: list[dict] = []
    previous = latest_trajectories(Path(progress_path)) if progress_path else {}
    remaining_tasks: list[tuple[int, dict, int]] = []
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id", ""))
        if not task_id:
            raise RuntimeError("trajectory task is missing task_id")
        if task_id in completed_task_ids:
            continue
        outcome = previous.get(task_id)
        if outcome and outcome["status"] == "accepted":
            existing_instances.append(outcome["instance"])
            continue
        if outcome and outcome["status"] == "failed" and not retry_failed:
            existing_failures.append(outcome["instance"])
            continue
        remaining_tasks.append((index, task, next_attempt(outcome)))

    if previous:
        print(
            f"  Resuming from {len(previous)} trajectory progress records; "
            f"retry_failed={retry_failed}",
            flush=True,
        )

    if not remaining_tasks:
        print(f"  No trajectory attempts need to run", flush=True)
        return existing_instances, existing_failures

    toolqa_examples = None
    if dataset == "toolqa":
        from sragents.toolqa.fewshots import TOOLQA_EXAMPLES
        print("  Initializing ToolQA tool environment for trajectories...", flush=True)
        _prewarm_toolqa_env()
        toolqa_examples = TOOLQA_EXAMPLES

    print(
        f"  Using {num_workers} parallel workers for "
        f"{len(remaining_tasks)} trajectory attempts",
        flush=True,
    )

    # Per-thread state: each thread gets its own ToolEnvironment + OpenAI client
    _local = threading.local()

    # Connection params for per-thread client creation
    _api_base = str(client.base_url)
    _api_key = client.api_key
    _eval_api_base = str(eval_client.base_url) if eval_client else None
    _eval_api_key = eval_client.api_key if eval_client else None

    def _get_thread_resources():
        """Return (traj_client, eval_client_or_None, toolqa_env_or_None).

        Each thread lazily creates its own OpenAI clients and ToolEnvironment.
        Heavy data (DataFrames, embeddings) is shared via process-level caches.
        """
        if not hasattr(_local, "traj_client"):
            _local.traj_client = create_client(api_key=_api_key, base_url=_api_base)
            _local.eval_client = (create_client(api_key=_eval_api_key, base_url=_eval_api_base)
                                  if _eval_api_base else None)
            if dataset == "toolqa":
                from sragents.toolqa import ToolEnvironment
                _local.toolqa_env = ToolEnvironment(EXTERNAL_DIR / "toolqa")
        else:
            # Reset mutable state for reuse across instances
            if dataset == "toolqa":
                _local.toolqa_env.reset()
        return _local.traj_client, _local.eval_client, getattr(_local, "toolqa_env", None)

    instances: list[dict] = []
    failed_instances: list[dict] = []
    lock = threading.Lock()
    completed_count = [len(tasks) - len(remaining_tasks)]
    _progress_file = open(progress_path, "a") if progress_path else None

    def worker(args):
        idx, task, attempt = args
        question = task.get("question", "")
        eval_data = task.get("eval_data", {})

        if delay > 0:
            time.sleep(delay * idx % num_workers)

        instance = {
            "instance_id": "syn_" + str(task["task_id"]).removeprefix("task_"),
            "task_id": task.get("task_id"),
            "dataset": dataset,
            "question": question,
            "skill_annotations": [skill_id],
            "eval_data": eval_data,
        }

        t_client, t_eval_client, local_env = _get_thread_resources()

        try:
            trajectory = generate_trajectory(
                t_client, skill_content, instance, model, tools, extra_body,
                temperature=temperature,
                toolqa_env=local_env, toolqa_examples=toolqa_examples,
                max_tokens=max_tokens,
            )

            if trajectory.get("truncated"):
                # A truncated response is incomplete training data even when
                # its prefix happens to contain a plausible answer.
                validation = {
                    "acceptable": False, "valid": False,
                    "reason": "truncated: generation hit max_tokens "
                              "(finish_reason=length)",
                }
                instance["trajectory"] = trajectory
                instance["validation"] = validation
                return (idx, attempt, instance, False, validation)

            output_tokens = len(output_tokenizer.encode(
                trajectory.get("output", ""), add_special_tokens=False,
            ))
            if output_tokens > output_max_tokens:
                validation = {
                    "acceptable": False, "valid": False,
                    "reason": (
                        f"visible trajectory exceeds output token limit: "
                        f"{output_tokens} > {output_max_tokens}"
                    ),
                }
                instance["trajectory"] = trajectory
                instance["validation"] = validation
                return (idx, attempt, instance, False, validation)

            if evaluator_enabled:
                if not (t_eval_client and eval_model):
                    raise RuntimeError("evaluator is enabled but has no client/model")
                validation = llm_validate_trajectory(
                    t_eval_client, eval_model,
                    trajectory["output"], instance, skill_content, dataset,
                )
            else:
                validation = validate_trajectory_format(
                    trajectory["output"], instance,
                )
            acceptable = validation.get("acceptable", False)
            instance["trajectory"] = trajectory
            instance["validation"] = validation
            if acceptable and admission:
                acceptable, reason = admission(instance)
                if not acceptable:
                    validation = {
                        **validation, "acceptable": False, "valid": False,
                        "reason": reason,
                    }
                    instance["validation"] = validation
            return (idx, attempt, instance, acceptable, validation)
        except Exception as e:
            # Do NOT swallow per-task errors. A teacher that is unreachable after _chat's
            # 3 retries raises APIConnectionError here — recording it as a FAILED instance
            # (instead of letting it vanish from both valid+failed) keeps counts honest and
            # lets the downstream 0-valid fail-fast detect a dead teacher, instead of
            # silently saving an empty trajectory file (which traps regen + train_lora).
            err = f"WORKER_ERROR: {type(e).__name__}: {str(e)[:200]}"
            instance["trajectory"] = {"output": "", "error": err}
            instance["validation"] = {"acceptable": False, "valid": False, "reason": err}
            return (idx, attempt, instance, False, instance["validation"])

    total_tasks = len(tasks)

    def update_progress(result):
        idx, attempt, instance, valid, validation = result
        with lock:
            completed_count[0] += 1
            if valid:
                instances.append(instance)
                if validation:
                    extracted = validation.get("extracted_answer", "")[:20]
                    reason = validation.get("reason", "")[:60]
                    if reason:
                        print(f"  [{completed_count[0]}/{total_tasks}] Acceptable ({extracted}): {reason}", flush=True)
                    else:
                        print(f"  [{completed_count[0]}/{total_tasks}] Acceptable: {extracted}", flush=True)
                else:
                    print(f"  [{completed_count[0]}/{total_tasks}] Done", flush=True)
            else:
                failed_instances.append(instance)
                reason = validation.get("reason", "unknown") if validation else "unknown"
                print(f"  [{completed_count[0]}/{total_tasks}] Rejected: {reason[:300]}", flush=True)
            if _progress_file:
                outcome = {
                    "schema": TRAJECTORY_PROGRESS_SCHEMA,
                    "task_id": instance["task_id"],
                    "attempt": attempt,
                    "status": "accepted" if valid else "failed",
                    "instance": instance,
                }
                _progress_file.write(json.dumps(outcome, ensure_ascii=False) + "\n")
                _progress_file.flush()

    worker_args = remaining_tasks

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker, args): args for args in worker_args}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    idx, task, attempt = futures[future]
                    reason = (
                        f"WORKER_ERROR: {type(error).__name__}: "
                        f"{str(error)[:200]}"
                    )
                    instance = {
                        "instance_id": "syn_" + str(task["task_id"]).removeprefix("task_"),
                        "task_id": task["task_id"],
                        "dataset": dataset,
                        "question": task.get("question", ""),
                        "skill_annotations": [skill_id],
                        "eval_data": task.get("eval_data", {}),
                        "trajectory": {"output": "", "error": reason},
                        "validation": {
                            "acceptable": False, "valid": False, "reason": reason,
                        },
                    }
                    print(traceback.format_exc(), flush=True)
                    result = (
                        idx, attempt, instance, False, instance["validation"],
                    )
                update_progress(result)
    finally:
        if _progress_file:
            _progress_file.close()

    # Replay the journal so a successful retry supersedes its earlier failure.
    if progress_path:
        current = latest_trajectories(Path(progress_path))
        instances = [
            row["instance"] for row in current.values()
            if row["status"] == "accepted"
        ]
        failed_instances = [
            row["instance"] for row in current.values()
            if row["status"] == "failed"
        ]
    else:
        instances = existing_instances + instances
        failed_instances = existing_failures + failed_instances

    print(f"  Completed: {len(instances)}/{len(tasks)} valid trajectories", flush=True)
    return instances, failed_instances
