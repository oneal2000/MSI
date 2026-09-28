"""Validation: skill-tool loading, answer comparison, generation + teacher-forced val (vllm / inprocess)."""
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import torch

from msi.models.config import CORPUS_PATH, EXTERNAL_DIR
from msi.models.inprocess_peft import batched_generate, batched_tool_validate
from sragents.evaluate import evaluate as eval_dispatch
from sragents.evaluate.common import within_eps
from sragents.infer import get_engine
from sragents.infer.engines.tool_loop import run_with_tools
from sragents.llm import chat, get_extra_body, strip_think_tags
from sragents.prompts import build_prompt

from msi.training.dataset import _build_masked_sample, _strip_skill_from_messages


# medcalc._extract is vendored-internal; fall back to a local extractor if the
# private symbol is ever renamed/removed upstream.
try:
    from sragents.evaluate.datasets.medcalcbench import _extract as _medcalc_extract
except Exception:  # noqa: BLE001
    def _medcalc_extract(raw_output: str, eval_data: dict | None = None) -> str:
        for line in reversed(raw_output.strip().split("\n")):
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                return line[len("ANSWER:"):].strip().strip("*").strip()
        nums = re.findall(r"-?\d+\.?\d*", raw_output)
        return nums[-1] if nums else ""
    print("WARNING: could not import medcalcbench._extract; using local fallback")
def load_skill_tools(skill_id: str) -> list[dict]:
    """Load executable ``tools`` for a skill from the corpus (medcalc only).

    theoremqa/logicbench/... carry no tools ([]); medcalc skills expose a
    ``tools`` field (name/description/parameters/implementation) consumed by
    ``run_with_tools`` during validation. Skill prose itself comes from the
    trajectory file's top-level ``skill_content`` (no corpus read needed).
    """
    if skill_id.split("_")[0] != "medcalcbench":
        return []
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    for s in corpus:
        if s.get("skill_id") == skill_id:
            return s.get("tools", []) or []
    print(f"WARNING: skill {skill_id} not in corpus; assuming no tools")
    return []
def compare_answer(raw_output: str, gt: str, dataset: str, question: str = "") -> bool:
    """Compare a generated answer to the teacher's ``validation.extracted_answer``.

    Routes by dataset: theoremqa & logicbench reuse the full sragents evaluator
    (synthetic eval_data from extracted_answer; logicbench MCQA needs the question);
    medcalcbench extracts then typed-compares (no range bounds available).
    NOTE: proxy, not the official benchmark evaluator (trajectories carry no
    eval_data bounds).
    """
    gt = str(gt).strip()
    if not gt:
        return False
    clean = strip_think_tags(raw_output or "")
    if dataset == "theoremqa":
        # Full reuse: handles latex/numeric/bool/MC with 4% tolerance.
        return bool(eval_dispatch(
            clean,
            {"dataset": "theoremqa",
             "eval_data": {"answer": gt, "answer_type": "float"}},
        )["correct"])
    if dataset == "logicbench":
        # BQA (yes/no) or MCQA (choice_X); MCQA extraction needs the question.
        task_type = "MCQA" if gt.lower().startswith("choice") else "BQA"
        return bool(eval_dispatch(
            clean,
            {"dataset": "logicbench", "question": question,
             "eval_data": {"answer": gt, "task_type": task_type}},
        )["correct"])
    if dataset == "toolqa":
        # ReAct: answer is inside the Finish[<answer>] action. The vendor toolqa
        # evaluator extracts it + normalizes; without this branch toolqa fell into
        # the medcalc default extractor and always scored 0.
        return bool(eval_dispatch(
            clean,
            {"dataset": "toolqa", "eval_data": {"answer": gt}},
        )["correct"])
    # medcalcbench — extract then typed-compare against gt
    pred = _medcalc_extract(clean, {})
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", gt):           # date
        try:
            return (datetime.strptime(pred.strip(), "%m/%d/%Y")
                    == datetime.strptime(gt, "%m/%d/%Y"))
        except ValueError:
            return False
    if re.fullmatch(r"-?\d+", gt):                            # integer
        try:
            return round(float(pred)) == int(gt)
        except (ValueError, TypeError):
            return False
    if re.fullmatch(r"-?\d+\.\d+", gt):                       # float (4% tol)
        try:
            return within_eps(float(pred), float(gt))
        except (ValueError, TypeError):
            return False
    return pred.strip().rstrip(".").lower() == gt.strip().rstrip(".").lower()
def extract_answer(raw_output: str, dataset: str, instance: dict | None = None) -> str:
    """Extract the canonical answer value from a model output (additive helper).

    Turns an anchor's base-model ``assistant`` output into a clean gt for anchor
    validation — mirrors the *pred* extraction inside :func:`compare_answer` so
    the anchor "correct" criterion is symmetric (extract from both the adapter
    output and the base output, then compare). Does NOT alter compare_answer.
    """
    clean = strip_think_tags(raw_output or "")
    if dataset == "theoremqa":
        from sragents.evaluate.datasets.theoremqa import _extract as _thm_extract
        return _thm_extract(clean)
    if dataset == "logicbench":
        from sragents.evaluate.datasets.logicbench import _extract as _lb_extract
        return _lb_extract(clean, instance or {})
    if dataset == "toolqa":
        from sragents.evaluate.datasets.toolqa import _extract as _tq_extract
        return _tq_extract(clean)
    return _medcalc_extract(clean, {})
def _make_validation_client(api_base: str, workers: int):
    """OpenAI client with a connection pool sized for high-concurrency validation.

    The default openai/httpx pool caps concurrent requests well below what the
    vLLM validate server (``--max-num-seqs 512``) can absorb; the server only
    saturates (~256 concurrent to fill batches). Size the pool to the worker
    count so we actually feed it instead of client-side throttling.
    """
    import httpx
    from openai import OpenAI
    from msi.models.llm_client import request_timeout
    limits = httpx.Limits(max_connections=max(workers + 32, 128),
                          max_keepalive_connections=max(workers, 64))
    return OpenAI(
        base_url=api_base,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        http_client=httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(
                request_timeout(), connect=10.0,
            ),
        ),
    )
def run_validation(
    val_instances: list[dict], skill_content: str, tools: list[dict], *,
    api_base: str, lora_name: str, dataset: str, base_model_name: str,
    temperature: float = 0.0, max_tokens: int = 4096, workers: int = 256,
    client=None,
) -> tuple[float, list[dict]]:
    """Generate answers and score them.

    Routes by dataset exactly like run_inference: medcalc (tools) via
    ``run_with_tools`` (TOOL_CALL interception + execution), theoremqa via
    single-shot ``chat``. ``base_model_name`` is passed to ``get_extra_body`` to
    suppress thinking; ``lora_name`` is used only as the API ``model=`` field.

    With ``val_backend="inprocess"``, pass a pre-built ``client`` (an
    :class:`lora.inprocess_peft.InProcessClient`); ``api_base`` is then unused
    and generation runs on the in-process PeftModel (adapter deterministically
    applied — unlike vLLM's LoRA engine on Qwen3.5).
    """
    if client is None:
        client = _make_validation_client(api_base, workers)
    extra = get_extra_body(base_model_name, thinking=False)
    skills = [skill_content] if skill_content else None
    use_tools = bool(tools)

    # ToolQA must validate via multi-step ReAct with real DB tool execution —
    # single-shot chat has no Observations, so the model loops forever and never
    # emits Finish[] (val_acc=0). The react engine mirrors run_inference's toolqa
    # path. toolqa_data_dir is passed explicitly because react.py reads the
    # (nonexistent) vendor-default EXTERNAL_DIR, not lora.config.
    react_engine = None
    react_skills = None
    if dataset == "toolqa":
        react_engine = get_engine(
            "react", max_tokens=max_tokens, thinking=False,
            temperature=temperature,
            toolqa_data_dir=str(EXTERNAL_DIR / "toolqa"),
        )
        react_skills = [{"skill_id": "toolqa", "content": skill_content or ""}]

    def _one(inst):
        finish_reason = None
        error = None
        react_meta = {}
        try:
            if react_engine is not None:
                result = react_engine.run(
                    inst, react_skills, client, lora_name,
                    base_model=base_model_name,
                )
                raw = result.raw_output
                react_meta = result.meta
            else:
                system, user = build_prompt(inst, skills=skills)
                if use_tools:
                    raw, _ = run_with_tools(
                        client, lora_name, system, user, tools,
                        temperature=temperature, max_tokens=max_tokens, extra_body=extra,
                    )
                else:
                    messages = ([{"role": "system", "content": system}] if system else [])
                    messages.append({"role": "user", "content": user})
                    kwargs = {
                        "model": lora_name, "messages": messages,
                        "temperature": temperature, "max_tokens": max_tokens,
                    }
                    if extra:
                        kwargs["extra_body"] = extra
                    response = client.chat.completions.create(**kwargs)
                    choice = response.choices[0]
                    raw = choice.message.content or ""
                    finish_reason = choice.finish_reason or "stop"
        except Exception as e:  # noqa: BLE001 — one bad sample must not abort validation
            raw = f"[VAL_ERROR] {e}"
            error = f"{type(e).__name__}: {e}"
        # GT via sragents' standard answer extraction (symmetric with pred
        # extraction), not the separate validation.extracted_answer field. For
        # toolqa that's the Finish[] extractor applied to the TEACHER's trajectory
        # output; other datasets keep validation.extracted_answer (already canonical).
        gt = inst.get("validation", {}).get("extracted_answer", "")
        if dataset == "toolqa":
            from sragents.evaluate.datasets.toolqa import _extract as _toolqa_extract
            teacher_out = inst.get("trajectory", {}).get("output", "")
            extracted_gt = _toolqa_extract(strip_think_tags(teacher_out))
            if extracted_gt:
                gt = extracted_gt
        truncated = (
            finish_reason == "length"
            or bool(react_meta.get("truncated"))
            or bool(react_meta.get("halted"))
        )
        return {"instance_id": inst.get("instance_id"), "gt": gt,
                "correct": (not truncated and not error and compare_answer(
                    raw, gt, dataset, question=inst.get("question", "")
                )),
                "finish_reason": finish_reason, "truncated": truncated, "error": error,
                "stop_reason": react_meta.get("stop_reason"),
                "raw_head": (raw or "")[:160]}

    details = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, inst) for inst in val_instances]
        for f in as_completed(futs):
            details.append(f.result())
    acc = sum(1 for d in details if d["correct"]) / max(len(details), 1)
    return acc, details
def build_val_samples(val_instances, tokenizer, max_length, include_skill, skill_content):
    """Tokenize val instances' own training_samples (same masking as train).

    Used for the teacher-forced val/loss curve; reuses ``_build_masked_sample``
    so it is on the same scale as the training loss.
    """
    samples = []
    for inst in val_instances:
        tr = inst.get("trajectory", {})
        for ts in tr.get("training_samples", []):
            msgs = ts["messages"]
            if not include_skill:
                msgs = _strip_skill_from_messages(msgs, skill_content)
            s = _build_masked_sample(tokenizer, ts.get("system", ""), msgs, max_length)
            if s is not None:
                samples.append(s)
    return samples
def compute_val_loss(model, val_samples, collator, batch_size, device) -> float:
    """Teacher-forced masked CE over val samples (no grad).

    Mirrors ``LTKTrainer.compute_loss``'s shift/lossits_to_keep alignment. Uses
    reduction='sum' + token count for an exact per-token mean.
    """
    if not val_samples:
        return float("nan")
    was_training = model.training
    model.eval()
    # val-loss is a no-grad extra metric (best-epoch uses val-accuracy via the serve).
    # Cap its batch to 1: logits [1, ltk, 248K] always fits on the 248K vocab regardless
    # of ltk (a [16, ltk, 248K] val-loss batch OOM'd at ~30GB). This is
    # independent of the in-process generation batch.
    batch_size = 1
    total_loss, total_items = 0.0, 0
    try:
        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch = collator(val_samples[i:i + batch_size])
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                # logits_to_keep: only the trainable span (same fix as compute_loss).
                # Full [bs, seq, 248K] logits OOM on the 248K vocab at bs=4 long batches.
                ltk = int((labels != -100).sum(dim=-1).max().item()) + 2
                logits = model(**batch, logits_to_keep=max(ltk, 1)).logits  # [bs, ltk, vocab]
                padded = torch.cat([
                    labels,
                    torch.full((labels.size(0), 1), -100,
                               device=device, dtype=labels.dtype),
                ], dim=-1)
                shift = padded[:, 1:].contiguous()[:, -logits.size(1):]
                mask = shift != -100
                total_items += int(mask.sum().item())
                # bf16 CE (no float32 logits copy): with ltk uncapped (up to ~5914),
                # logits.float() would materialize a ~23GB fp32 tensor at bs=4.
                # PyTorch's cross_entropy accumulates logsumexp in fp32 internally,
                # so bf16 input is numerically equivalent without the copy.
                total_loss += float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shift.reshape(-1), ignore_index=-100, reduction="sum",
                ).item())
    finally:
        model.train(was_training)
    return total_loss / max(total_items, 1)
def _neutralize_for_generate(model):
    """Disable gradient_checkpointing so use_cache works for fast generation.

    We do NOT undo the logits_to_keep forward-patch: it is correct AND
    memory-efficient during generation. logits_to_keep=ltk keeps only the last
    ltk positions' logits (~[bs, ltk, vocab]); undoing it would compute a full
    [bs, seq, vocab] slice that OOMs on long prompts at val-batch=32. ltk>=1
    always includes the last position `generate` needs for next-token selection.
    """
    base = model.base_model.model
    try:
        base.gradient_checkpointing_disable()
    except Exception:  # noqa: BLE001
        pass
def _restore_after_generate(model, fwd_saved=None):
    base = model.base_model.model
    try:
        base.gradient_checkpointing_enable()
    except Exception:  # noqa: BLE001
        pass
def _inprocess_generate_validation(
    model, tokenizer, val_instances, skill_content, tools, *,
    dataset, val_lora_name, base_model_name, temperature, max_tokens,
    batch_size, disable_adapter,
):
    """Generate val answers on the live in-process PeftModel and score them.

    No-tool datasets use batched left-padded ``generate`` (fast); medcalc uses
    :class:`InProcessClient` + ``run_with_tools`` (serial — the tool loop can't
    batch). ``disable_adapter=True`` scores the base model (no-LoRA baseline).
    """
    state = _neutralize_for_generate(model)
    try:
        if tools:  # medcalc: BATCHED tool-loop across instances (parallel, not serial)
            skills = [skill_content] if skill_content else None
            inst_msgs = []
            for inst in val_instances:
                system, user = build_prompt(inst, skills=skills)
                m = ([{"role": "system", "content": system}] if system else []) \
                    + [{"role": "user", "content": user}]
                inst_msgs.append((inst.get("instance_id"), m))
            raw_map = batched_tool_validate(
                model, tokenizer, inst_msgs, tools, max_new_tokens=max_tokens,
                batch_size=max(1, batch_size), disable_adapter=disable_adapter,
            )
            details = []
            for inst in val_instances:
                iid = inst.get("instance_id")
                raw = raw_map.get(iid, "")
                gt = inst.get("validation", {}).get("extracted_answer", "")
                details.append({
                    "instance_id": iid, "gt": gt,
                    "correct": compare_answer(raw, gt, dataset,
                                              question=inst.get("question", "")),
                    "raw_head": (raw or "")[:160],
                })
            acc = sum(1 for d in details if d["correct"]) / max(len(details), 1)
        else:  # no-tool: batched generate
            skills = [skill_content] if skill_content else None
            msgs = []
            for inst in val_instances:
                system, user = build_prompt(inst, skills=skills)
                m = []
                if system:
                    m.append({"role": "system", "content": system})
                m.append({"role": "user", "content": user})
                msgs.append(m)
            raws = batched_generate(
                model, tokenizer, msgs, max_new_tokens=max_tokens,
                batch_size=max(1, batch_size), temperature=temperature,
                disable_adapter=disable_adapter,
            )
            details = []
            for inst, raw in zip(val_instances, raws):
                gt = inst.get("validation", {}).get("extracted_answer", "")
                details.append({
                    "instance_id": inst.get("instance_id"), "gt": gt,
                    "correct": compare_answer(raw, gt, dataset,
                                              question=inst.get("question", "")),
                    "raw_head": (raw or "")[:160],
                })
            acc = sum(1 for d in details if d["correct"]) / max(len(details), 1)
    finally:
        _restore_after_generate(model, state)
    return acc, details
