"""Trajectory validation: benchmark evaluator + LLM judge."""
from openai import OpenAI
import json

from sragents.evaluate import evaluate
from sragents.llm import get_extra_body
from msi.models.llm_client import _chat
from msi.generation.prompts import (
    _DATASET_FORMAT_CONTEXT,
    _DATASET_TOOL_CONTEXT,
    _EVALUATOR_SYSTEM,
    _EVALUATOR_USER,
)


def validate_trajectory_format(trajectory_output: str, instance: dict) -> dict:
    """Validate and extract an answer without calling an LLM judge.

    This is the explicit ``--no-evaluator`` mode. It checks only that the
    dataset's normal answer extractor can recover a non-empty answer; it does
    not claim that the synthetic solution is correct.
    """
    from sragents.evaluate.common import strip_think_tags

    dataset = str(instance.get("dataset", ""))
    output = strip_think_tags(trajectory_output or "").strip()
    try:
        if dataset == "theoremqa":
            from sragents.evaluate.datasets.theoremqa import _extract
            extracted = _extract(output)
        elif dataset == "logicbench":
            from sragents.evaluate.datasets.logicbench import _extract
            question = str(instance.get("question", ""))
            task_type = "MCQA" if "choice_1" in question.casefold() else "BQA"
            extracted = _extract(output, {
                **instance, "eval_data": {"task_type": task_type},
            })
        elif dataset == "medcalcbench":
            from sragents.evaluate.datasets.medcalcbench import _extract
            extracted = _extract(output, {})
        elif dataset == "toolqa":
            from sragents.evaluate.datasets.toolqa import _extract
            extracted = _extract(output)
        else:
            extracted = ""
    except Exception as error:
        return {
            "acceptable": False, "valid": False, "extracted_answer": "",
            "reason": f"format-only extraction failed: {error}",
            "validator": "format_only",
        }
    extracted = str(extracted or "").strip()
    acceptable = bool(output and extracted)
    return {
        "acceptable": acceptable,
        "valid": acceptable,
        "extracted_answer": extracted,
        "reason": (
            "format-only validation; LLM evaluator disabled"
            if acceptable else "format-only validation found no answer"
        ),
        "validator": "format_only",
    }


def _parse_json_from_content(content: str) -> dict | list | None:
    """Extract and parse JSON from LLM output."""
    # Fast path: check for code blocks first
    if "```json" in content:
        content = content.split("```json")[-1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        content = parts[-2].strip() if len(parts) >= 3 else parts[1].strip()

    # Try direct parse (handles arrays `[...]` efficiently)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Slow path: brace-balanced extraction of first JSON object
    import re
    m = re.search(r'\{\s*"question"', content)
    if m:
        start = m.start()
        depth = 0
        for end in range(start, len(content)):
            if content[end] == '{':
                depth += 1
            elif content[end] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start:end + 1])
                    except json.JSONDecodeError:
                        break

    return None
def validate_trajectory(trajectory_output: str, instance: dict) -> dict:
    try:
        result = evaluate(trajectory_output, instance)
        return {
            "valid": result.get("correct", False),
            "extracted_answer": result.get("extracted_answer", ""),
            "correct": result.get("correct", False),
            "details": result,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
# The judge returns structured decisions; hidden reasoning is disabled to keep
# validation latency and output format predictable.
EVALUATOR_ENABLE_THINKING = False


def llm_validate_trajectory(
    client: OpenAI, model: str,
    trajectory_output: str, instance: dict,
    skill_content: str, dataset: str,
) -> dict:
    """Evaluate trajectory quality using an LLM judge.

    Returns {"acceptable": bool, "reason": str, "extracted_answer": str}.
    """
    format_context = _DATASET_FORMAT_CONTEXT.get(dataset, f"Dataset: {dataset}.")
    tool_context = _DATASET_TOOL_CONTEXT.get(dataset, "")
    system = _EVALUATOR_SYSTEM.format(format_context=format_context, tool_context=tool_context)
    user = _EVALUATOR_USER.format(
        dataset=dataset,
        question=instance.get("question", ""),
        skill_content=skill_content,
        trajectory_output=trajectory_output,
    )

    try:
        # Per-model thinking control (same helper as validate.py / trajectory.py):
        # get_extra_body("Qwen3.5-122B-A10B", ...) → chat_template_kwargs.enable_thinking
        # — the only key vLLM honors (top-level enable_thinking is silently ignored).
        content = _chat(client, model, system, [{"role": "user", "content": user}],
                        max_tokens=16384, temperature=0.0,
                        extra_body=get_extra_body(model, thinking=EVALUATOR_ENABLE_THINKING))
        parsed = _parse_json_from_content(content)
        if isinstance(parsed, dict) and "acceptable" in parsed:
            acceptable = bool(parsed.get("acceptable", False))
            extracted = str(parsed.get("extracted_answer", "")).strip()
            reason = str(parsed.get("reason", ""))
            if acceptable and not extracted:
                # The judge occasionally omits only the redundant extraction
                # field even though the trajectory ends in the exact dataset
                # format.  Recover it with the benchmark's deterministic
                # extractor; quality acceptance still comes exclusively from
                # the enabled LLM judge.
                fallback = validate_trajectory_format(trajectory_output, instance)
                if fallback.get("acceptable"):
                    extracted = str(fallback.get("extracted_answer", "")).strip()
                    reason = (reason + "; answer recovered by dataset extractor").lstrip("; ")
            return {
                "acceptable": acceptable,
                "reason": reason,
                "extracted_answer": extracted,
            }
        # Fallback: try to extract boolean from raw text
        acceptable = "acceptable" in content.lower() and "true" in content.lower()
        return {
            "acceptable": acceptable,
            "reason": content[:200],
            "extracted_answer": "",
        }
    except Exception as e:
        return {"acceptable": False, "reason": f"evaluator error: {e}", "extracted_answer": ""}
