"""Dataset construction: load trajectories, build loss-masked samples, split train/val, export."""
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from msi.training.provenance import normalize_question


def load_training_data(trajectory_file: str) -> dict:
    with open(trajectory_file) as f:
        return json.load(f)
def _strip_skill_from_messages(messages: list[dict], skill_content: str) -> list[dict]:
    """Remove skill injection prefix from user messages.

    The prefix may not be at position 0 — ToolQA samples prepend few-shot
    examples before the skill text, so we locate and remove it by content.
    """
    if not skill_content:
        return messages
    prefix = f"Relevant Skill:\n{skill_content}\n\n"
    result = []
    for msg in messages:
        if msg["role"] == "user" and prefix in msg["content"]:
            result.append({**msg, "content": msg["content"].replace(prefix, "", 1)})
        else:
            result.append(msg)
    return result
def build_anchor_samples(anchor_file: str, tokenizer: AutoTokenizer, max_length: int = 16384,
                         skill_content: str = "", include_skill: bool = True,
                         near_count: int | None = None,
                         random_count: int | None = None):
    """Load ANCHOR negative samples and build loss-masked TRAIN-ONLY samples.

    Each ANCHOR sample is a W-injected prompt (the retrieved skill W is in the
    user message, matching retrieved_lora inference) whose target is the base-
    by-hand answer produced by the anchor pipeline. Reuses
    ``_build_masked_sample`` directly.

    ``include_skill`` (default True, the with-text regime): W stays in the prompt
    so the adapter learns to "see through" the wrong skill and revert to base.
    Under config-2 (``include_skill=False``), W is STRIPPED so the ANCHOR prompt
    matches the no-text APPLY prompt (non-fitting question, no W → base output) —
    otherwise APPLY(no W) vs ANCHOR(with W) is an inconsistent discrimination.

    Returns ``(samples, n_skipped)``.
    """
    data = json.load(open(anchor_file))
    raw = data["samples"] if isinstance(data, dict) and "samples" in data else data
    if near_count is not None or random_count is not None:
        if not isinstance(data, dict):
            raise SystemExit("anchor count selection requires the release anchor schema")
        available_near = int(data.get("n_nearmiss_kept", -1))
        available_random = int(data.get("n_random_kept", -1))
        selected_near = available_near if near_count is None else near_count
        selected_random = available_random if random_count is None else random_count
        if (selected_near < 0 or selected_random < 0
                or selected_near > available_near
                or selected_random > available_random):
            raise SystemExit(
                "requested anchor subset is unavailable: "
                f"near={selected_near}/{available_near}, "
                f"random={selected_random}/{available_random}; "
                "finish anchor construction or explicitly change the training quota"
            )
        # assemble.py publishes near rows first, followed by random rows.
        raw = (
            raw[:selected_near]
            + raw[available_near:available_near + selected_random]
        )
    samples, n_skipped = [], 0
    for s in raw:
        msgs = s.get("messages", [])
        if not include_skill:
            msgs = _strip_skill_from_messages(msgs, skill_content)
        ms = _build_masked_sample(tokenizer, s.get("system", ""), msgs, max_length)
        if ms is None:
            n_skipped += 1
            continue
        samples.append(ms)
    return samples, n_skipped
_NO_LOSS_ROLES = {"user", "system"}
def _build_masked_sample(
    tokenizer: AutoTokenizer,
    system: str,
    messages: list[dict],
    max_length: int,
) -> dict | None:
    """Tokenize a multi-turn conversation with loss masking.

    Loss is computed only on the **last** assistant turn — all other content
    (system, user, earlier assistant turns) is masked with -100. This matches
    the per-turn training paradigm where each sample represents one API call
    and only the latest model response should be learned.
    """
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    chat.extend(messages)

    rendered = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)

    if not input_ids:
        return None

    labels = [-100] * len(input_ids)

    # Find the last assistant turn
    last_asst_idx = max(
        (i for i, msg in enumerate(chat) if msg["role"] == "assistant"),
        default=None,
    )
    if last_asst_idx is None:
        return None

    # Boundary: everything before the last assistant turn
    prefix = chat[:last_asst_idx]
    if prefix:
        prefix_rendered = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        boundary = len(tokenizer.encode(prefix_rendered, add_special_tokens=False))
    else:
        boundary = 0

    # End: up to and including the last assistant turn
    full_rendered = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    end = min(len(tokenizer.encode(full_rendered, add_special_tokens=False)),
              len(input_ids))

    for j in range(boundary, end):
        labels[j] = input_ids[j]

    # NO silent truncation: a sample longer than max_length would lose its tail
    # (the end of the assistant response = the answer/TOOL_CALL) if right-
    # truncated. Skip it instead and let the caller count/log skips, so we
    # never train on a sample whose prefix OR suffix was dropped. The caller
    # (prepare_dataset) sets max_length high enough (default 16384) that this
    # rarely/never triggers on real data.
    if len(input_ids) > max_length:
        return None
    seq_len = len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * seq_len,
        "labels": labels,
    }
def _verify_loss_masking(labels: list[int]) -> None:
    """Log loss masking statistics to catch potential bugs."""
    total = len(labels)
    trainable = sum(1 for l in labels if l != -100)
    pct = trainable / total * 100 if total > 0 else 0
    if trainable == 0:
        print(f"  WARNING: No trainable tokens in sample (all masked)")
    else:
        print(f"  Loss masking: {trainable}/{total} tokens ({pct:.1f}% trainable)")
def prepare_dataset(
    data: dict, tokenizer: AutoTokenizer, max_length: int = 16384,
    include_skill: bool = True,
) -> Dataset:
    """Prepare training dataset from trajectory data.

    The release schema requires ``training_samples`` for every trajectory.
    """
    skill_content = data.get("skill_content", "")
    skill_id = data.get("skill_id", "")
    samples = []
    total_trainable = 0
    total_tokens = 0
    n_attempted = 0
    n_skipped = 0

    def _add(system: str, messages: list[dict]):
        nonlocal total_trainable, total_tokens, n_attempted, n_skipped
        n_attempted += 1
        sample = _build_masked_sample(tokenizer, system, messages, max_length)
        if sample is None:
            n_skipped += 1
            return
        samples.append(sample)
        total_trainable += sum(1 for l in sample["labels"] if l != -100)
        total_tokens += len(sample["labels"])

    for instance in data.get("instances", []):
        trajectory = instance.get("trajectory", {})

        training_samples = trajectory.get("training_samples")
        if not isinstance(training_samples, list) or not training_samples:
            raise SystemExit(
                "release trajectories require non-empty "
                f"trajectory.training_samples ({skill_id})"
            )
        for ts in training_samples:
            msgs = ts["messages"]
            if not include_skill:
                msgs = _strip_skill_from_messages(msgs, skill_content)
            _add(ts.get("system", ""), msgs)

    print(f"Prepared {len(samples)} training samples")
    if n_skipped > 0:
        raise SystemExit(
            f"{n_skipped}/{n_attempted} training samples exceed "
            f"max_length={max_length}. The release run never silently drops long "
            "trajectory turns; top up the admitted source or raise --max-length."
        )
    if total_tokens > 0:
        print(f"  Loss masking: {total_trainable}/{total_tokens} tokens "
              f"({total_trainable/total_tokens*100:.1f}% trainable)")

    max_trainable = max(
        (sum(1 for l in s["labels"] if l != -100) for s in samples),
        default=1,
    )
    return Dataset.from_list(samples), max_trainable
def export_sharegpt(data: dict, output_path: Path, include_skill: bool = True) -> None:
    """Export training data in sharegpt/conversation format for vLLM compatibility.

    Format: one JSON line per sample with {"messages": [...]} structure.
    Uses ``training_samples`` (per-turn) when available.
    """
    skill_content = data.get("skill_content", "")
    count = 0

    with open(output_path, "w") as f:
        for instance in data.get("instances", []):
            trajectory = instance.get("trajectory", {})

            training_samples = trajectory.get("training_samples")
            if not isinstance(training_samples, list) or not training_samples:
                raise SystemExit(
                    "release trajectories require non-empty "
                    "trajectory.training_samples"
                )
            for ts in training_samples:
                msgs = ts["messages"]
                if not include_skill:
                    msgs = _strip_skill_from_messages(msgs, skill_content)
                messages = []
                if ts.get("system"):
                    messages.append({"role": "system", "content": ts["system"]})
                messages.extend(msgs)
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Exported {count} samples in sharegpt format to {output_path}")
def save_deployment_config(
    output_path: Path, base_model: str, skill_id: str, dataset: str, rank: int, epochs: int,
) -> None:
    """Save deployment_config.json with vLLM serve commands."""
    adapter_path = str(output_path)

    config = {
        "schema": "msi.adapter-deployment",
        "base_model": base_model,
        "adapter_path": adapter_path,
        "lora_name": skill_id,
        "skill_id": skill_id,
        "dataset": dataset,
        "rank": rank,
        "epochs": epochs,
        "note": (
            "Use the immutable resolved config plus `msi evaluate`. "
            "The matching notext/withtext adapter pool determines the method."
        ),
    }
    with open(output_path / "deployment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Deployment config saved to {output_path / 'deployment_config.json'}")
def split_train_val(
    data: dict, n_train: int = 150, n_val: int = 500, seed: int = 42,
    fix_val: bool = False, fixed_val_offset: int | None = None,
) -> list[dict]:
    """Partition acceptable instances into train + validate.

    Keeps only instances with ``validation.acceptable == True`` and a non-empty
    ``extracted_answer``. Seeded shuffle; the first ``n_train`` become the train
    set (mutates ``data["instances"]`` so ``prepare_dataset`` sees only them),
    the next ``n_val`` are returned as the validate list (full instance dicts).

    With ``fix_val=True`` the training set remains the nested prefix and the
    validation set starts at ``fixed_val_offset`` after the seeded shuffle.
    This lets a size sweep reuse the main run's validation slice (for example,
    offset 400 gives ``[400:480]``) without changing any training prefixes.
    Default uses train-first ordering.
    """
    acceptable = [
        i for i in data.get("instances", [])
        if i.get("validation", {}).get("acceptable")
        and str(i.get("validation", {}).get("extracted_answer", "")).strip()
    ]
    normalized = [normalize_question(i.get("question", "")) for i in acceptable]
    duplicates = [q for q, count in __import__("collections").Counter(normalized).items()
                  if q and count > 1]
    if duplicates:
        raise SystemExit(
            f"{len(duplicates)} normalized questions are duplicated before "
            "train/val split; regenerate and re-audit the trajectory"
        )
    random.Random(seed).shuffle(acceptable)
    val_start = (
        int(fixed_val_offset) if fix_val and fixed_val_offset is not None
        else n_train
    )
    if val_start < n_train:
        raise SystemExit(
            f"fixed validation offset {val_start} overlaps the {n_train} training rows"
        )
    if len(acceptable) < max(n_train, val_start + n_val):
        raise SystemExit(
            f"only {len(acceptable)} acceptable instances; requested {n_train} "
            f"train and validation [{val_start}:{val_start + n_val}]; "
            "generate more trajectories or explicitly reduce the split counts"
        )
    if fix_val:
        train = acceptable[:n_train]
        val = acceptable[val_start:val_start + n_val]
    else:
        train = acceptable[:n_train]
        val = acceptable[n_train:n_train + n_val]
    data["instances"] = train
    print(f"Split: {len(train)} train / {len(val)} val (of {len(acceptable)} acceptable)"
          f"{' [fix_val offset=' + str(val_start) + ']' if fix_val else ''}")
    return val
