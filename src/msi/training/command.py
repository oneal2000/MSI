#!/usr/bin/env python3
"""Train and validate one skill-specific LoRA adapter."""
import argparse
import json
import os
# Reduce CUDA fragmentation so large-ltk batches don't OOM; must precede any CUDA alloc.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import shutil
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from datasets import Dataset, concatenate_datasets

from msi.models.config import CORPUS_PATH, RESULTS_DIR
from msi.models.qwen_thinking import apply_for as apply_qwen_thinking
from msi.training.dataset import (
    build_anchor_samples,
    export_sharegpt,
    load_training_data,
    prepare_dataset,
    save_deployment_config,
    split_train_val,
)
from msi.training.trainer import LTKTrainer, ValidationCallback
from msi.training.provenance import assert_shared_output, file_sha256
from msi.training.validate import (
    _inprocess_generate_validation,
    build_val_samples,
    load_skill_tools,
    run_validation,
)


# LoRA target-module presets for ablation on the HYBRID Qwen3.5 architecture.
#   full_attention  (6 layers, softmax GQA):  self_attn.{q,k,v,o}_proj
#   linear_attention(18 layers, Mamba2 SSM):  linear_attn.{in_proj_a,_b,_qkv,_z,out_proj}
#   mlp             (24 layers, every block): mlp.{gate,up,down}_proj
# ``standard`` targets full-attention and MLP projections. SSM projections are
# available as an explicit ablation.
TARGET_GROUPS = {
    "standard":  ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
    "full_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ssm":       ["in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"],
    "mlp":       ["gate_proj", "up_proj", "down_proj"],
    "all_attn":  ["q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"],
    "all":       ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"],
}
TARGET_MODULES_DEFAULT = TARGET_GROUPS["standard"]


def train_lora(
    base_model: str,
    skill_id: str,
    trajectory_file: str,
    output_dir: str = None,
    rank: int = 8,
    epochs: int = 2,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    max_length: int = 16384,
    include_skill: bool = True,
    use_quantization: bool = False,
    lora_dropout: float = 0.05,
    max_samples: int | None = None,
    target_modules: list[str] | None = None,
    # ---- per-epoch validation + TensorBoard ----
    num_train: int | None = 150,
    num_val: int | None = 500,
    split_seed: int = 42,
    val_backend: str = "none",          # {"none", "vllm", "inprocess"}
    val_api_base: str | None = None,
    val_lora_name_base: str | None = None,
    vl_rename: bool = False,            # Qwen3.5: rename adapter keys to VL nested-LM path on register
    text_only_load: bool = False,       # Gemma4: build text *ForCausalLM from multimodal ckpt (avoid ClippableLinear)
    val_temperature: float = 0.0,
    val_max_tokens: int = 16384,
    val_workers: int = 256,
    base_model_name: str | None = None,
    tb_log: bool = True,
    tb_dir: str | None = None,  # TB log dir (default output_path/tb); set to distinguish runs by purpose
    val_async: bool = False,
    val_concurrency: int = 2,
    val_freq: int = 1,  # gen-val cadence (epochs) for all datasets
    compute_val_loss: bool = True,
    val_fix_val: bool = False,  # fix val set across train sizes (sweeps)
    fixed_val_offset: int | None = None,
    skip_base_baseline: bool = False,  # skip pre-training base (no-LoRA) eval
    anchor_file: str | None = None,  # ANCHOR negatives (retrieve-miss robustness); train-only
    anchor_near_count: int | None = None,
    anchor_random_count: int | None = None,
    training_job_id: str | None = None,
    export_chat: bool = False,
):
    """Train skill-specific LoRA adapter with vLLM-compatible output."""
    if not training_job_id:
        raise SystemExit("--training-job-id is mandatory")
    anchor_sha256 = file_sha256(anchor_file) if anchor_file else None
    n_anchor_samples = 0
    n_near_anchor_samples = 0
    n_random_anchor_samples = 0
    if not anchor_file and any(
        value not in (None, 0)
        for value in (anchor_near_count, anchor_random_count)
    ):
        raise SystemExit("anchor counts require --anchor-file")
    if anchor_file:
        with open(anchor_file, encoding="utf-8") as handle:
            anchor_payload = json.load(handle)
        anchor_meta = anchor_payload.get("provenance", {})
        expected_model = base_model_name or Path(base_model).name
        if anchor_meta.get("model") != expected_model:
            raise SystemExit(
                f"anchor targets were generated by {anchor_meta.get('model')!r}, "
                f"not the training base model {expected_model!r}"
            )
        if anchor_payload.get("skill_id") != skill_id:
            raise SystemExit(
                f"anchor skill_id={anchor_payload.get('skill_id')!r} "
                f"does not match training skill {skill_id!r}"
            )
        for kind in ("nearmiss", "random"):
            kept = int(anchor_payload.get(f"n_{kind}_kept", -1))
            requested = int(anchor_payload.get(f"n_{kind}_requested", -1))
            if kept < 0 or requested < 0 or kept != requested:
                raise SystemExit(
                    f"incomplete {kind} anchors for {skill_id}: "
                    f"kept={kept}, requested={requested}"
                )
    if not output_dir:
        raise SystemExit("--output-dir is mandatory in the release pipeline")
    assert_shared_output(output_dir)
    data = load_training_data(trajectory_file)
    dataset_name = data.get("dataset", skill_id.split("_")[0])
    skill_content = data.get("skill_content", "")
    # When training without skill text (--strip-skill), the generation val must
    # ALSO be no-text to match the golden_lora eval condition (else best-epoch is
    # selected on a with-text prompt the adapter will never see at golden_lora).
    val_skill_content = "" if not include_skill else skill_content
    n_available = len(data.get("instances", []))
    if max_samples is not None and max_samples < n_available:
        data["instances"] = data["instances"][:max_samples]
        print(f"Capping to first {max_samples} instances (from {n_available} available)")

    # Held-out train/val split (mutates data["instances"] -> train slice) so
    # prepare_dataset trains on only ~num_train samples and we can validate on
    # a disjoint ~num_val set every epoch.
    val_instances: list[dict] = []
    tools: list[dict] = []
    if val_backend != "none":
        val_instances = split_train_val(
            data, n_train=num_train or 0, n_val=num_val or 0, seed=split_seed,
            fix_val=val_fix_val, fixed_val_offset=fixed_val_offset,
        )
        tools = load_skill_tools(skill_id)
    n_instances = len(data.get("instances", []))
    if val_backend != "none":
        expected_train = int(num_train or 0)
        expected_val = int(num_val or 0)
        if n_instances != expected_train or len(val_instances) != expected_val:
            raise SystemExit(
                f"{skill_id} cannot provide the fixed split: "
                f"train={n_instances}/{expected_train}, val={len(val_instances)}/{expected_val}. "
                "Top up with admitted trajectories and re-audit before loading the model."
            )
    print(f"Skill: {skill_id}, Dataset: {dataset_name}, Instances: {n_instances}")

    print(f"Loading base model: {base_model}")
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    if use_quantization:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if text_only_load:
        # Gemma4 etc.: build the text *ForCausalLM from the multimodal checkpoint
        # (remaps nested model.language_model.* keys), avoiding the ClippableLinear
        # vision/audio towers that are not valid PEFT targets.
        from msi.models.text_only_load import load_text_causal_lm
        model = load_text_causal_lm(base_model, dtype=torch.bfloat16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the loss-target region (last assistant turn) is always at the
    # sequence end. This is REQUIRED for the logits_to_keep loss window
    # (shift_labels[:, -ltk:]) to cover the full trainable region on every
    # sample regardless of batch-internal length variance. Safe for this hybrid
    # Mamba2 model: modeling_qwen3_5.apply_mask_to_padding_states zeroes padding
    # at the start of each SSM layer, and position_ids come from
    # attention_mask.cumsum (both left-pad correct).
    tokenizer.padding_side = "left"

    # The CLI normally supplies architecture-detected targets. The standard
    # transformer projection set is the programmatic default.
    resolved_targets = target_modules or TARGET_MODULES_DEFAULT
    print(f"LoRA target_modules: {resolved_targets}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=lora_dropout,
        target_modules=resolved_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Prepare dataset and compute max trainable tokens for logits_to_keep.
    # ltk = max_trainable + 1 (NOT capped): every trainable token of every
    # sample's last assistant turn must land inside the loss window. Capping
    # would silently drop the head of long responses (e.g. a 5913-token
    # multi-step calculation). Combined with left-padding, the [-ltk:] window
    # always covers the full trainable region. Memory is bounded by bf16 CE
    # (no float32 logits copy); the worst case (ltk≈5914) is validated by prior
    # training runs that used the heavier float32 path.
    dataset, max_trainable = prepare_dataset(data, tokenizer, max_length, include_skill)
    # FAIL-FAST: 0 train samples means split_train_val found 0 acceptable instances —
    # almost always a validation-field problem (e.g. trajectories regenerated with
    # --skip-validation lack validation.acceptable). Abort loudly instead of training
    # on nothing and silently producing base-only downstream results.
    if len(dataset) == 0:
        n_acc = sum(1 for i in data.get("instances", [])
                    if i.get("validation", {}).get("acceptable"))
        n_total = len(data.get("instances", []))
        raise SystemExit(
            f"[FAIL-FAST] 0 training samples (acceptable={n_acc}/{n_total} instances). "
            f"Likely cause: trajectory file lacks validation.acceptable/extracted_answer "
            f"(regen with --skip-validation?). Run the evaluator (llm_validate_trajectory) "
            f"on the trajectory file, or check split_train_val's acceptable filter."
        )
    # ANCHOR negatives: W-injected prompts whose target is the base-by-hand answer
    # (retrieve-miss robustness). Train-ONLY — appended to the train Dataset, never
    # the val split (which stays APPLY-only and comparable across arms). Built via
    # build_anchor_samples -> _build_masked_sample; regime-specific skill
    # stripping matches the corresponding APPLY and inference condition.
    n_positive_samples = len(dataset)
    if anchor_file:
        available_near = int(anchor_payload["n_nearmiss_kept"])
        available_random = int(anchor_payload["n_random_kept"])
        n_near_anchor_samples = (
            available_near if anchor_near_count is None else anchor_near_count
        )
        n_random_anchor_samples = (
            available_random if anchor_random_count is None else anchor_random_count
        )
        anchor_samples, n_anchor_skip = build_anchor_samples(
            anchor_file, tokenizer, max_length,
            skill_content=skill_content, include_skill=include_skill,
            near_count=n_near_anchor_samples,
            random_count=n_random_anchor_samples,
        )
        if n_anchor_skip:
            raise SystemExit(
                f"{n_anchor_skip} anchor samples exceed "
                f"max_length={max_length}; no requested negative may be silently dropped"
            )
        if anchor_samples:
            n_anchor_samples = len(anchor_samples)
            anchor_max = max(
                (sum(1 for l in s["labels"] if l != -100) for s in anchor_samples), default=0,
            )
            max_trainable = max(max_trainable, anchor_max)
            dataset = concatenate_datasets([dataset, Dataset.from_list(anchor_samples)])
            print(f"  ANCHOR: added {len(anchor_samples)} negative samples to train "
                  f"(skipped {n_anchor_skip} over max_length); max_trainable={max_trainable}")
        else:
            print(f"  ANCHOR: WARNING no anchor samples loaded from {anchor_file}")
    ltk_value = max_trainable + 1  # +1 for label shift in loss
    _base_fwd = model.base_model.model.forward
    model.base_model.model.forward = partial(_base_fwd, logits_to_keep=ltk_value)
    # Stash the original forward so in-process validation generation can undo the
    # logits_to_keep windowing (see _neutralize_for_generate).
    model._ltk_base_fwd = _base_fwd
    print(f"  logits_to_keep={ltk_value} (max trainable tokens per sample: {max_trainable})")

    # Tokenize val instances (for the teacher-forced val/loss curve). Same
    # masking as train via build_val_samples -> _build_masked_sample.
    val_samples = (build_val_samples(val_instances, tokenizer, max_length,
                                     include_skill, skill_content)
                   if val_backend != "none" and val_instances else [])

    # Output path
    if output_dir:
        output_path = Path(output_dir) / skill_id
    else:
        model_name = base_model.split("/")[-1]
        output_path = RESULTS_DIR / "lora" / model_name / skill_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Training
    # logging_steps=1 + grad_accum=1 ⇒ one TensorBoard point per batch.
    # logits_to_keep avoids materializing full [bs, seq_len, vocab] logits.
    #
    # NOTE: transformers 5.x TensorBoardCallback reads its log dir from the
    # TENSORBOARD_LOGGING_DIR env var, NOT args.logging_dir (deprecated). Point
    # it at our tb/ dir so the Trainer's per-step train/loss merges with the
    # callback's train/loss_epoch, val/loss, val/accuracy in one event dir.
    if tb_log:
        tb_dir = Path(tb_dir) if tb_dir else output_path / "tb"
        # Clear stale event files: TensorBoard merges ALL event files in one
        # directory into a single run, so re-training (which overwrites the
        # adapter at output_path, matching semantics here) would otherwise pile
        # multiple runs' steps onto the same axes → garbled/overlapping curves.
        if tb_dir.exists():
            for old in tb_dir.glob("events.out.tfevents.*"):
                old.unlink()
            for pf in tb_dir.glob("plugin*"):
                if pf.is_dir():
                    shutil.rmtree(pf)
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to=["tensorboard"] if tb_log else "none",
        logging_dir=str(tb_dir) if tb_log else None,
        seed=split_seed,
        # NOTE: transformers 5.x TrainingArguments has no group_by_length /
        # dataloader_shuffle arg (random batches). ltk is uncapped, so a very
        # long response (e.g. 5913 tok) can land in a batch with short samples;
        # memory stays feasible because (a) bf16 CE no longer materializes a
        # float32 logits copy. Per-skill batch overrides remain available for
        # unusually long responses.
    )

    # Validation callback + dedicated SummaryWriter (same logdir as the Trainer,
    # so custom scalars merge with the Trainer's per-step train/loss).
    callbacks = []
    tb_writer = None
    if tb_log:
        tb_writer = SummaryWriter(str(tb_dir))
    if (val_backend != "none" and val_instances
            and (val_api_base or val_backend == "inprocess")):
        callbacks.append(ValidationCallback(
            tb_writer=tb_writer,
            val_instances=val_instances,
            val_samples=val_samples,
            collator=collator,
            skill_content=val_skill_content,
            tools=tools,
            dataset=dataset_name,
            base_model_name=base_model_name or base_model.split("/")[-1],
            api_base=val_api_base,
            lora_name=val_lora_name_base or f"val_{skill_id}",
            output_path=output_path,
            model=model,
            device=next(model.parameters()).device,
            inprocess_batch_size=batch_size,
            temperature=val_temperature,
            max_tokens=val_max_tokens,
            workers=val_workers,
            val_async=val_async,
            val_concurrency=val_concurrency,
            tokenizer=tokenizer,
            val_backend=val_backend,
            val_freq=val_freq,
            compute_val_loss_flag=compute_val_loss,
            root=str(Path(__file__).resolve().parent.parent),
            base_model_path=base_model,
            skill_id=skill_id,
            vl_rename=vl_rename,
        ))
    else:
        print("Validation disabled (val_backend=none or no val_api_base/inprocess)")

    # Base (no-LoRA) baseline: base+skill accuracy before any training, so the
    # LoRA's marginal contribution (useful-if-trained vs useless-if-untrained) can be measured.
    base_acc = None
    baseline_future = None  # vllm path runs async (overlaps epoch-1 training)
    _baseline_pool = None
    if (val_backend != "none" and val_instances
            and (val_api_base or val_backend == "inprocess")
            and not skip_base_baseline):
        bname = base_model_name or base_model.split("/")[-1]
        if val_backend == "inprocess":
            # Sync: uses the live model with adapter disabled — can't overlap
            # training safely (the model is being updated). Blocks epoch-1 start.
            print("Running base (no-LoRA) baseline eval on val set...", flush=True)
            try:
                base_acc, _ = _inprocess_generate_validation(
                    model, tokenizer, val_instances, val_skill_content, tools,
                    dataset=dataset_name, val_lora_name="__base__",
                    base_model_name=bname, temperature=val_temperature,
                    max_tokens=val_max_tokens, batch_size=batch_size,
                    disable_adapter=True,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("val/base_accuracy", base_acc, 0)
                json.dump({"base_accuracy": base_acc, "n": len(val_instances)},
                          open(output_path / "base_accuracy.json", "w"), indent=2)
                print(f"  base+skill accuracy = {base_acc:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  base eval failed: {e}", flush=True)
        else:
            # vLLM path: runs on the server (model=base, no train GPU needed) —
            # submit ASYNC so it overlaps epoch-1 training instead of idling the
            # GPU. Result collected after trainer.train() below.
            import concurrent.futures
            _baseline_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="baseval")
            baseline_future = _baseline_pool.submit(
                run_validation, val_instances, val_skill_content, tools,
                api_base=val_api_base, lora_name=bname, dataset=dataset_name,
                base_model_name=bname, temperature=val_temperature,
                max_tokens=val_max_tokens, workers=val_workers,
            )
            print("base-baseline submitted (async; overlaps training)", flush=True)

    trainer = LTKTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()

    # Collect async base-baseline (vllm) result — submitted before train() so it
    # overlapped epoch-1 instead of idling the GPU.
    if baseline_future is not None:
        try:
            base_acc, _ = baseline_future.result(timeout=3600)
            if tb_writer is not None:
                tb_writer.add_scalar("val/base_accuracy", base_acc, 0)
            json.dump({"base_accuracy": base_acc, "n": len(val_instances)},
                      open(output_path / "base_accuracy.json", "w"), indent=2)
            print(f"  base+skill accuracy (async) = {base_acc:.4f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  base eval (async) failed: {e}", flush=True)
        _baseline_pool.shutdown(wait=False)

    if tb_writer is not None:
        tb_writer.close()

    # Save adapter
    model.save_pretrained(str(output_path))

    # Deploy BEST adapter (not last epoch) — fixes the last!=best deployment bug.
    # best/ holds the highest-val_acc epoch's adapter; the root adapter_model
    # .safetensors is what _discover_lora_adapter deploys, so overwrite it.
    val_cb = callbacks[0] if callbacks else None
    if val_backend != "none" and (
        val_cb is None or getattr(val_cb, "best_epoch", -1) < 0
    ):
        shutil.rmtree(output_path / "_val_adapters", ignore_errors=True)
        raise SystemExit(
            f"no successfully scored epoch for {skill_id}; "
            "the adapter is not complete and TRAINING_DONE will not be written"
        )
    if val_cb is not None and getattr(val_cb, "best_epoch", -1) >= 0:
        best_dir = output_path / "best"
        for fname in ("adapter_model.safetensors", "adapter_config.json"):
            src = best_dir / fname
            if src.exists():
                shutil.copy(src, output_path / fname)
                src.unlink()  # root now contains the deployed best; avoid a duplicate adapter
        shutil.rmtree(best_dir, ignore_errors=True)
        print(f"Deployed BEST adapter (epoch {val_cb.best_epoch}, "
              f"val_acc={val_cb.best_acc:.4f}) over last-epoch at {output_path}")
    else:
        print(f"No best-epoch validation ran; LAST-epoch adapter remains at "
              f"{output_path}")

    # KEEP_EPOCH_WEIGHTS="4,8" (env-gated, default off): persist the EXACT
    # per-epoch staging weights before the cleanup below, so post-hoc epoch
    # policy comparisons never need a retrain. Production runs without the
    # env keep the original delete-on-finish behavior unchanged.
    _keep_env = os.environ.get("KEEP_EPOCH_WEIGHTS", "").strip()
    if _keep_env:
        for _ep in [e.strip() for e in _keep_env.split(",") if e.strip().isdigit()]:
            _src = output_path / "_val_adapters" / f"e{_ep}"
            if _src.exists():
                _dst = output_path / "epoch_weights" / f"e{_ep}"
                shutil.copytree(_src, _dst, dirs_exist_ok=True)
                print(f"Kept epoch {_ep} weights at {_dst}", flush=True)
            else:
                print(f"KEEP_EPOCH_WEIGHTS: epoch {_ep} staging missing "
                      f"for {skill_id}", flush=True)

    # Epoch adapters are temporary inputs to validation and best-epoch selection.
    shutil.rmtree(output_path / "_val_adapters", ignore_errors=True)

    validated_epochs = val_cb.validated_epochs() if val_cb is not None else []
    if anchor_file and file_sha256(anchor_file) != anchor_sha256:
        raise SystemExit(
            f"anchor changed while training {skill_id}; refusing to publish completion"
        )

    # Save metadata
    metadata = {
        "skill_id": skill_id,
        "base_model": base_model,
        "rank": rank,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "include_skill": include_skill,
        "target_modules": resolved_targets,
        "training_job_id": training_job_id,
        "training_inputs": {
            "trajectory": str(Path(trajectory_file).resolve()),
            "anchor": str(Path(anchor_file).resolve()) if anchor_file else None,
            "anchor_sha256": anchor_sha256,
        },
        "input_validation": {
            "mode": "inline-fail-closed",
            "skill_corpus": str(CORPUS_PATH.resolve()),
        },
        "apply_train_samples": n_instances,
        "anchor_train_samples": n_anchor_samples,
        "train_samples": len(dataset),
        "training_counts": {
            "positive_instances": n_instances,
            "positive_samples": n_positive_samples,
            "near_anchor_samples": n_near_anchor_samples,
            "random_anchor_samples": n_random_anchor_samples,
            "total_samples": len(dataset),
        },
        "max_samples": max_samples,
        "validation": {
            "val_backend": val_backend,
            "val_async": val_async,
            "val_concurrency": val_concurrency,
            "val_workers": val_workers,
            "val_freq": val_freq,
            "compute_val_loss": compute_val_loss,
            "num_train": num_train,
            "num_val": num_val,
            "fixed_val_offset": fixed_val_offset if val_fix_val else None,
            "val_samples_seen": len(val_instances),
            "validated_epochs": validated_epochs,
            "best_epoch": val_cb.best_epoch if val_cb else None,
            "best_acc": val_cb.best_acc if val_cb else None,
            "base_acc": base_acc,
        },
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Optional debug export. Disabled in production because it duplicates every
    # training sample for every adapter and can consume tens of GB at 1,224 runs.
    if export_chat:
        export_sharegpt(data, output_path / "training_data_chat.jsonl", include_skill)

    # Save deployment config (vLLM serve commands)
    save_deployment_config(output_path, base_model, skill_id, dataset_name, rank, epochs)

    print(f"\nLoRA adapter saved to: {output_path}")
    print(f"Deployment metadata: {output_path / 'deployment_config.json'}")

    # Completion marker — written ONLY after all epochs + deploy-best succeed.
    # step_train skips a unit iff this exists (NOT best_epoch.json, which is
    # written after epoch 1 and would let an interrupted 1-epoch run look "done").
    adapter_path = output_path / "adapter_model.safetensors"
    json.dump(
        {"epochs": epochs, "best_epoch": val_cb.best_epoch if val_cb else None,
         "best_acc": val_cb.best_acc if val_cb else None,
         "validation_mode": (
             "async-generation" if val_async else "synchronous-generation"
         ) if val_backend != "none" else "none",
         "validated_epochs": validated_epochs,
         "batch_size": batch_size,
         "gradient_accumulation_steps": gradient_accumulation_steps,
         "effective_batch_size": batch_size * gradient_accumulation_steps,
         "training_job_id": training_job_id,
         "anchor_sha256": anchor_sha256,
         "adapter": adapter_path.name},
        open(output_path / "TRAINING_DONE", "w"), indent=2)



def main():
    parser = argparse.ArgumentParser(description="Train skill LoRA adapter")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--skill-id", required=True, help="Skill ID")
    parser.add_argument("--trajectory-file", required=True, help="Trajectory JSON file")
    parser.add_argument("--output-dir", required=True,
                        help="Shared output directory (must be under MSI_SHARED_ROOTS)")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size × this value)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-length", type=int, default=20480, help="Max sequence length")
    parser.add_argument("--strip-skill", action="store_true", default=False,
                        help="Strip skill text from training data for ablation")
    parser.add_argument("--use-quantization", type=bool, default=False, help="Use 4bit quantization")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (take first N instances from trajectory)")
    parser.add_argument("--target-group", type=str, default=None,
                        choices=list(TARGET_GROUPS),
                        help="Preset LoRA target-module group for ablation on the "
                             "hybrid arch: standard/full_attn/ssm/mlp/all_attn/all.")
    parser.add_argument("--target-modules", type=str, default=None,
                        help="Comma-separated LoRA target module suffixes "
                             "(overrides --target-group)")
    # ---- per-epoch validation + TensorBoard ----
    parser.add_argument("--num-train", type=int, default=150,
                        help="Held-out TRAIN split size (acceptable instances)")
    parser.add_argument("--num-val", type=int, default=500,
                        help="Held-out VALIDATE split size")
    parser.add_argument("--split-seed", type=int, default=42, help="Train/val split seed")
    parser.add_argument("--val-backend", choices=["none", "vllm", "inprocess"],
                        default="none",
                        help="Validation backend: 'vllm' generates on a validate server via "
                             "dynamic LoRA; 'inprocess' generates on the live PeftModel; "
                             "'none' disables validation")
    parser.add_argument("--vl-rename", action="store_true",
                        help="Rename adapter keys to a VL nested language-model path before "
                             "vLLM registration; architecture detection enables this when needed")
    parser.add_argument("--val-api-base", default=None,
                        help="Validate server base URL, e.g. http://localhost:8005/v1")
    parser.add_argument("--val-lora-name-base", default=None,
                        help="Stable lora_name to register the adapter under (default val_<skill_id>)")
    parser.add_argument("--val-temperature", type=float, default=0.0, help="Validation sampling temperature")
    parser.add_argument("--val-max-tokens", type=int, default=20480, help="Validation max new tokens")
    parser.add_argument("--val-workers", type=int, default=256,
                        help="Concurrent validation requests (the vLLM server only saturates ~256+ "
                             "to fill batches; lower values underutilize it)")
    parser.add_argument("--base-model-name", default=None,
                        help="Model name for thinking-suppression (e.g. Qwen3.5-2B); default = base_model basename")
    parser.add_argument("--val-async", action="store_true",
                        help="Run validation generation on a background thread so it does NOT block "
                             "training (overlaps val with the next epoch). val/accuracy still logs at "
                             "step=epoch — no TB lag.")
    parser.add_argument("--val-concurrency", type=int, default=2,
                        help="With --val-async: how many epochs validate concurrently per skill "
                             "(versioned lora_names avoid adapter clobber). Higher = more server load.")
    parser.add_argument("--val-freq", type=int, default=1,
                        help="Generation-validation cadence in epochs for every dataset and "
                             "backend. val_loss still logs every epoch when enabled. "
                             "1 = run gen-val (val/accuracy) every epoch.")
    parser.add_argument("--skip-val-loss", action="store_true",
                        help="Skip synchronous teacher-forced validation loss; production "
                             "selects epochs by async generation accuracy instead")
    parser.add_argument("--fix-val", action="store_true",
                        help="Keep train as a nested prefix and fix validation at --val-offset")
    parser.add_argument("--val-offset", type=int, default=None,
                        help="With --fix-val, shuffled validation slice start")
    parser.add_argument("--skip-base-baseline", action="store_true",
                        help="Skip the pre-training base (no-LoRA) baseline eval (base acc is "
                             "known from prior naive runs; saves the 128-gen overhead per run).")
    parser.add_argument("--anchor-file", default=None,
                        help="ANCHOR negatives JSON (retrieve-miss robustness). Added to the "
                             "train Dataset only (never validation).")
    parser.add_argument("--anchor-near-count", type=int, default=None,
                        help="Use the first N near-miss rows from the anchor file")
    parser.add_argument("--anchor-random-count", type=int, default=None,
                        help="Use the first N random rows from the anchor file")
    parser.add_argument("--training-job-id", required=True,
                        help="Resolved adapter-resume identity")
    parser.add_argument("--export-chat", action="store_true",
                        help="Debug only: duplicate training samples into each adapter directory")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--tb-dir", type=str, default=None,
                        help="TensorBoard log directory (default: <output>/tb). Set to a "
                             "distinct folder per run purpose (e.g. tb/sweep, tb/production) "
                             "so runs are clearly separated in tensorboard --logdir.")
    args = parser.parse_args()
    apply_qwen_thinking(args.base_model)

    # Always detect the arch profile (for text_only_load); targets come from it
    # unless --target-modules/--target-group explicitly override.
    from msi.models.arch_profile import detect
    _prof = detect(args.base_model)
    text_only_load = _prof.text_only_load
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    elif args.target_group:
        target_modules = TARGET_GROUPS[args.target_group]
    else:
        # Auto-detect model-family-specific LoRA targets.
        target_modules = _prof.lora_targets
        print(f"[arch_profile] auto LoRA targets for {args.base_model}: {target_modules} | "
              f"text_only_load={text_only_load}", flush=True)

    train_lora(
        base_model=args.base_model,
        skill_id=args.skill_id,
        trajectory_file=args.trajectory_file,
        output_dir=args.output_dir,
        rank=args.rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        include_skill=not args.strip_skill,
        use_quantization=args.use_quantization,
        lora_dropout=args.lora_dropout,
        max_samples=args.max_samples,
        target_modules=target_modules,
        num_train=args.num_train,
        num_val=args.num_val,
        split_seed=args.split_seed,
        val_backend=args.val_backend,
        val_api_base=args.val_api_base,
        val_lora_name_base=args.val_lora_name_base,
        vl_rename=args.vl_rename or _prof.vl_rename,
        text_only_load=text_only_load,
        val_temperature=args.val_temperature,
        val_max_tokens=args.val_max_tokens,
        val_workers=args.val_workers,
        base_model_name=args.base_model_name,
        tb_log=not args.no_tb,
        tb_dir=args.tb_dir,
        val_async=args.val_async,
        val_concurrency=args.val_concurrency,
        val_freq=args.val_freq,
        compute_val_loss=not args.skip_val_loss,
        val_fix_val=args.fix_val,
        fixed_val_offset=args.val_offset,
        skip_base_baseline=args.skip_base_baseline,
        anchor_file=args.anchor_file,
        anchor_near_count=args.anchor_near_count,
        anchor_random_count=args.anchor_random_count,
        training_job_id=args.training_job_id,
        export_chat=args.export_chat,
    )


if __name__ == "__main__":
    main()
