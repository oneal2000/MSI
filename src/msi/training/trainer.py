"""Custom Trainer (logits_to_keep loss) + ValidationCallback (per-epoch val, best-epoch deploy)."""
import json
import math
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from msi.models.vllm_lora import register_lora as _register_lora, unregister_lora as _unregister_lora
from msi.training.validate import _inprocess_generate_validation, compute_val_loss, run_validation


class LTKTrainer(Trainer):
    """Custom Trainer: computes loss externally with logits_to_keep label alignment."""

    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        # Only the last assistant turn has non-(-100) labels; compute logits ONLY there
        # via logits_to_keep instead of full [bs, seq, 248K]. On the 248K Qwen3.5 vocab a
        # 5k-token batch at bs=4 materializes ~10GB bf16 logits -> ~30GB after the float
        # CE upcast -> OOM. ltk = max trainable span in batch (+2 for the shift) keeps
        # logits to [bs, ~ltk, vocab]; loss is identical (the -100 logits are never used).
        ltk = int((labels != -100).sum(dim=-1).max().item()) + 2
        outputs = model(**inputs, logits_to_keep=max(ltk, 1))
        logits = outputs.logits  # [bs, ltk, vocab]

        padded = torch.cat([
            labels,
            torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype),
        ], dim=-1)
        shift_labels = padded[:, 1:].contiguous()[:, -logits.size(1):]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return loss
class ValidationCallback(TrainerCallback):
    """Per-epoch validation + TensorBoard logging.

    Writes to a dedicated ``SummaryWriter`` pointed at the same ``logging_dir``
    as the Trainer, so custom scalars (``train/loss_epoch``, ``val/accuracy``,
    ``val/loss``) merge with the Trainer's per-step ``train/loss``. Reads
    ``loss`` from ``on_log`` read-only — never re-logs custom metrics via
    ``trainer.log()`` (it re-enters ``on_log`` and corrupts the loss buffer).

    Validation generation normally BLOCKS training (synchronous in on_epoch_end:
    the training GPU idles while held-out generations run on the server). With
    ``val_async=True`` the slow generation step is offloaded to a background
    thread — on_epoch_end only does the quick synchronous part (save adapter +
    teacher-forced val_loss) and returns, so training resumes immediately and
    validation overlaps with the next epoch. TB scalars are logged at
    ``step=epoch`` regardless of wall-clock, so curves are not shifted.
    """

    def __init__(self, *, tb_writer, val_instances, val_samples, collator,
                 skill_content, tools, dataset, base_model_name, api_base,
                 lora_name, output_path, model, device, inprocess_batch_size,
                 temperature, max_tokens, workers, compute_val_loss_flag=True,
                 val_async=False, val_concurrency=2,
                 tokenizer=None, val_backend="vllm", val_freq=1,
                 root=None, base_model_path=None, skill_id=None,
                 vl_rename: bool = False):
        self.tb = tb_writer
        self.tokenizer = tokenizer
        self.val_backend = val_backend
        self.val_freq = max(1, val_freq)
        self.root = root
        self.base_model_path = base_model_path
        self.skill_id = skill_id
        self.val_instances = val_instances
        self.val_samples = val_samples
        self.collator = collator
        self.skill_content = skill_content
        self.tools = tools
        self.dataset = dataset
        self.base_model_name = base_model_name
        self.api_base = api_base
        self.lora_name = lora_name  # base name; per-epoch name = f"{lora_name}_e{ep}"
        self.output_path = Path(output_path)
        self.model = model
        self.device = device
        self.inprocess_batch_size = inprocess_batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers
        self.compute_val_loss_flag = compute_val_loss_flag
        self.vl_rename = vl_rename
        self.val_async = val_async
        self.val_concurrency = val_concurrency
        self.epoch_loss_buf: list[float] = []
        self.best_acc = -1.0
        self.best_epoch = -1
        # Concurrent validation pool. Versioned lora_names (val_<skill>_e<ep>)
        # let multiple epochs validate at once without clobbering the adapter
        # (each epoch's adapter coexists on the server under its own name).
        self._lock = threading.Lock()
        self._futures: list[tuple[int, object]] = []
        self._scheduled_epochs: list[int] = []
        self._pool = None
        if val_async:
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, val_concurrency), thread_name_prefix="val")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.epoch_loss_buf.append(logs["loss"])

    def _staging_dir(self, ep: int) -> Path:
        # Per-epoch frozen snapshot dir (race-free under concurrent validation).
        return self.output_path / "_val_adapters" / f"e{ep}"

    def _generation_validation(self, ep: int, train_loss: float, val_loss: float):
        """Run generation validation for epoch ``ep`` and log/save best.

        ``val_backend="vllm"``: register the saved per-epoch adapter and generate
        on the validation server.  VL-nested models such as Qwen3.5 use the
        architecture profile's key rename before registration. Sync or async.
        ``val_backend="inprocess"``: generate on the live PeftModel (adapter
        deterministically applied — the PRAG primitive). Generation-val runs
        every ``val_freq`` epochs for ALL datasets (no-tool datasets batch;
        medcalc's serial tool loop can't).
        """
        acc = float("nan")
        details: list[dict] = []
        ran = True

        if self.val_backend == "inprocess":
            if (self.val_freq > 1
                    and ep % self.val_freq != 0):
                ran = False  # gen-val cadence (val_loss still every epoch)
            if ran and self.val_instances:
                acc, details = _inprocess_generate_validation(
                    self.model, self.tokenizer, self.val_instances,
                    self.skill_content, self.tools, dataset=self.dataset,
                    val_lora_name=self.lora_name,
                    base_model_name=self.base_model_name,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    batch_size=self.inprocess_batch_size, disable_adapter=False,
                )
        else:  # vllm
            staging = self._staging_dir(ep)
            name = f"{self.lora_name}_e{ep}"
            reg_path = None
            try:
                reg_path = _register_lora(
                    self.api_base, name, str(staging.resolve()),
                    vl_rename=self.vl_rename)
                if not reg_path:
                    raise RuntimeError(f"failed to register validation adapter for epoch {ep}")
                if not self.val_instances:
                    raise RuntimeError("validation set is empty")
                acc, details = run_validation(
                    self.val_instances, self.skill_content, self.tools,
                    api_base=self.api_base, lora_name=name,
                    dataset=self.dataset, base_model_name=self.base_model_name,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    workers=self.workers,
                )
            finally:
                # Free the per-epoch LoRA even if generation raises, so one bad
                # validation cannot exhaust the shared server's LoRA slots.
                if reg_path:
                    _unregister_lora(self.api_base, name)
                    if self.vl_rename:  # temp *_vl dir for Qwen3.5
                        shutil.rmtree(reg_path, ignore_errors=True)

        if ran and self.val_instances:
            expected_ids = {row.get("instance_id") for row in self.val_instances}
            actual_ids = [row.get("instance_id") for row in details]
            if (len(details) != len(self.val_instances)
                    or len(actual_ids) != len(set(actual_ids))
                    or set(actual_ids) != expected_ids):
                raise RuntimeError(
                    f"validation coverage mismatch at epoch {ep}: "
                    f"expected={len(expected_ids)}, rows={len(details)}, "
                    f"unique={len(set(actual_ids))}"
                )
            errored = [row for row in details if row.get("error")]
            truncated_rows = [row for row in details if row.get("truncated")]
            if errored:
                raise RuntimeError(
                    f"validation produced {len(errored)}/{len(details)} error "
                    f"rows at epoch {ep}; examples={errored[:3]}"
                )
            if truncated_rows:
                # Runaway generations are model behaviour, not failures: they
                # already count as incorrect in val_accuracy (same semantics as
                # the anchor runaway-skip decision).
                print(f"[epoch {ep}] val note: {len(truncated_rows)}/{len(details)} "
                      f"truncated (runaway) rows counted as incorrect",
                      flush=True)
            if self.tb is not None:
                with self._lock:
                    self.tb.add_scalar("val/accuracy", acc, ep)
                    self.tb.flush()
            print(f"[epoch {ep}] val_accuracy={acc:.4f} "
                  f"({sum(1 for d in details if d['correct'])}/{len(details)})",
                  flush=True)
            with self._lock:
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_epoch = ep
                    best_dir = self.output_path / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(self._staging_dir(ep), best_dir)
                    json.dump({"best_epoch": ep, "best_acc": acc},
                              open(self.output_path / "best_epoch.json", "w"),
                              indent=2)
        elif not ran:
            print(f"[epoch {ep}] generation validation skipped "
                  f"(backend={self.val_backend}, medcalc cadence)", flush=True)

        with self._lock:
            json.dump(
                {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                 "val_accuracy": acc, "n": len(details), "sample": details[:20]},
                open(self.output_path / f"val_epoch{ep}.json", "w"), indent=2,
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = round(state.epoch)
        train_loss = (sum(self.epoch_loss_buf) / len(self.epoch_loss_buf)
                      if self.epoch_loss_buf else float("nan"))
        if self.tb is not None:
            self.tb.add_scalar("train/loss_epoch", train_loss, ep)
        print(f"\n[epoch {ep}] train_loss_epoch={train_loss:.4f}", flush=True)

        # val/loss — teacher-forced held-out (same scale as train loss). Always
        # in-process & valid (unlike the vLLM generation path on Qwen3.5).
        val_loss = float("nan")
        if self.compute_val_loss_flag and self.val_samples:
            val_loss = compute_val_loss(
                self.model, self.val_samples, self.collator,
                self.inprocess_batch_size, self.device,
            )
            if self.tb is not None:
                self.tb.add_scalar("val/loss", val_loss, ep)
            print(f"[epoch {ep}] val_loss={val_loss:.4f}", flush=True)

        # Both backends need exact epoch weights for 1-SE deployment;
        # only vLLM also registers these snapshots with a separate server.
        if self.val_freq > 1 and ep % self.val_freq != 0:
            self.epoch_loss_buf = []
            return control
        self.model.save_pretrained(str(self._staging_dir(ep)))
        self._scheduled_epochs.append(ep)

        if self.val_backend == "vllm" and self.val_async:
            fut = self._pool.submit(
                self._generation_validation, ep, train_loss, val_loss)
            self._futures.append((ep, fut))
            print(f"[epoch {ep}] validation submitted (async, "
                  f"pool={self.val_concurrency})", flush=True)
        else:
            self._generation_validation(ep, train_loss, val_loss)
            if self.tb is not None:
                self.tb.flush()

        self.epoch_loss_buf = []
        return control

    def _apply_one_se_rule(self):
        """1-SE rule (one-standard-error): among trained epochs whose val_acc is
        within 1 SE of the max, pick the EARLIEST. Overrides the incremental
        argmax ``best/`` set during training.

        Why: argmax over K noisy val points suffers order-statistic optimism
        (~+1 SE) and chases upward fluctuations that won't reproduce at test.
        Epochs within 1 SE of the max are statistically tied; preferring the
        earliest (least-trained) guards against both val-noise and overfitting.
        Zero extra eval — operates on already-recorded val points. n=64 →
        SE≈0.056 at p=0.73, so the rule is deliberately conservative.
        """
        import math
        pairs = []
        for f in self.output_path.glob("val_epoch*.json"):
            try:
                d = json.load(open(f))
                acc, ep = d.get("val_accuracy"), d.get("epoch")
                if acc is not None and ep is not None and acc == acc:  # skip NaN
                    pairs.append((int(ep), float(acc)))
            except Exception:  # noqa: BLE001
                continue
        if not pairs:
            return
        acc_max = max(a for _, a in pairs)
        n = len(self.val_instances) if self.val_instances else 64
        se = math.sqrt(acc_max * (1 - acc_max) / max(n, 1))
        thr = acc_max - se
        cands = [(e, a) for e, a in pairs if a >= thr]
        chosen_ep, chosen_acc = min(cands, key=lambda ea: ea[0])
        staging = self._staging_dir(chosen_ep)
        if not staging.exists():
            raise RuntimeError(f"selected epoch staging adapter is missing: {staging}")
        best_dir = self.output_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(staging, best_dir)
        self.best_epoch = chosen_ep
        self.best_acc = chosen_acc
        json.dump({
            "best_epoch": chosen_ep, "best_acc": chosen_acc, "rule": "1-SE",
            "acc_max": acc_max, "se": se, "threshold": thr, "n_val": n,
            "within_thr": sorted(cands),
        }, open(self.output_path / "best_epoch.json", "w"), indent=2)
        print(f"[1-SE] chosen epoch {chosen_ep} (acc={chosen_acc:.4f}); "
              f"acc_max={acc_max:.4f} SE={se:.4f} thr={thr:.4f} "
              f"within-thr={sorted(e for e, _ in cands)}", flush=True)

    def validated_epochs(self) -> list[int]:
        """Return complete required validation epochs, or fail closed.

        This is deliberately reusable after ``Trainer.train()`` so the exact
        evidence set can be persisted in metadata and ``TRAINING_DONE`` rather
        than inferred later from a best-epoch file alone.
        """
        if not self.val_instances:
            return []
        expected = sorted(set(self._scheduled_epochs))
        completed = []
        for ep in expected:
            path = self.output_path / f"val_epoch{ep}.json"
            if not path.is_file():
                continue
            payload = json.load(open(path))
            accuracy = payload.get("val_accuracy")
            if (accuracy is not None and math.isfinite(float(accuracy))
                    and int(payload.get("n", -1)) == len(self.val_instances)):
                completed.append(ep)
        if not expected or completed != expected:
            raise RuntimeError(
                "required validation checkpoints are incomplete: "
                f"expected={expected}, completed={completed}"
            )
        return completed

    def on_train_end(self, args, state, control, **kwargs):
        # Drain pending async validations so the final epochs still get scored.
        if self._pool is not None:
            print("waiting for pending async validations to finish...", flush=True)
            validation_errors = []
            for ep, fut in self._futures:
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    validation_errors.append(f"epoch {ep}: {type(e).__name__}: {e}")
                    print(f"async validation error at epoch {ep}: {e}", flush=True)
            self._pool.shutdown(wait=True)
            if self.tb is not None:
                self.tb.flush()
            if validation_errors:
                raise RuntimeError(
                    "one or more required async validations failed: "
                    + "; ".join(validation_errors)
                )
        if self.val_instances:
            self.validated_epochs()
            self._apply_one_se_rule()
