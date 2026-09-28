"""In-process PEFT inference for model families not served reliably by vLLM.

Two entry points:
* :class:`InProcessClient` — an OpenAI-compatible client whose
  ``chat.completions.create(...)`` runs ``PeftModel.generate``. Drop-in for the
  vendor ``chat``/``chat_messages``/``run_with_tools`` helpers (which only touch
  ``client.chat.completions.create`` and read ``.choices[0].message.content``),
  so medcalc's tool loop works unchanged. Per-request ``set_adapter`` /
  ``disable_adapter``; thread-locked → serial within a process (obtain
  parallelism by sharding across processes).
* :func:`batched_generate` — left-padded batched generation for tool-less
  validation/inference (~×batch faster than serial per-instance).

Adapters are preloaded once and each instance routes to one skill adapter.
"""

from __future__ import annotations

import contextlib
import threading

import torch


# ---------------------------------------------------------------------------
# Minimal OpenAI-response look-alikes (vendor only reads .choices[0].message.content)
# ---------------------------------------------------------------------------

class _Message:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message", "finish_reason")

    def __init__(self, content: str, finish_reason: str):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    __slots__ = ("completion_tokens",)

    def __init__(self, completion_tokens: int):
        self.completion_tokens = completion_tokens


class _Response:
    __slots__ = ("choices", "usage")

    def __init__(self, content: str, finish_reason: str, completion_tokens: int):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(completion_tokens)


def _enable_thinking_from_extra(extra_body) -> bool:
    """Parse chat_template_kwargs.enable_thinking (Qwen3) from an extra_body."""
    if not isinstance(extra_body, dict):
        return False
    ctk = extra_body.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        return bool(ctk.get("enable_thinking", False))
    return bool(extra_body.get("enable_thinking", False))


def _ensure_pad_left(tokenizer) -> None:
    """Left-pad so the generated region is always at the sequence end."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


# ---------------------------------------------------------------------------
# OpenAI-compatible shim client
# ---------------------------------------------------------------------------

class InProcessClient:
    """OpenAI-compatible client backed by an in-process ``PeftModel``.

    ``client.chat.completions.create(model=..., messages=..., ...)`` generates
    via ``peft_model.generate``. In a final LoRA run every model name must be
    present in ``request_to_adapter``; an unknown name is a hard error. An
    explicitly configured diagnostic fallback may set ``allow_base_model``.
    A client with an empty map is the explicit base-only client used by
    non-LoRA methods.

    Generation is serialized by a lock (``PeftModel.generate`` is not
    thread-safe); callers obtain process-level parallelism by selecting
    multiple GPUs in ``msi evaluate``.
    """

    def __init__(self, peft_model, tokenizer, request_to_adapter: dict[str, str],
                 default_max_tokens: int = 2048, allow_base_model: bool = False):
        self.model = peft_model
        self.tokenizer = tokenizer
        self.request_to_adapter = request_to_adapter
        self.default_max_tokens = default_max_tokens
        self.allow_base_model = allow_base_model
        self._lock = threading.Lock()
        _ensure_pad_left(tokenizer)

    # client.chat.completions.create(...)  →  self.create(...)
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, *, model: str, messages: list[dict],
               temperature: float | None = None, max_tokens: int | None = None,
               extra_body: dict | None = None, stop: list[str] | None = None,
               **kwargs):
        enable_thinking = _enable_thinking_from_extra(extra_body)
        max_new = max_tokens if max_tokens is not None else self.default_max_tokens
        do_sample = float(temperature) > 0 if temperature is not None else False

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        if stop:
            gen_kwargs["stop_strings"] = stop
            gen_kwargs["tokenizer"] = self.tokenizer

        with self._lock:
            if hasattr(self.model, "set_adapter") and model in self.request_to_adapter:
                self.model.set_adapter(self.request_to_adapter[model])
                with torch.no_grad():
                    out = self.model.generate(**gen_kwargs)
            elif self.request_to_adapter and not self.allow_base_model:
                raise RuntimeError(
                    f"unknown adapter request in closed-world run: {model}"
                )
            elif hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    with torch.no_grad():
                        out = self.model.generate(**gen_kwargs)
            else:
                # base-only model (no PeftModel wrapper) — just generate
                with torch.no_grad():
                    out = self.model.generate(**gen_kwargs)
            new = out[0][inputs.input_ids.shape[1]:]
            content = self.tokenizer.decode(new, skip_special_tokens=True)
        finish_reason = "length" if int(new.numel()) >= max_new else "stop"
        return _Response(content, finish_reason, int(new.numel()))

# ---------------------------------------------------------------------------
# Batched generation (tool-less validation / inference)
# ---------------------------------------------------------------------------

def batched_generate(peft_model, tokenizer, messages_batch: list[list[dict]], *,
                     max_new_tokens: int = 512, batch_size: int = 16,
                     temperature: float = 0.0, disable_adapter: bool = False,
                     enable_thinking: bool = False) -> list[str]:
    """Left-padded batched generation over a list of message-lists.

    Each item is a chat ``messages`` list (e.g. system+user). Returns the
    decoded assistant response per item (same order). ``disable_adapter=True``
    generates with the base model (for the no-LoRA baseline). Thinking is
    suppressed by default (``enable_thinking=False``) to match non-reasoning
    baselines.
    """
    _ensure_pad_left(tokenizer)
    pad_id = tokenizer.eos_token_id
    do_sample = float(temperature) > 0

    was_training = peft_model.training
    peft_model.eval()
    outs: list[str | None] = [None] * len(messages_batch)
    ctx = peft_model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx:
        for i in range(0, len(messages_batch), batch_size):
            mb = messages_batch[i:i + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for m in mb
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(peft_model.device)
            gk = dict(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=pad_id,
            )
            if do_sample:
                gk["temperature"] = float(temperature)
            with torch.no_grad():
                out = peft_model.generate(**gk)
            prompt_len = enc.input_ids.shape[1]  # left-padded → same for all
            for j in range(len(mb)):
                outs[i + j] = tokenizer.decode(out[j][prompt_len:],
                                               skip_special_tokens=True)
            del out, enc
            torch.cuda.empty_cache()  # free this batch's KV/logits before the next
    peft_model.train(was_training)
    return outs  # type: ignore[return-value]


def batched_tool_validate(peft_model, tokenizer, instances_msgs, tools, *,
                          max_new_tokens=2048, batch_size=16, disable_adapter=False,
                          max_rounds=5, enable_thinking=False):
    """Batched tool-loop validation (medcalc): ``run_with_tools`` across many
    instances, batching generation across instances at each round.

    The tool loop is sequential *per instance* (each round depends on the prior
    tool result), but different instances are independent — so at round R, all
    still-active instances batch-generate their next turn together (GPU-efficient,
    single process, no per-epoch reload — unlike multi-process; no lock stall —
    unlike multi-thread). Finished instances exit; the batch shrinks each round.

    ``instances_msgs``: list of ``(key, messages)`` where ``messages`` starts as
    ``[system, user]`` and is mutated as tool results append. Returns
    ``{key: accumulated model_output}`` (heads + final answer, matching
    ``run_with_tools``'s ``model_output`` semantics).
    """
    from sragents.infer.engines.tool_loop import parse_tool_call, execute_tool
    tool_index = {t["name"]: t for t in tools}
    outputs: dict = {}
    acc: dict = {}
    active = [(k, m) for k, m in instances_msgs]
    ctx = peft_model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx:
        for _ in range(max_rounds):
            if not active:
                break
            msgs = [m for _, m in active]
            gens = batched_generate(
                peft_model, tokenizer, msgs, max_new_tokens=max_new_tokens,
                batch_size=batch_size, disable_adapter=False,
                enable_thinking=enable_thinking,
            )
            nxt = []
            for (key, m), out in zip(active, gens):
                parsed = parse_tool_call(out or "", tool_index)
                if parsed is None:
                    acc[key] = acc.get(key, "") + (out or "")
                    outputs[key] = acc[key]
                else:
                    head, name, args = parsed
                    acc[key] = acc.get(key, "") + head
                    try:
                        result = execute_tool(tool_index[name], args)
                    except Exception as e:  # noqa: BLE001
                        result = f"Error: {e}"
                    m.append({"role": "assistant", "content": head})
                    m.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
                    nxt.append((key, m))
            active = nxt
    for key, _ in active:  # unfinished after max_rounds
        outputs.setdefault(key, acc.get(key, ""))
    return outputs


# ---------------------------------------------------------------------------
# Model loading helpers for the optional in-process backend
# ---------------------------------------------------------------------------

def load_peft_with_adapters(base_model_path: str,
                            adapter_paths: dict[str, str],
                            dtype=torch.bfloat16):
    """Load base once + preload every adapter (named by its skill id).

    Returns ``(peft_model, tokenizer)``. Each adapter is registered under its
    skill id so :class:`InProcessClient` can ``set_adapter(sid)`` per request.
    Adapters are frozen (``is_trainable=False``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    _ensure_pad_left(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, local_files_only=True,
    ).to("cuda").eval()

    items = list(adapter_paths.items())
    sid0, path0 = items[0]
    pm = PeftModel.from_pretrained(base, path0, adapter_name=sid0, is_trainable=False)
    for sid, path in items[1:]:
        pm.load_adapter(path, adapter_name=sid)
    return pm.eval(), tokenizer


def load_base_only(base_model_path: str, dtype=torch.bfloat16):
    """Load base model only (no adapters) for non-LoRA methods (naive/golden_skill).

    Returns ``(model, tokenizer)`` — a plain AutoModelForCausalLM (no PeftModel
    wrapper). InProcessClient handles it via the ``else`` branch (just generate,
    no set/disable_adapter). Same greedy decode as PeftModel-disabled.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    _ensure_pad_left(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, local_files_only=True,
    ).to("cuda").eval()
    return model, tokenizer
