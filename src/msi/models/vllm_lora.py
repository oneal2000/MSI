"""Dynamic LoRA register/unregister helpers for a vLLM student server.

General for any model: ``register_lora`` POSTs ``/v1/load_lora_adapter`` and
``unregister_lora`` POSTs ``/v1/unload_lora_adapter``. Pair them to keep the
server's LoRA pool bounded (``--max-loras``) across skills/epochs/datasets.

**Qwen3.5-specific quirk (``vl_rename=True``):** adapters trained on text
``Qwen3_5ForCausalLM`` carry PEFT keys ``base_model.model.model.layers.*``, but
vLLM serves Qwen3.5 only via the VL ``Qwen3_5ForConditionalGeneration`` (no
text-only arch is registered), whose LM is nested under
``base_model.model.language_model.model.layers.*``. vLLM's PEFT loader matches by
full module path, so the text-prefix keys never reach the LM (silent base
no-op). ``vl_rename`` writes a renamed copy on the way to the server. This is
opt-in: off by default (general), on for Qwen3.5 (set by the pipeline / caller).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from msi.models.vl_rename import rename_adapter_to_vl


def _post(api_base: str, path: str, payload: dict, timeout: float = 30.0):
    url = api_base.rstrip("/") + path
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")[:300]
    except (urllib.error.URLError, OSError) as e:
        # Transient transport failures (e.g. Connection refused while the
        # server briefly swaps LoRA slots) must not escape: return a non-2xx
        # status so the callers' existing retry loops absorb them.
        return 0, f"{type(e).__name__}: {e}"


def register_lora(api_base: str, name: str, path: str,
                  vl_rename: bool = False,
                  retries: int = 10, delay: float = 3.0) -> str | None:
    """POST /v1/load_lora_adapter. Returns the path that was registered on
    success (the original ``path`` when ``vl_rename=False``; a unique temp dir
    when ``vl_rename=True``), else None.

    With VLLM_ALLOW_RUNTIME_LORA_UPDATING on the server, re-posting the same
    name hot-swaps. ``vl_rename`` is the Qwen3.5-specific key-path fix (see
    module docstring); leave it False for any other model. On success the caller
    should :func:`unregister_lora` (by name) and, only if ``vl_rename`` was set,
    ``rmtree`` the returned temp dir.
    """
    reg_path = str(path)
    if vl_rename:
        src = Path(path)
        if not (src / "adapter_model.safetensors").exists():
            print(f"    register_vl {name}: no adapter at {src}", flush=True)
            return None
        vl_dir = Path(tempfile.mkdtemp(prefix=f".{src.name}_vl-", dir=src.parent))
        try:
            rename_adapter_to_vl(src, vl_dir)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(vl_dir, ignore_errors=True)
            print(f"    register_vl {name}: rename failed: {e}", flush=True)
            return None
        reg_path = str(vl_dir)

    payload = {"lora_name": name, "lora_path": reg_path}
    for _ in range(retries):
        code, body = _post(api_base, "/load_lora_adapter", payload)
        if 200 <= code < 300:
            return reg_path
        if code in (400, 409) and "already" in body.lower():
            return reg_path
        print(f"    register {name}: HTTP {code} {body[:160]}", flush=True)
        time.sleep(delay)
    if vl_rename:
        shutil.rmtree(reg_path, ignore_errors=True)
    return None


def register_lora_vl(api_base: str, name: str, path: str,
                     retries: int = 10, delay: float = 3.0) -> str | None:
    """Qwen3.5 convenience wrapper: register_lora(..., vl_rename=True)."""
    return register_lora(api_base, name, path, vl_rename=True,
                         retries=retries, delay=delay)


def unregister_lora(api_base: str, name: str,
                    retries: int = 5, delay: float = 2.0) -> bool:
    """POST /v1/unload_lora_adapter — free a registered adapter. Idempotent."""
    payload = {"lora_name": name}
    for _ in range(retries):
        code, body = _post(api_base, "/unload_lora_adapter", payload)
        if 200 <= code < 300:
            return True
        if code in (400, 404) and "not" in body.lower():
            return True  # already gone
        print(f"    unregister {name}: HTTP {code} {body[:160]}", flush=True)
        time.sleep(delay)
    return False
