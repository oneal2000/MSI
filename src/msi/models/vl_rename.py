"""Rename a PEFT LoRA adapter's weight keys to vLLM's VL-model path.

Problem
-------
Adapters are trained on the text-only ``Qwen3_5ForCausalLM`` (via
``AutoModelForCausalLM``), so PEFT saves keys like::

    base_model.model.model.layers.N.mlp.down_proj.lora_A.weight

vLLM serves the **VL** ``Qwen3_5ForConditionalGeneration``, whose language model
is nested under ``.language_model``::

    base_model.model.language_model.model.layers.N.mlp.down_proj.lora_A.weight

vLLM's PEFT loader matches adapter weights by **full module path**, so the
text-prefix keys never reach the nested LM -> the adapter is registered, the
LoRA kernels run, but the delta never enters the forward pass (silent no-op).

Fix
---
Insert the missing ``language_model.`` segment. This is a pure key rewrite on
``adapter_model.safetensors`` — no retraining, no weight change. Idempotent.

Usage
-----
    from msi.models.vl_rename import rename_adapter_to_vl
    rename_adapter_to_vl(src_dir, dst_dir)          # write a renamed copy
    rename_adapter_to_vl(src_dir, src_dir)          # in-place (overwrites)

    # CLI
    python -m msi.models.vl_rename <src_adapter_dir> [dst_adapter_dir]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Insert this segment after the PEFT wrapper prefix.
_PREFIX = "base_model.model.model."
_VL_PREFIX = "base_model.model.language_model.model."


def _is_vl_path(key: str) -> bool:
    return "language_model.model." in key or ".visual." in key


def rename_adapter_to_vl(src_dir: str | Path, dst_dir: str | Path | None = None,
                          overwrite: bool = True) -> Path:
    """Rewrite ``adapter_model.safetensors`` keys to the VL nested-LM path.

    Args:
        src_dir: adapter directory containing ``adapter_model.safetensors`` +
            ``adapter_config.json``.
        dst_dir: destination directory (default: ``src_dir`` = in-place).
        overwrite: allow ``dst_dir == src_dir`` (rewrites the safetensors file).

    Returns:
        The destination directory.
    """
    src = Path(src_dir)
    dst = Path(dst_dir) if dst_dir is not None else src
    src_w = src / "adapter_model.safetensors"
    src_cfg = src / "adapter_config.json"
    if not src_w.exists():
        raise FileNotFoundError(f"no adapter_model.safetensors in {src}")

    in_place = dst.resolve() == src.resolve()
    if not in_place:
        dst.mkdir(parents=True, exist_ok=True)

    renamed: dict = {}
    skipped = 0
    with safe_open(str(src_w), framework="pt") as f:
        for k in f.keys():
            if k.startswith(_PREFIX):
                nk = _VL_PREFIX + k[len(_PREFIX):]
            elif _is_vl_path(k):
                nk = k  # already VL-path (or a non-LM key) -> leave as-is
                skipped += 1
            else:
                nk = k  # unrelated key (shouldn't happen for these adapters)
                skipped += 1
            renamed[nk] = f.get_tensor(k)

    out_w = dst / "adapter_model.safetensors"
    if in_place and out_w.exists() and not overwrite:
        raise FileExistsError(out_w)
    save_file(renamed, str(out_w), metadata={"format": "pt"})

    # copy config + any side files (tokenizer etc.) when writing a new dir
    if not in_place:
        if src_cfg.exists():
            shutil.copy(src_cfg, dst / "adapter_config.json")
        for extra in ("tokenizer.json", "tokenizer_config.json",
                      "added_tokens.json", "special_tokens_map.json"):
            s = src / extra
            if s.exists():
                shutil.copy(s, dst / extra)
    return dst


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m msi.models.vl_rename "
              "<src_adapter_dir> [dst_adapter_dir]",
              file=sys.stderr)
        return 2
    src = argv[0]
    dst = argv[1] if len(argv) > 1 else None
    out = rename_adapter_to_vl(src, dst)
    print(f"renamed adapter -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
