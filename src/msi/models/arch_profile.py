"""Derive LoRA and vLLM settings from a base model configuration.

The profile binds model-family-specific target modules, reasoning parser
settings, text-only loading, and multimodal LoRA key alignment in one place.
Detection reads configuration only and never loads model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from transformers import AutoConfig

# Standard text-transformer projections; embeddings and output heads are
# intentionally excluded.
STANDARD_TARGETS: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Model-specific settings keyed by Hugging Face ``model_type``.
_FAMILY_TABLE: dict[str, dict] = {
    "qwen3_5": {"family": "qwen35", "reasoning_parser": "qwen3",
                "suppress_thinking": True, "text_only_load": False},
    "qwen3":   {"family": "qwen3",  "reasoning_parser": "qwen3",
                "suppress_thinking": True, "text_only_load": False},
    # Gemma4 uses the text-only path and vLLM's default non-thinking template.
    "gemma4":  {"family": "gemma4", "reasoning_parser": None,
                "suppress_thinking": True, "text_only_load": True},
    "llama":   {"family": "llama",  "reasoning_parser": None,
                "suppress_thinking": False, "text_only_load": False},
    "gemma2":  {"family": "gemma2", "reasoning_parser": None,
                "suppress_thinking": False, "text_only_load": False},
    "granite": {"family": "granite", "reasoning_parser": None,
                "suppress_thinking": False, "text_only_load": False},
    # Apertus uses separate attention projections and an ungated MLP.
    "apertus": {"family": "apertus", "reasoning_parser": None,
                "suppress_thinking": False, "text_only_load": False,
                "lora_targets": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj",
                ]},
    # Phi-3 fuses QKV and gate/up projections.
    "phi3":    {"family": "phi3",   "reasoning_parser": None,
                "suppress_thinking": False, "text_only_load": False,
                "lora_targets": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
                "max_model_len": 40960},
}


@dataclass
class ArchProfile:
    family: str                       # human label (qwen35 / gemma4 / llama / …)
    lora_targets: list[str]           # PEFT target_modules
    reasoning_parser: str | None      # vLLM --reasoning-parser (None = omit)
    language_model_only: bool         # vLLM --language-model-only (strip VL head)
    vl_rename: bool                   # adapter key language_model. prefix (see vl_rename.py)
    suppress_thinking: bool           # chat_template_kwargs enable_thinking=False
    text_only_load: bool = False      # Gemma4: load *ForCausalLM text class, avoid ClippableLinear
    max_model_len: int | None = None  # --max-model-len when vLLM's derived len is too small (phi3)


def _family_of(model_type: str) -> dict | None:
    mt = (model_type or "").lower()
    # longest-prefix first so "qwen3_5" wins over "qwen3"
    for key in (
        "qwen3_5", "gemma4", "gemma2", "granite", "apertus", "phi3", "qwen3", "llama"
    ):
        if mt == key or mt.startswith(key + "_") or mt.startswith(key):
            # guard: don't mis-map e.g. "qwen3_vl" (different arch) onto qwen3_5;
            # startswith(qwen3) catches qwen3_vl → qwen3 family (parser qwen3, fine).
            return _FAMILY_TABLE[key]
    return None


def _is_vl_nested(cfg) -> bool:
    """Does vLLM serve this model as multimodal/VL-nested (LM under language_model.*)?

    True => --language-model-only at serve AND --vl-rename for adapter keys.
    Heuristics: a nested ``text_config`` (HF VL configs) or an
    ``architectures`` entry ending in ``ForConditionalGeneration``. Pure
    *ForCausalLM models (Llama) have neither.
    """
    if getattr(cfg, "text_config", None) is not None:
        return True
    archs = getattr(cfg, "architectures", None) or []
    return any("ForConditionalGeneration" in str(a) for a in archs)


def detect(base_model: str) -> ArchProfile:
    """Read the config at ``base_model`` and return its ArchProfile.

    Config-only (no model load, no GPU). Raises ValueError for an unsupported
    model_type so an unknown family fails loudly instead of silently mis-configuring.
    """
    import os
    kw = {"trust_remote_code": True}
    if os.path.isdir(base_model):  # local path → never hit network
        kw["local_files_only"] = True
    cfg = AutoConfig.from_pretrained(base_model, **kw)
    model_type = getattr(cfg, "model_type", "") or ""
    fam = _family_of(model_type)
    if fam is None:
        raise ValueError(
            f"Unsupported model_type={model_type!r} at {base_model}. "
            f"Add an entry to arch_profile._FAMILY_TABLE (supported: "
            f"{sorted(_FAMILY_TABLE)}). Needed: lora targets, reasoning parser, "
            f"thinking suppression, and whether vLLM serves it VL-nested."
        )
    is_vl = _is_vl_nested(cfg)
    return ArchProfile(
        family=fam["family"],
        lora_targets=fam.get("lora_targets", list(STANDARD_TARGETS)),
        reasoning_parser=fam["reasoning_parser"],
        language_model_only=is_vl,
        vl_rename=is_vl,
        suppress_thinking=fam["suppress_thinking"],
        text_only_load=fam["text_only_load"],
        max_model_len=fam.get("max_model_len"),
    )


# ---- shell-emit for the bash driver ----
def _emit_shell(p: ArchProfile) -> str:
    """Emit eval-able shell assignments for serve/infer flags (empty string = off)."""
    rparser = f"--reasoning-parser {p.reasoning_parser}" if p.reasoning_parser else ""
    lmonly = "--language-model-only" if p.language_model_only else ""
    vlrename = "--vl-rename" if p.vl_rename else ""
    mlen = f"--max-model-len {p.max_model_len}" if p.max_model_len else ""
    return (
        f"FAMILY={p.family!r}\n"
        f"RPARSER={rparser!r}\n"
        f"LMONLY={lmonly!r}\n"
        f"VLRENAME={vlrename!r}\n"
        f"MLEN={mlen!r}\n"
        f"SUPPRESS_THINKING={1 if p.suppress_thinking else 0}\n"
        f"TEXT_ONLY_LOAD={1 if p.text_only_load else 0}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect a base model's LoRA/serve profile.")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--emit-shell", action="store_true",
                    help="print eval-able shell vars (FAMILY/RPARSER/LMONLY/VLRENAME/…)")
    ap.add_argument("--json", action="store_true", help="print full profile as JSON")
    args = ap.parse_args()
    p = detect(args.base_model)
    if args.emit_shell:
        sys.stdout.write(_emit_shell(p))
    if args.json:
        d = asdict(p)
        sys.stdout.write(json.dumps(d, indent=2) + "\n")
    if not args.emit_shell and not args.json:
        # default: human-readable summary
        sys.stdout.write(
            f"[arch_profile] {args.base_model}\n"
            f"  family            = {p.family}\n"
            f"  lora_targets      = {p.lora_targets}\n"
            f"  reasoning_parser  = {p.reasoning_parser}\n"
            f"  language_model_only = {p.language_model_only}\n"
            f"  vl_rename         = {p.vl_rename}\n"
            f"  suppress_thinking = {p.suppress_thinking}\n"
            f"  text_only_load    = {p.text_only_load}\n"
            f"  max_model_len     = {p.max_model_len}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
