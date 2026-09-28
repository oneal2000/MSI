"""Load the TEXT-only causal LM from a multimodal checkpoint (Gemma4 fix).

Problem
-------
Some checkpoints ship as multimodal ``*ForConditionalGeneration`` (e.g. Gemma4-E2B-it,
``architectures=["Gemma4ForConditionalGeneration"]``). ``AutoModelForCausalLM`` then
loads the FULL model incl. vision/audio towers, whose projections are wrapped in
``Gemma4ClippableLinear`` — and PEFT aborts ("Target module Gemma4ClippableLinear is
not supported") before ever reaching the text layers. Instantiating the text
``*ForCausalLM`` directly via ``from_pretrained`` also fails: the checkpoint stores
the LM nested under ``model.language_model.*`` while the text class expects flat
``model.layers.*`` → every key is UNEXPECTED → random init.

Fix (mirrors vLLM's own ``Gemma4ForCausalLM.hf_to_vllm_mapper``)
---------------------------------------------------------------
Build the text ``*ForCausalLM`` from ``config.text_config``, then load the checkpoint
after stripping the ``model.language_model.`` prefix → ``model.``. The text path's
projections are plain ``nn.Linear`` (no ClippableLinear), so PEFT LoRA works.
Tied ``lm_head`` (``tie_word_embeddings=True``) is retied after load.

The trained adapter keys are FLAT (``base_model.model.model.layers.*``), so vLLM must
serve this model as the flat text class (``--language-model-only`` → ``Gemma4ForCausalLM``)
for the adapter to mount WITHOUT ``vl_rename`` — see arch_profile + smoke gate.
"""
from __future__ import annotations

import glob
import os

from safetensors.torch import load_file

# text model_type -> causal-LM class (add more text_only families here)
def _text_cls(model_type: str):
    mt = (model_type or "").lower()
    if mt in ("gemma4_text", "gemma4"):
        from transformers import Gemma4ForCausalLM
        return Gemma4ForCausalLM
    raise ValueError(f"text_only_load: no text causal-LM class for model_type={model_type!r}")


def load_text_causal_lm(base_model: str, dtype=None, nested_prefix: str = "model.language_model."):
    """Build the text *ForCausalLM from the multimodal checkpoint + remap nested keys.

    Args:
        base_model: path to the (multimodal) checkpoint dir.
        dtype: torch dtype for the instantiated model.
        nested_prefix: checkpoint prefix holding the LM (Gemma4: ``model.language_model.``).
    Returns: a text ``*ForCausalLM`` with real weights loaded + weights tied.
    """
    import torch
    from transformers import AutoConfig

    local = os.path.isdir(base_model)
    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True, local_files_only=local)
    tcfg = getattr(cfg, "text_config", None)
    if tcfg is None:
        raise ValueError(f"{base_model} has no text_config — not a multimodal checkpoint; text_only_load not applicable")
    cls = _text_cls(tcfg.model_type)
    # *ForCausalLM.__init__ takes only the config (no dtype kwarg); build then cast.
    if dtype is not None:
        try:
            tcfg.torch_dtype = dtype  # hint so init allocates in-target dtype where respected
        except Exception:
            pass
    model = cls(tcfg)
    if dtype is not None:
        model = model.to(dtype)

    flat = nested_prefix.replace("language_model.", "", 1)  # "model.language_model." -> "model."
    sd = {}
    for f in sorted(glob.glob(os.path.join(base_model, "*.safetensors"))):
        for k, v in load_file(f).items():
            if k.startswith(nested_prefix):
                sd[k.replace(nested_prefix, flat, 1)] = v
    if not sd:
        raise ValueError(f"text_only_load: no '{nested_prefix}*' tensors found in {base_model}")
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.tie_weights()
    print(f"[text_only_load] {cls.__name__} from {base_model}: loaded {len(sd)} tensors | "
          f"missing={len(miss)} (expect tied lm_head) unexpected={len(unexp)} (ClippableLinear/PLE buffers)", flush=True)
    return model
