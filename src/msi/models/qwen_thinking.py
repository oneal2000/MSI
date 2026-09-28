"""Apply Qwen thinking controls when LoRA request names hide the model family."""
import sragents.llm as _llm


def _qwen3_extra_body(_model: str, thinking: bool = False):
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def apply_for(base_model: str) -> None:
    """Patch SR-Agents only for a Qwen3-family process.

    vLLM adapter requests use a skill ID as the model name, so SR-Agents cannot
    infer Qwen's chat-template option from that request name.  Phi and Llama
    processes must retain the upstream model-specific behavior.
    """
    if "qwen3" not in base_model.lower().rsplit("/", 1)[-1]:
        return
    _llm.get_extra_body = _qwen3_extra_body
    import sragents.infer.engines.react as _r
    import sragents.infer.engines.direct as _d
    import sragents.infer.engines.progressive_disclosure as _p
    _r.get_extra_body = _qwen3_extra_body
    _d.get_extra_body = _qwen3_extra_body
    _p.get_extra_body = _qwen3_extra_body
