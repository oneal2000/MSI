"""Shared request settings for the synthetic-data teacher."""

from __future__ import annotations

from sragents.llm import get_extra_body

from msi.protocol import TEACHER_ENABLE_THINKING


def teacher_extra_body(
    model: str, *, thinking: bool = TEACHER_ENABLE_THINKING,
) -> dict | None:
    """Return explicit thinking and anti-repetition settings for the teacher."""
    extra = dict(get_extra_body(model, thinking=thinking) or {})
    if "qwen3" in model.casefold():
        extra.update({
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        })
    return extra or None
