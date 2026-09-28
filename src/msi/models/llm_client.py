"""OpenAI client factory + chat helper.

Thinking config is NOT applied here (qwen_thinking.py remains the student-server
thinking authority; the teacher resolves its own extra_body). Kept student/teacher
neutral so generate_training_data (122B teacher) can reuse it.
"""
import threading
import time

from openai import OpenAI
from sragents.llm import create_llm_client


DEFAULT_REQUEST_TIMEOUT_SECONDS = 5400.0


def request_timeout() -> float:
    """Return the shared long-generation timeout, with an environment override."""
    import os

    return float(os.environ.get("SRAGENTS_REQ_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_SECONDS))

# Finish reason of the most recent _chat call, per thread. 'length' means the
# response was truncated at max_tokens — such an output is a mid-reasoning
# amputation and must never be accepted as a training trajectory.
_finish = threading.local()


def last_finish_reason() -> str | None:
    """Return the finish_reason of the last _chat call on the current thread."""
    return getattr(_finish, "reason", None)


def create_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    if not base_url:
        raise ValueError("an explicit OpenAI-compatible base_url is required")
    # Keep SR-Agents' endpoint/key handling, then apply this repository's
    # shared timeout so queued generation, validation, and inference requests
    # honor the same operator override.
    client = create_llm_client(api_base=base_url, api_key=api_key)
    return client.with_options(timeout=request_timeout())


def _chat(client: OpenAI, model: str, system: str, messages: list[dict],
          max_tokens: int = 16384, retries: int = 3, extra_body: dict | None = None,
          temperature: float = 0.7) -> str:
    all_msgs = []
    if system:
        all_msgs.append({"role": "system", "content": system})
    all_msgs.extend(messages)
    for attempt in range(retries):
        try:
            start = time.time()
            print(f"  [API] {model} max_tokens={max_tokens} started at {time.strftime('%H:%M:%S')}...", flush=True)
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens, messages=all_msgs,
                temperature=temperature, extra_body=extra_body,
            )
            elapsed = time.time() - start
            reason = (resp.choices[0].finish_reason or "stop") if resp.choices else "stop"
            _finish.reason = reason
            warn = " ⚠️TRUNCATED" if reason == "length" else ""
            print(f"  [API] Done in {elapsed:.1f}s, tokens={resp.usage.total_tokens if resp.usage else 'N/A'}, finish={reason}{warn}", flush=True)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  _chat retry {attempt+1}/{retries} after {type(e).__name__}: {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise
