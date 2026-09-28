"""Constants fixed by the paper protocol.

Stage YAML files contain experiment and hardware choices.  Dataset-universe
facts and the paper's output-token ceilings live here so every public command
uses the same definition while still allowing YAML or CLI overrides.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


DATASET_SKILL_COUNTS: Mapping[str, int] = {
    "theoremqa": 320,
    "medcalcbench": 55,
    "logicbench": 19,
    "toolqa": 14,
}
FINAL_DATASETS = tuple(DATASET_SKILL_COUNTS)
SKILL_UNIVERSE_SIZE = sum(DATASET_SKILL_COUNTS.values())

# The synthetic-data teacher uses Qwen's explicit thinking mode for task and
# non-ReAct trajectory generation.  Evaluation remains a separate non-thinking
# judge, while ToolQA's benchmark ReAct loop keeps thinking disabled to match
# the established data generation protocol and its newline-delimited
# Thought/Action contract.
TEACHER_ENABLE_THINKING = True
TOOLQA_REACT_ENABLE_THINKING = False
TEACHER_REQUEST_MAX_TOKENS = 32768

# Generation limits apply to the visible, trainable trajectory after the
# teacher's hidden reasoning has been separated by the serving layer.
GENERATION_OUTPUT_MAX_TOKENS: Mapping[str, int] = {
    "theoremqa": 10240,
    "medcalcbench": 8192,
    "logicbench": 8192,
    "toolqa": 6144,
}
INFERENCE_MAX_TOKENS: Mapping[str, int] = {
    "theoremqa": 10240,
    "medcalcbench": 8192,
    "logicbench": 8192,
    "toolqa": 16384,
}


def inspect_skill_universe(rows: Iterable[dict]) -> tuple[list[str], dict[str, int], bool]:
    """Return ordered IDs, per-dataset counts, and protocol validity."""
    materialized = list(rows)
    skill_ids = [str(row.get("skill_id", "")) for row in materialized]
    counts = dict(Counter(skill_id.rsplit("_", 1)[0] for skill_id in skill_ids))
    valid = (
        len(materialized) == SKILL_UNIVERSE_SIZE
        and len(set(skill_ids)) == SKILL_UNIVERSE_SIZE
        and counts == dict(DATASET_SKILL_COUNTS)
    )
    return skill_ids, counts, valid


def token_defaults(schema: str | None, dataset: str | None) -> dict[str, int]:
    """Return the lowest-precedence token default for a stage and dataset."""
    if dataset not in DATASET_SKILL_COUNTS:
        return {}
    if schema == "msi.generate":
        return {
            "max_tokens": TEACHER_REQUEST_MAX_TOKENS,
            "output_max_tokens": GENERATION_OUTPUT_MAX_TOKENS[dataset],
        }
    if schema == "msi.evaluate":
        return {"max_tokens": INFERENCE_MAX_TOKENS[dataset]}
    if schema == "msi.train":
        return {"val_max_tokens": INFERENCE_MAX_TOKENS[dataset]}
    return {}
