"""Skill corpus access: load_skill + dataset inference from skill_id."""
import json

from msi.models.config import CORPUS_PATH


def load_skill(skill_id: str) -> dict:
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    return {s["skill_id"]: s for s in corpus}.get(skill_id)
def get_dataset_from_skill_id(skill_id: str) -> str:
    prefixes = ["medcalcbench", "theoremqa", "logicbench", "toolqa"]
    for p in prefixes:
        if skill_id.startswith(p + "_") or skill_id.startswith(p):
            return p
    return skill_id.split("_")[0]
