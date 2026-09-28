"""Small helpers shared by the four public pipelines."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from msi import REPO_ROOT
from msi.config import effective_values
from msi.protocol import FINAL_DATASETS



def path(config: dict[str, Any], name: str, default: str | None = None) -> Path:
    raw = config.get("paths", {}).get(name, default)
    if raw is None:
        raise ValueError(f"missing paths.{name}")
    value = Path(str(raw)).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def datasets(config: dict[str, Any]) -> list[str]:
    selected = config.get("_selection", {}).get("datasets") or list(FINAL_DATASETS)
    unknown = set(selected) - set(FINAL_DATASETS)
    if unknown:
        raise ValueError(f"unsupported datasets: {sorted(unknown)}")
    return list(dict.fromkeys(selected))


def skills(config: dict[str, Any]) -> list[str]:
    return list(config.get("_selection", {}).get("skills") or [])


def dataset_skill_ids(config: dict[str, Any]) -> list[str]:
    """Return corpus skills belonging to the selected datasets."""
    rows = json.loads(path(config, "corpus").read_text(encoding="utf-8"))
    universe = [str(row["skill_id"]) for row in rows]
    chosen_datasets = set(datasets(config))
    return [skill for skill in universe
            if skill.rsplit("_", 1)[0] in chosen_datasets]


def selected_skill_ids(config: dict[str, Any]) -> list[str]:
    """Resolve a generic dataset/skill selection against the declared corpus."""
    universe = dataset_skill_ids(config)
    universe_set = set(universe)
    chosen_skills = skills(config)
    unknown = set(chosen_skills) - universe_set
    if unknown:
        raise ValueError(f"unknown skills: {sorted(unknown)}")
    if chosen_skills:
        return list(dict.fromkeys(chosen_skills))
    return universe


def selection_scope(config: dict[str, Any]) -> str:
    """Return a stable artifact scope for a dataset or exact-skill selection."""
    chosen_skills = sorted(set(skills(config)))
    if not chosen_skills:
        return "+".join(datasets(config))
    if len(chosen_skills) == 1:
        return chosen_skills[0]
    digest = hashlib.sha256("\n".join(chosen_skills).encode("utf-8")).hexdigest()[:12]
    return f"skills-{digest}"


def settings(
    config: dict[str, Any], *, dataset: str | None = None,
    skill: str | None = None, model: str | None = None,
) -> dict[str, Any]:
    model = (
        model
        or config.get("model", {}).get("name")
        or config.get("_selection", {}).get("model")
    )
    values = effective_values(config, model=model, dataset=dataset, skill=skill)
    values.update(config.get("_direct_defaults", {}))
    return values


def repeated(flag: str, values) -> list[str]:
    result: list[str] = []
    for value in values:
        result += [flag, str(value)]
    return result
