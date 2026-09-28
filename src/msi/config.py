"""Typed configuration loading and command-line overrides.

Stage configuration files are intentionally independent.  Every scalar can be
overridden with ``--set dotted.path=value``; the public CLI additionally maps
the frequently used fields to ordinary flags such as ``--batch-size``.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from msi import REPO_ROOT
from msi.protocol import token_defaults


class ConfigError(ValueError):
    """Raised when a stage configuration is malformed."""


def _yaml_module():
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - production dependency
        raise ConfigError(
            "PyYAML is required to read stage configs; install the project "
            "environment with `pip install -e .`"
        ) from error
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")
    yaml = _yaml_module()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root must be a mapping: {source}")
    payload = copy.deepcopy(payload)
    payload["_config_file"] = str(source)
    return payload


def parse_value(text: str) -> Any:
    """Parse one CLI value with YAML scalar/list semantics."""
    yaml = _yaml_module()
    return yaml.safe_load(text)


def _parts(path: str) -> list[str]:
    values = [part for part in path.split(".") if part]
    if not values:
        raise ConfigError("empty configuration key")
    return values


def set_value(config: dict[str, Any], path: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = _parts(path)
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ConfigError(f"cannot set {path!r}: {part!r} is not a mapping")
        cursor = existing
    cursor[parts[-1]] = value


def get_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in _parts(path):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def apply_assignments(config: dict[str, Any], assignments: Iterable[str]) -> None:
    for item in assignments:
        if "=" not in item:
            raise ConfigError(f"expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        set_value(config, key, parse_value(raw))


def apply_scoped(
    config: dict[str, Any], scope: str, assignments: Iterable[str]
) -> None:
    """Apply ``NAME.KEY=VALUE`` under ``overrides.<scope>.<NAME>``."""
    for item in assignments:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ConfigError(f"expected NAME.KEY=VALUE, got {item!r}")
        left, raw = item.split("=", 1)
        declared = set(config.get("overrides", {}).get(scope, {}))
        if scope == "models":
            declared.update(config.get("models", {}))
            current = config.get("model", {}).get("name")
            if current:
                declared.add(str(current))
        matches = [name for name in declared if left.startswith(name + ".")]
        if matches:
            name = max(matches, key=len)
            key = left[len(name) + 1:]
        else:
            name, key = left.split(".", 1)
        scoped = config.setdefault("overrides", {}).setdefault(scope, {})
        target = scoped.setdefault(name, {})
        if not isinstance(target, dict):
            raise ConfigError(f"override scope {scope}.{name} is not a mapping")
        set_value(target, key, parse_value(raw))


def effective_values(
    config: dict[str, Any], *, model: str | None = None,
    dataset: str | None = None, skill: str | None = None,
) -> dict[str, Any]:
    """Resolve protocol -> default -> model -> dataset -> skill precedence."""
    result = token_defaults(config.get("schema"), dataset)
    result = deep_merge(result, config.get("defaults", {}))
    overrides = config.get("overrides", {})
    for scope, name in (("models", model), ("datasets", dataset), ("skills", skill)):
        values = overrides.get(scope, {}).get(name, {}) if name else {}
        if values:
            result = deep_merge(result, values)
    return result


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand(value: Any, variables: dict[str, str]) -> Any:
    """Expand environment variables and ``{run_id}``-style placeholders."""
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    try:
        return expanded.format_map(variables)
    except KeyError as error:
        raise ConfigError(f"unknown path placeholder {error.args[0]!r} in {value!r}") from error


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or REPO_ROOT) / path
    return path.resolve()


def validate_stage(config: dict[str, Any], stage: str) -> None:
    schema = config.get("schema")
    expected = f"msi.{stage}"
    if schema != expected:
        raise ConfigError(f"{stage} config schema must be {expected!r}, got {schema!r}")
    if not isinstance(config.get("paths"), dict):
        raise ConfigError(f"{stage} config requires a paths mapping")
    if not isinstance(config.get("defaults"), dict):
        raise ConfigError(f"{stage} config requires a defaults mapping")


def write_resolved(config: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    yaml = _yaml_module()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def stable_json(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
