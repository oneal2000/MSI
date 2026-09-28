"""Small, dependency-free helpers shared across release commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid


def atomic_json(path: str | Path, payload: Any) -> None:
    """Atomically replace *path* with a UTF-8 JSON document.

    The temporary file is unique and lives beside the destination, so separate
    workers cannot overwrite one another's temporary output and ``os.replace``
    remains atomic on the shared filesystem.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
