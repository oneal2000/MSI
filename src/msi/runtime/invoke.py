"""Internal stage invocation with auditable command rendering."""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager


@contextmanager
def _argv(arguments: list[str]):
    previous = sys.argv
    sys.argv = [previous[0], *arguments]
    try:
        yield
    finally:
        sys.argv = previous


class StageRunner:
    def __init__(self, *, dry_run: bool = False, python: str | None = None):
        self.dry_run = dry_run
        self.python = python or sys.executable

    def run(
        self, module: str, arguments: list[object], *, isolated: bool = False,
        allowed_codes: tuple[int, ...] = (0,), environment: dict[str, str] | None = None,
    ) -> int:
        argv = [str(item) for item in arguments]
        command = [self.python, "-m", module, *argv]
        print("+ " + shlex.join(command), flush=True)
        if self.dry_run:
            return 0
        if isolated:
            env = os.environ.copy()
            env.update(environment or {})
            code = subprocess.run(command, env=env).returncode
            if code not in allowed_codes:
                raise SystemExit(code)
            return code
        target = importlib.import_module(module)
        try:
            with _argv(argv):
                result = target.main()
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
            if code not in allowed_codes:
                raise
            return code
        code = int(result or 0)
        if code not in allowed_codes:
            raise SystemExit(code)
        return code
