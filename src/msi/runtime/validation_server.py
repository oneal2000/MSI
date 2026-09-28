"""Lifecycle management for the dedicated training-validation GPU."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _served(api_base: str, expected: str) -> bool:
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as response:
            payload = json.load(response)
        return expected in {row.get("id") for row in payload.get("data", [])}
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def managed_validation_server(
    *, python: str, model_path: str, served_name: str,
    gpu: str | list[str], port: int,
    shared_root: Path, cache_dir: Path, temp_dir: Path, log_path: Path,
    gpu_memory_utilization: float = 0.92, timeout: int = 2400,
    max_model_len: int | None = None,
    lora: bool = True,
    dry_run: bool = False,
):
    """Start, verify, and stop a pipeline-owned model server."""
    api_base = f"http://127.0.0.1:{port}/v1"
    gpu_argument = ",".join(gpu) if isinstance(gpu, list) else gpu
    command = [
        python, "-m", "msi.runtime.model_server",
        "--model-path", model_path, "--served-name", served_name,
        "--gpu", gpu_argument, "--port", str(port),
        "--shared-root", str(shared_root),
        "--cache-dir", str(cache_dir), "--temp-dir", str(temp_dir),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        command += ["--max-model-len", str(max_model_len)]
    if not lora:
        command += ["--no-lora"]
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        yield api_base
        return
    if _served(api_base, served_name):
        raise RuntimeError(
            f"{api_base} already serves {served_name}; pass --api-base to reuse it explicitly"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=os.environ.copy(),
        )
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"validation server exited {server.returncode}; see {log_path}"
                )
            if _served(api_base, served_name):
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"validation server did not become ready; see {log_path}")
        yield api_base
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


@contextmanager
def managed_server_pool(managers):
    """Enter and leave independent model servers concurrently."""
    managers = list(managers)
    entered = []
    try:
        with ThreadPoolExecutor(max_workers=len(managers)) as pool:
            futures = {pool.submit(manager.__enter__): manager for manager in managers}
            endpoints = {}
            errors = []
            for future in as_completed(futures):
                manager = futures[future]
                try:
                    endpoints[manager] = future.result()
                    entered.append(manager)
                except Exception as error:  # noqa: BLE001
                    errors.append(error)
            if errors:
                raise errors[0]
        yield [endpoints[manager] for manager in managers]
    finally:
        if entered:
            with ThreadPoolExecutor(max_workers=len(entered)) as pool:
                futures = [pool.submit(manager.__exit__, None, None, None)
                           for manager in reversed(entered)]
                for future in futures:
                    future.result()
