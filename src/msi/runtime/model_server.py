#!/usr/bin/env python3
"""Start a validation/inference server with model-correct flags."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from msi.models.arch_profile import detect
from msi.runtime.gpus import GPUSelectionError, physical_index
from msi.training.provenance import assert_shared_output


def vllm_executable_for(python_executable: str) -> Path:
    """Return vLLM beside the admitted interpreter without resolving venv links.

    ``venv/bin/python`` is commonly a symlink to a shared base interpreter.
    Resolving that symlink escapes the admitted environment and looks for vLLM
    beside the base Python even though the console script is in ``venv/bin``.
    """
    return Path(python_executable).absolute().parent / "vllm"


def verified_vllm_cuda_devices(
    python_executable: str, selected_devices: list[str],
) -> tuple[list[str], list[str]]:
    """Map UUIDs to physical indices and verify CUDA sees them in that order."""
    if not selected_devices or len(selected_devices) != len(set(selected_devices)):
        raise GPUSelectionError("the model server requires distinct selected GPUs")
    resolved = [physical_index(device) for device in selected_devices]
    indices = [index for index, _uuid in resolved]
    expected_uuids = [uuid for _index, uuid in resolved]
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(indices),
    })
    probe = subprocess.run(
        [python_executable, "-c", (
            "import torch; "
            "print('\\n'.join(str(getattr(torch.cuda.get_device_properties(i), "
            "'uuid', '')) for i in range(torch.cuda.device_count())))"
        )],
        text=True, capture_output=True, timeout=30, env=environment,
    )
    actual = [line.strip().lower() for line in probe.stdout.splitlines() if line.strip()]
    expected = [uuid.removeprefix("GPU-").lower() for uuid in expected_uuids]
    if probe.returncode != 0 or actual != expected:
        raise GPUSelectionError(
            "CUDA physical-index verification failed: "
            f"selected={selected_devices}, indices={indices}, expected={expected_uuids}, "
            f"actual={actual!r}, stderr={probe.stderr.strip()!r}"
        )
    return indices, expected_uuids


def verified_vllm_cuda_device(
    python_executable: str, selected_device: str,
) -> tuple[str, str]:
    """Backward-compatible single-device wrapper."""
    indices, uuids = verified_vllm_cuda_devices(
        python_executable, [selected_device],
    )
    return indices[0], uuids[0]


def socket_safe_temp_dir(requested: Path, shared_root: Path) -> Path:
    """Keep vLLM's UUID-suffixed IPC socket below Linux's 107-byte limit."""
    requested = requested.resolve()
    # vLLM appends '/' plus a 36-character UUID. Leave several bytes of margin
    # for implementation details while keeping every temp file on shared disk.
    if len(str(requested)) + 37 <= 100:
        return requested
    digest = hashlib.sha256(str(requested).encode("utf-8")).hexdigest()[:16]
    shortened = (shared_root.resolve() / "work" / ".tmp" / digest).resolve()
    if len(str(shortened)) + 37 > 100:
        raise SystemExit(
            "shared root is too long for vLLM IPC sockets even after shortening: "
            f"{shortened}"
        )
    return shortened


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--gpu", default="3",
                        help="Comma-separated physical GPU indices or UUIDs")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert_shared_output(args.shared_root)
    profile = detect(args.model_path)
    vllm_executable = vllm_executable_for(sys.executable)
    if not vllm_executable.is_file():
        raise SystemExit(f"vLLM executable is not beside the admitted Python: {vllm_executable}")
    selected_gpus = [value.strip() for value in args.gpu.split(",") if value.strip()]
    if not selected_gpus or len(selected_gpus) != len(set(selected_gpus)):
        raise SystemExit("--gpu requires one or more distinct comma-separated devices")
    command = [
        str(vllm_executable), "serve", args.model_path,
        "--served-model-name", args.served_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16", "--gpu-memory-utilization",
        str(args.gpu_memory_utilization), "--max-num-seqs", "512",
        "--enable-prefix-caching", "--trust-remote-code",
    ]
    if args.lora:
        command += ["--enable-lora", "--max-loras", "16", "--max-lora-rank", "64"]
    if len(selected_gpus) > 1:
        command += ["--tensor-parallel-size", str(len(selected_gpus))]
    if profile.reasoning_parser:
        command += ["--reasoning-parser", profile.reasoning_parser]
    if profile.language_model_only:
        command += ["--language-model-only"]
    max_model_len = args.max_model_len or profile.max_model_len
    if max_model_len is not None:
        if max_model_len < 1:
            raise SystemExit("--max-model-len must be positive")
        command += ["--max-model-len", str(max_model_len)]
    if args.dry_run:
        print("SELECTED_GPU=" + ",".join(selected_gpus), " ".join(command), flush=True)
        return
    cache_root = Path(args.cache_dir).resolve()
    requested_temp_root = Path(args.temp_dir).resolve()
    temp_root = socket_safe_temp_dir(
        requested_temp_root, Path(args.shared_root).resolve()
    )
    assert_shared_output(cache_root)
    assert_shared_output(temp_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    cuda_indices, verified_uuids = verified_vllm_cuda_devices(
        sys.executable, selected_gpus,
    )
    print(
        f"SELECTED_GPU={','.join(selected_gpus)} "
        f"CUDA_VISIBLE_DEVICES={','.join(cuda_indices)} "
        f"VERIFIED_UUID={','.join(verified_uuids)} TMPDIR={temp_root}",
        " ".join(command), flush=True,
    )
    os.environ.update({
        "HF_HOME": str(cache_root / "huggingface"),
        "XDG_CACHE_HOME": str(cache_root),
        "TRANSFORMERS_CACHE": str(cache_root / "huggingface"),
        "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        "TORCH_HOME": str(cache_root / "torch"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "CUDA_CACHE_PATH": str(cache_root / "cuda"),
        "NUMBA_CACHE_DIR": str(cache_root / "numba"),
        "TMPDIR": str(temp_root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    })
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_indices)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
