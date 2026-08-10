"""Run all ten candidates on a MetaX C500 and archive target-device evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


TASKS = (
    "01_grouped_topk",
    "02_fused_moe",
    "03_flex_attention",
    "04_splade_sparse_pooler",
    "05_music_flamingo_rotary_embedding",
    "06_mm_encoder_attention",
    "07_mhc_post",
    "08_hc_split_sinkhorn",
    "09_centre_random_augmentation",
    "10_head_compute_mix_bwd",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "task" in payload:
                return payload
    return {"status": "failed", "error": "runner emitted no JSON result"}


def probe_environment() -> dict[str, object]:
    environment: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "maca_path": os.environ.get("MACA_PATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }
    try:
        import torch

        torch.cuda.set_device(0)
        environment.update(
            {
                "torch": torch.__version__,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except Exception as exc:
        environment["device_probe_error"] = repr(exc)
    try:
        import triton

        environment["triton"] = triton.__version__
    except Exception as exc:
        environment["triton_probe_error"] = repr(exc)
    try:
        completed = subprocess.run(["mx-smi", "-L"], text=True, capture_output=True, check=False)
        environment["mx_smi_returncode"] = completed.returncode
        environment["mx_smi"] = completed.stdout + completed.stderr
    except Exception as exc:
        environment["mx_smi_error"] = repr(exc)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {task: sha256(args.artifact / f"{task}.py") for task in TASKS}
    environment = probe_environment()
    environment["source_commit"] = args.source_commit
    environment["artifact_hashes"] = artifact_hashes

    results = []
    for task in TASKS:
        completed = subprocess.run(
            [
                sys.executable,
                str(args.runner),
                "--artifact",
                str(args.artifact),
                "--reference",
                str(args.reference),
                "--task",
                task,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        log_path = args.output / f"{task}.log"
        log_path.write_text(
            completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        result = parse_result(completed.stdout)
        result.update(
            {
                "returncode": completed.returncode,
                "log": log_path.name,
                "artifact_sha256": artifact_hashes[task],
            }
        )
        results.append(result)

    summary = {
        "schema_version": 1,
        "kind": "offline_target_chip_validation",
        "chip_key": "metax_c500",
        "environment": environment,
        "results": results,
        "passed": sum(item.get("status") == "passed" for item in results),
        "total": len(TASKS),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
