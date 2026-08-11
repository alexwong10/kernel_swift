"""Run all ten candidates on an Enflame S60 and archive target evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    }
    try:
        import torch_gcu  # noqa: F401
        import torch

        backend = torch.gcu
        backend.set_device(0)
        environment.update(
            {
                "torch": torch.__version__,
                "torch_gcu": getattr(torch_gcu, "__version__", "unknown"),
                "triton": __import__("triton").__version__,
                "triton_gcu": getattr(__import__("triton_gcu"), "__version__", "unknown"),
                "device_count": backend.device_count(),
                "device_name": backend.get_device_name(0),
                "gcu_available": bool(backend.is_available()),
            }
        )
    except Exception as exc:
        environment["device_probe_error"] = repr(exc)
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
            completed.stdout
            + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
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
        "chip_key": "enflame_s60",
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
