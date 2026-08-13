"""Benchmark one arbitrary artifact with the pinned official evaluator.

This is the tuning/diagnostic path.  It never updates ``coverage.json``.  Use
``run_all.py`` for a score-bearing run after the chip profile and environment
lock have been verified.  Keeping this path separate prevents a quick tuning
run from being mistaken for a competition result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator_contract import (  # noqa: E402
    EVALUATION_MODE,
    official_command,
    parse_official_pass,
    verify_official_evaluator,
)
from prepare_case import prepare_pair  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS, validate_all_profiles  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one artifact through the pinned official KernelSwift evaluator"
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--task", choices=TASK_KEYS, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.timeout < 0:
        raise SystemExit("warmup >= 0, repeat > 0 and timeout >= 0 are required")
    validate_all_profiles()
    artifact = args.artifact.resolve()
    reference = (args.reference or ROOT / "reference" / f"{args.task}.py").resolve()
    bench = args.bench.resolve()
    for path, label in ((artifact, "artifact"), (reference, "reference"), (bench, "evaluator")):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    evaluator = verify_official_evaluator(bench)
    if not evaluator["verified"]:
        raise SystemExit(f"official evaluator verification failed: {evaluator['detail']}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    preparation = prepare_pair(reference, artifact, prepared, args.chip)
    command = official_command(
        bench,
        prepared / "reference.py",
        prepared / "artifact.py",
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        warmup=args.warmup,
        repeat=args.repeat,
        python=sys.executable,
    )
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            text=True,
            errors="replace",
            capture_output=True,
            timeout=args.timeout or None,
            check=False,
        )
        returncode: int | None = completed.returncode
        output_text = completed.stdout + (
            "\n[stderr]\n" + completed.stderr if completed.stderr else ""
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output_text = stdout + ("\n[stderr]\n" + stderr if stderr else "")
        output_text += f"\n[harness] timed out after {args.timeout} seconds\n"

    (output / "evaluator.log").write_text(output_text, encoding="utf-8")
    metrics = parse_official_pass(output_text)
    official_pass = returncode == 0 and metrics is not None and not timed_out
    summary = {
        "schema_version": 1,
        "kind": "official_evaluator_diagnostic",
        "evaluation_mode": EVALUATION_MODE,
        "official_pass": official_pass,
        "chip_key": args.chip,
        "task_key": args.task,
        "source_commit": git_commit(),
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "reference": str(reference),
        "reference_sha256": sha256(reference),
        "evaluator": evaluator,
        "command": command,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "returncode": returncode,
        "timed_out": timed_out,
        "metrics": metrics,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "preparation": preparation,
        "log": str(output / "evaluator.log"),
        "measured_at_utc": utc_now(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if official_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
