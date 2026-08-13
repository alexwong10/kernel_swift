"""Convenient entry point for the score-bearing official evaluation path.

The wrapper locates or accepts the DLBlas ``auto_bench.py`` and delegates all
measurement work to ``run_all.py``.  It exists to make the distinction between
official evaluation and legacy target-device smoke tests explicit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator_contract import EVALUATOR_COMMIT, EVALUATOR_URL, verify_official_evaluator  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402


def discover_evaluator() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("KERNELSWIFT_AUTO_BENCH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "third_party" / "DLBlas" / "benchmarks" / "ks" / "auto_bench.py",
            ROOT.parent / "DLBlas" / "benchmarks" / "ks" / "auto_bench.py",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned official KernelSwift evaluator through run_all.py"
    )
    parser.add_argument("--bench", type=Path, help="path to DLBlas/benchmarks/ks/auto_bench.py")
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--task", choices=TASK_KEYS)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-evaluator-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bench = (args.bench or discover_evaluator())
    if bench is None:
        raise SystemExit(
            "official auto_bench.py not found; clone DLBlas at the pinned commit "
            f"{EVALUATOR_COMMIT} or set KERNELSWIFT_AUTO_BENCH.\n{EVALUATOR_URL}"
        )
    bench = bench.resolve()
    verification = verify_official_evaluator(bench)
    if not verification["verified"] and not args.allow_evaluator_mismatch:
        raise SystemExit(f"official evaluator verification failed: {verification['detail']}")

    command = [
        sys.executable,
        str(ROOT / "tools" / "run_all.py"),
        "--bench",
        str(bench),
        "--chip",
        args.chip,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--timeout",
        str(args.timeout),
    ]
    if args.task:
        command.extend(["--task", args.task])
    if args.diagnostic:
        command.append("--diagnostic")
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.allow_evaluator_mismatch:
        command.append("--allow-evaluator-mismatch")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
