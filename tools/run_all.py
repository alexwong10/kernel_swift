"""Run the official DLBlas KernelSwift evaluator for all ten submissions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--chip", required=True, help="Stable result directory name")
    parser.add_argument("--only", help="Two-digit task number, e.g. 01")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bench = args.bench.resolve()
    if not bench.is_file():
        raise SystemExit(f"official evaluator not found: {bench}")
    result_dir = ROOT / "results" / args.chip
    result_dir.mkdir(parents=True, exist_ok=True)

    references = sorted((ROOT / "reference").glob("[0-9][0-9]_*.py"))
    if args.only:
        references = [path for path in references if path.name.startswith(args.only + "_")]
    if not references:
        raise SystemExit("no matching tasks")

    summary: list[dict[str, object]] = []
    overall_ok = True
    for reference in references:
        optimized = ROOT / "triton_kernels" / reference.name
        command = [
            sys.executable,
            str(bench),
            "--v0_file",
            str(reference),
            "--v1_file",
            str(optimized),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--full-traceback",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        log = completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
        (result_dir / f"{reference.stem}.log").write_text(log, encoding="utf-8")
        passed = completed.returncode == 0
        overall_ok &= passed
        summary.append(
            {
                "task": reference.stem,
                "passed": passed,
                "returncode": completed.returncode,
                "log": f"{reference.stem}.log",
            }
        )
        print(("PASS" if passed else "FAIL"), reference.stem)

    payload = {
        "chip": args.chip,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bench": str(bench),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "atol": args.atol,
        "rtol": args.rtol,
        "tasks": summary,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

