"""Run the pinned evaluator using only files contained in this package.

This is a local evaluator entrypoint. It never submits to KernelSwift and never
updates the formal coverage ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_case import prepare_pair


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PACKAGE_ROOT / "run"
DEFAULT_BENCH = RUN_ROOT / "auto_bench.py"
EVALUATOR_COMMIT = "9b5b3627a0f2e5e543ad9d05bf051308bafbd12c"
EVALUATOR_SHA256 = "357751a12552d1712ad5f66caa4e0fbd79d940b58a99342f83144fdfc9abb5db"
EVALUATOR_URL = (
    "https://github.com/DeepLink-org/DLBlas/blob/"
    f"{EVALUATOR_COMMIT}/benchmarks/ks/auto_bench.py"
)

TASKS = [
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
]
CHIPS = [
    "ascend_a2_910b",
    "metax_c500",
    "iluvatar_bi150",
    "enflame_s60",
    "thead_810e",
    "cambricon_mlu590_m9d",
    "hygon_bw1000",
    "moore_threads_ph100",
    "kunlun_p800",
    "nvidia_h200",
]
PASS_PATTERN = re.compile(
    r"PASS accuracy;\s*v0=(?P<reference_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"v1=(?P<optimized_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"speedup=(?P<speedup>[0-9.eE+-]+)x"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the package-local pinned DLBlas auto_bench evaluator"
    )
    parser.add_argument("--chip", choices=CHIPS, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--bench", type=Path, help="optional exact pinned evaluator override")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--results-root", type=Path, default=RUN_ROOT / "results")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="compatibility flag; all package runs are local-only and non-scoring",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluator_path(args: argparse.Namespace) -> Path:
    bench = (args.bench or DEFAULT_BENCH).resolve()
    if not bench.is_file():
        raise SystemExit(f"package evaluator not found: {bench}")
    actual = sha256(bench)
    if actual != EVALUATOR_SHA256:
        raise SystemExit(
            f"evaluator SHA-256 mismatch: {actual}; expected pinned {EVALUATOR_SHA256}"
        )
    return bench


def load_chip_runtime(chip: str) -> dict[str, Any]:
    path = RUN_ROOT / "chip_runtime.json"
    if not path.is_file():
        raise SystemExit(f"package runtime configuration not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload.get(chip)
    if not isinstance(runtime, dict):
        raise SystemExit(f"package runtime configuration missing chip: {chip}")
    return {"chip_key": chip, "runtime": runtime}


def task_command(
    bench: Path,
    prepared: Path,
    *,
    seed: int,
    atol: float,
    rtol: float,
    warmup: int,
    repeat: int,
) -> list[str]:
    return [
        sys.executable,
        str(bench),
        "--v0_file",
        str(prepared / "reference.py"),
        "--v1_file",
        str(prepared / "artifact.py"),
        "--seed",
        str(seed),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--full-traceback",
    ]


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.timeout < 0:
        raise SystemExit("warmup >= 0, repeat > 0 and timeout >= 0 are required")
    bench = evaluator_path(args)
    tasks = [args.task] if args.task else TASKS
    results_root = args.results_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = results_root / args.chip / f"{stamp}-{secrets.token_hex(4)}"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "package_root": str(PACKAGE_ROOT),
                    "evaluator": relative(bench),
                    "evaluator_commit": EVALUATOR_COMMIT,
                    "evaluator_sha256": sha256(bench),
                    "chip_key": args.chip,
                    "tasks": tasks,
                    "results_root": str(run_dir),
                    "platform_submission": False,
                    "formal_coverage_update": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    all_passed = True
    for task in tasks:
        task_dir = PACKAGE_ROOT / task
        reference = RUN_ROOT / "reference" / f"{task}.py"
        artifact = task_dir / "code" / f"{args.chip}.py"
        runtime = load_chip_runtime(args.chip)
        if not reference.is_file() or not artifact.is_file():
            raise SystemExit(f"package task files missing: {task}/{args.chip}")
        actual_hash = sha256(artifact)

        task_run_dir = run_dir / task
        task_run_dir.mkdir(parents=True, exist_ok=True)
        prepared_dir = task_run_dir / "prepared"
        prepare_pair(reference, artifact, prepared_dir, runtime)
        command = task_command(
            bench,
            prepared_dir,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        process_env = os.environ.copy()
        process_env["KERNELSWIFT_CHIP_KEY"] = args.chip
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=PACKAGE_ROOT,
                text=True,
                errors="replace",
                capture_output=True,
                timeout=args.timeout or None,
                env=process_env,
            )
            returncode: int | None = completed.returncode
            output = completed.stdout + (
                "\n[stderr]\n" + completed.stderr if completed.stderr else ""
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            output = stdout + ("\n[stderr]\n" + stderr if stderr else "")
            output += f"\n[package-harness] timed out after {args.timeout} seconds\n"

        log_path = task_run_dir / "evaluator.log"
        log_path.write_text(output, encoding="utf-8")
        match = PASS_PATTERN.search(output)
        metrics = (
            {
                "reference_ms": float(match.group("reference_ms")),
                "optimized_ms": float(match.group("optimized_ms")),
                "speedup": float(match.group("speedup")),
            }
            if match is not None
            else None
        )
        passed = returncode == 0 and metrics is not None
        all_passed &= passed
        summary: dict[str, Any] = {
            "task_key": task,
            "chip_key": args.chip,
            "status": "passed" if passed else "failed",
            "returncode": returncode,
            "timed_out": timed_out,
            "official_pass": passed,
            "metrics": metrics,
            "command": command,
            "artifact": relative(artifact),
            "artifact_sha256": actual_hash,
            "reference": relative(reference),
            "log": relative(log_path),
        }
        summaries.append(summary)
        print(("PASS" if passed else "FAIL"), f"{args.chip}/{task}")
        if output.strip():
            print(output.rstrip())

    final = {
        "schema_version": 1,
        "kind": "package_local_official_auto_bench_run",
        "chip_key": args.chip,
        "tasks": summaries,
        "official_pass_count": sum(item["official_pass"] for item in summaries),
        "official_total": len(summaries),
        "evaluator_commit": EVALUATOR_COMMIT,
        "evaluator_sha256": sha256(bench),
        "evaluator_url": EVALUATOR_URL,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "platform_submission": False,
        "formal_coverage_update": False,
        "results_directory": relative(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SUMMARY {relative(run_dir / 'summary.json')}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
