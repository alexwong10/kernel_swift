"""Build, prepare, run, parse and archive the pinned KernelSwift evaluator."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import (  # noqa: E402
    CHIP_KEYS,
    TASK_KEYS,
    load_chip_profile,
    load_environment_manifest,
    validate_all_profiles,
)
from build_submission import build_submission  # noqa: E402
from coverage_lib import (  # noqa: E402
    EVALUATOR_COMMIT,
    load_coverage,
    update_cell,
    validate_coverage,
    write_coverage,
)
from prepare_case import prepare_pair  # noqa: E402
from probe_environment import probe_environment  # noqa: E402


PASS_PATTERN = re.compile(
    r"PASS accuracy;\s*v0=(?P<reference_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"v1=(?P<optimized_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"speedup=(?P<speedup>[0-9.eE+-]+)x"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--task", choices=TASK_KEYS)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--coverage", type=Path, default=ROOT / "results" / "coverage.json")
    parser.add_argument("--no-update-coverage", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-evaluator-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def git_output(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={cwd.as_posix()}", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )


def source_state() -> tuple[str, bool]:
    commit = git_output("rev-parse", "HEAD").stdout.strip() or "unknown"
    dirty = bool(git_output("status", "--porcelain").stdout.strip())
    return commit, dirty


def verify_evaluator(bench: Path) -> dict[str, Any]:
    top = subprocess.run(
        ["git", "-C", str(bench.parent), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if top.returncode != 0:
        return {"verified": False, "detail": "evaluator is not inside a Git checkout"}
    repo = Path(top.stdout.strip()).resolve()
    try:
        relative = bench.relative_to(repo).as_posix()
    except ValueError:
        return {"verified": False, "detail": "evaluator path is outside its Git root"}
    expected = subprocess.run(
        ["git", "-C", str(repo), "show", f"{EVALUATOR_COMMIT}:{relative}"],
        capture_output=True,
    )
    if expected.returncode != 0:
        return {
            "verified": False,
            "detail": f"pinned commit does not contain {relative}",
            "repo": str(repo),
        }
    verified = expected.stdout == bench.read_bytes()
    return {
        "verified": verified,
        "detail": "exact pinned file" if verified else "working-tree evaluator differs from pinned file",
        "repo": str(repo),
        "relative_path": relative,
        "commit": EVALUATOR_COMMIT,
    }


def parse_metrics(output: str) -> dict[str, float] | None:
    match = PASS_PATTERN.search(output)
    if match is None:
        return None
    return {key: float(value) for key, value in match.groupdict().items()}


def failure_status(output: str, timed_out: bool, returncode: int | None) -> str:
    lowered = output.lower()
    if timed_out:
        return "infrastructure_failed"
    if "failed to load definitions" in lowered or "failed to read" in lowered:
        return "import_failed"
    if "compilationerror" in lowered or "compile failed" in lowered or "compiler" in lowered:
        return "compile_failed"
    if "tensor values differ" in lowered or "accuracy" in lowered and "fail" in lowered:
        return "accuracy_failed"
    if returncode == 0:
        return "benchmark_failed"
    return "runtime_failed"


def failure_detail(output: str, timed_out: bool, timeout: float) -> str:
    if timed_out:
        return f"evaluator exceeded per-task timeout of {timeout} seconds"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return (lines[-1] if lines else "evaluator failed without output")[:500]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def build_command(bench: Path, reference: Path, artifact: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(bench),
        "--v0_file",
        str(reference),
        "--v1_file",
        str(artifact),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--full-traceback",
    ]


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.timeout < 0:
        raise SystemExit("warmup >= 0, repeat > 0 and timeout >= 0 are required")
    validate_all_profiles()
    chip_profile = load_chip_profile(args.chip)
    environment_manifest = load_environment_manifest(args.chip)
    if chip_profile["runtime"]["evaluator_support"] != "native_after_case_preparation":
        raise SystemExit(
            f"{args.chip}: evaluator/runtime path is unconfirmed; update the chip profile first"
        )
    if not args.no_update_coverage and chip_profile["runtime"]["verified"] is not True:
        raise SystemExit(
            f"{args.chip}: runtime profile is unverified; use --no-update-coverage for diagnostics"
        )
    if not args.no_update_coverage and environment_manifest["status"] != "verified":
        raise SystemExit(
            f"{args.chip}: environment lock is unverified; use --no-update-coverage for diagnostics"
        )

    bench = args.bench.resolve()
    if not bench.is_file():
        raise SystemExit(f"evaluator not found: {bench}")
    evaluator = verify_evaluator(bench)
    if not evaluator["verified"] and not args.allow_evaluator_mismatch:
        raise SystemExit(f"evaluator verification failed: {evaluator['detail']}")
    if not evaluator["verified"] and not args.no_update_coverage:
        raise SystemExit("an evaluator mismatch requires --no-update-coverage")

    commit, dirty = source_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("working tree is dirty; commit changes or pass --allow-dirty for diagnostics")
    if dirty and not args.no_update_coverage:
        raise SystemExit("dirty-tree runs require --no-update-coverage")

    tasks = (args.task,) if args.task else TASK_KEYS
    if args.dry_run:
        print(
            json.dumps(
                {
                    "chip_key": args.chip,
                    "tasks": tasks,
                    "source_commit": commit,
                    "dirty": dirty,
                    "evaluator": evaluator,
                    "runtime": chip_profile["runtime"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    environment = probe_environment(args.chip)
    run = run_id()
    run_dir = args.results_root.resolve() / args.chip / commit / run
    run_dir.mkdir(parents=True, exist_ok=False)
    environment_path = run_dir / "environment.json"
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not environment["ready"]:
        raise SystemExit(f"runtime probe failed; evidence saved at {environment_path}")
    if not args.no_update_coverage and not environment["reproducible"]:
        raise SystemExit(
            f"environment does not match its verified lock; evidence saved at {environment_path}"
        )

    coverage_path = args.coverage.resolve()
    coverage = load_coverage(coverage_path)
    validate_coverage(coverage, expected_tasks=TASK_KEYS, expected_chips=CHIP_KEYS)
    all_passed = True
    task_summaries: list[dict[str, Any]] = []
    for task_key in tasks:
        task_dir = run_dir / task_key
        artifact_path = task_dir / "artifact" / f"{task_key}.py"
        artifact_manifest_path = task_dir / "artifact" / "manifest.json"
        artifact_manifest = build_submission(
            task_key,
            args.chip,
            artifact_path,
            manifest_path=artifact_manifest_path,
        )
        reference = ROOT / "reference" / f"{task_key}.py"
        prepared_dir = task_dir / "prepared"
        prepare_pair(reference, artifact_path, prepared_dir, args.chip)
        prepared_reference = prepared_dir / "reference.py"
        prepared_artifact = prepared_dir / "artifact.py"
        command = build_command(bench, prepared_reference, prepared_artifact, args)
        process_env = os.environ.copy()
        process_env["KERNELSWIFT_CHIP_KEY"] = args.chip
        timed_out = False
        try:
            completed = subprocess.run(
                command,
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
            output += f"\n[harness]\ntimed out after {args.timeout} seconds\n"

        log_path = task_dir / "evaluator.log"
        log_path.write_text(output, encoding="utf-8")
        metrics = parse_metrics(output)
        passed = returncode == 0 and metrics is not None
        all_passed &= passed
        status = "passed" if passed else failure_status(output, timed_out, returncode)
        task_summary: dict[str, Any] = {
            "task_key": task_key,
            "chip_key": args.chip,
            "status": status,
            "returncode": returncode,
            "timed_out": timed_out,
            "metrics": metrics,
            "command": command,
            "artifact_manifest": relative_or_absolute(artifact_manifest_path),
            "log": relative_or_absolute(log_path),
        }
        task_summaries.append(task_summary)
        print(("PASS" if passed else "FAIL"), f"{args.chip}/{task_key}")

        measured_at = utc_now()
        if passed:
            cell: dict[str, Any] = {
                "status": "passed",
                "task_key": task_key,
                "chip_key": args.chip,
                "run_id": run,
                "reference_ms": metrics["reference_ms"],
                "optimized_ms": metrics["optimized_ms"],
                "speedup": metrics["speedup"],
                "git_commit": commit,
                "evaluator_commit": EVALUATOR_COMMIT,
                "artifact_sha256": artifact_manifest["artifact_sha256"],
                "environment": relative_or_absolute(environment_path),
                "log": relative_or_absolute(log_path),
                "summary": relative_or_absolute(run_dir / "summary.json"),
                "measured_at_utc": measured_at,
            }
        else:
            cell = {
                "status": status,
                "run_id": run,
                "detail": failure_detail(output, timed_out, args.timeout),
                "log": relative_or_absolute(log_path),
                "measured_at_utc": measured_at,
            }
        if not args.no_update_coverage:
            update_cell(coverage, args.chip, task_key, cell)

    summary = {
        "schema_version": 1,
        "run_id": run,
        "chip_key": args.chip,
        "source_commit": commit,
        "source_dirty": dirty,
        "evaluator": evaluator,
        "environment": relative_or_absolute(environment_path),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "tasks": task_summaries,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_update_coverage:
        write_coverage(
            coverage_path,
            coverage,
            expected_tasks=TASK_KEYS,
            expected_chips=CHIP_KEYS,
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
