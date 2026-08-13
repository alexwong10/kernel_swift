"""Pinned KernelSwift evaluator contract.

All score-bearing measurements must execute the official DLBlas
``benchmarks/ks/auto_bench.py`` at the pinned commit.  The helpers in this
module intentionally do not implement a replacement timer; they only verify,
invoke, and parse the official evaluator.
"""

from __future__ import annotations

import math
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_URL = (
    "https://github.com/DeepLink-org/DLBlas/blob/"
    "9b5b3627a0f2e5e543ad9d05bf051308bafbd12/benchmarks/ks/auto_bench.py"
)
EVALUATOR_COMMIT = "9b5b3627a0f2e5e543ad9d05bf051308bafbd12"
EVALUATOR_RELATIVE_PATH = "benchmarks/ks/auto_bench.py"
# The pinned source is also recorded by content so an air-gapped or
# GitHub-mirror environment can run the exact official file without a local
# DLBlas checkout.  This is not a replacement evaluator: the bytes must still
# match the pinned upstream file exactly.
EVALUATOR_SHA256 = "357751a12552d1712ad5f66caa4e0fbd79d940b58a99342f83144fdfc9abb5db"
EVALUATION_MODE = "official_auto_bench"

PASS_PATTERN = re.compile(
    r"PASS accuracy;\s*v0=(?P<reference_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"v1=(?P<optimized_ms>[0-9.eE+-]+)\s*ms,\s*"
    r"speedup=(?P<speedup>[0-9.eE+-]+)x"
)


def parse_official_pass(output: str) -> dict[str, float] | None:
    """Parse the official PASS line, rejecting malformed/non-finite metrics."""

    match = PASS_PATTERN.search(output)
    if match is None:
        return None
    metrics = {key: float(value) for key, value in match.groupdict().items()}
    if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
        return None
    # The official line is authoritative, but a grossly inconsistent speedup
    # indicates a truncated or mixed log and must not enter the ledger.
    expected = metrics["reference_ms"] / metrics["optimized_ms"]
    if not math.isclose(metrics["speedup"], expected, rel_tol=5e-3, abs_tol=5e-3):
        return None
    return metrics


def verify_official_evaluator(bench: Path) -> dict[str, Any]:
    """Verify that *bench* is byte-identical to the pinned official script.

    Prefer a real DLBlas checkout when available.  Detached files are accepted
    only when their SHA-256 matches the pinned upstream bytes; this supports
    vendor containers where GitHub's Git transport is unavailable while
    preserving an auditable evaluator identity.
    """

    bench = bench.resolve()
    if bench.name != "auto_bench.py":
        return {
            "verified": False,
            "detail": "evaluator filename must be auto_bench.py",
            "path": str(bench),
            "commit": EVALUATOR_COMMIT,
            "url": EVALUATOR_URL,
        }
    top = subprocess.run(
        ["git", "-C", str(bench.parent), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if top.returncode != 0:
        actual_sha256 = hashlib.sha256(bench.read_bytes()).hexdigest()
        verified = actual_sha256 == EVALUATOR_SHA256
        return {
            "verified": verified,
            "detail": (
                "exact pinned file by SHA-256 (detached)"
                if verified
                else "evaluator is detached and SHA-256 differs from pinned file"
            ),
            "path": str(bench),
            "commit": EVALUATOR_COMMIT,
            "sha256": actual_sha256,
            "expected_sha256": EVALUATOR_SHA256,
            "url": EVALUATOR_URL,
        }
    repo = Path(top.stdout.strip()).resolve()
    try:
        relative = bench.relative_to(repo).as_posix()
    except ValueError:
        return {
            "verified": False,
            "detail": "evaluator path is outside its Git root",
            "path": str(bench),
            "commit": EVALUATOR_COMMIT,
            "url": EVALUATOR_URL,
        }
    if relative != EVALUATOR_RELATIVE_PATH:
        return {
            "verified": False,
            "detail": f"expected repository path {EVALUATOR_RELATIVE_PATH}, got {relative}",
            "repo": str(repo),
            "relative_path": relative,
            "path": str(bench),
            "commit": EVALUATOR_COMMIT,
            "url": EVALUATOR_URL,
        }
    expected = subprocess.run(
        ["git", "-C", str(repo), "show", f"{EVALUATOR_COMMIT}:{relative}"],
        capture_output=True,
    )
    if expected.returncode != 0:
        return {
            "verified": False,
            "detail": f"pinned commit does not contain {relative}",
            "repo": str(repo),
            "relative_path": relative,
            "path": str(bench),
            "commit": EVALUATOR_COMMIT,
            "url": EVALUATOR_URL,
        }
    payload = bench.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    verified = expected.stdout == payload and actual_sha256 == EVALUATOR_SHA256
    return {
        "verified": verified,
        "detail": "exact pinned file" if verified else "working-tree evaluator differs from pinned file",
        "repo": str(repo),
        "relative_path": relative,
        "path": str(bench),
        "commit": EVALUATOR_COMMIT,
        "sha256": actual_sha256,
        "expected_sha256": EVALUATOR_SHA256,
        "url": EVALUATOR_URL,
    }


def official_command(
    bench: Path,
    reference: Path,
    artifact: Path,
    *,
    seed: int,
    atol: float,
    rtol: float,
    warmup: int,
    repeat: int,
    python: str,
) -> list[str]:
    """Build the command line accepted by the pinned official evaluator."""

    return [
        python,
        str(bench),
        "--v0_file",
        str(reference),
        "--v1_file",
        str(artifact),
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
