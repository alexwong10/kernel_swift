"""Strict validation and updates for the 10-by-10 coverage ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_COMMIT = "9b5b3627a0f2e5e543ad9d05bf051308bafbd12c"
EVALUATION_MODE = "official_auto_bench"
CELL_STATUSES = {
    "not_run",
    "resource_blocked",
    "queued_observed",
    "running_observed",
    "infrastructure_failed",
    "import_failed",
    "compile_failed",
    "runtime_failed",
    "accuracy_failed",
    "benchmark_failed",
    "passed",
}


def load_coverage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("coverage root must be an object")
    return payload


def _positive(cell: dict[str, Any], key: str, context: str) -> float:
    value = cell.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AssertionError(f"{context}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise AssertionError(f"{context}: {key} must be finite and positive")
    return result


def _nonempty_string(cell: dict[str, Any], key: str, context: str) -> str:
    value = cell.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{context}: {key} must be a non-empty string")
    return value


def _existing_file(cell: dict[str, Any], key: str, context: str) -> None:
    value = _nonempty_string(cell, key, context)
    path = Path(value)
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        raise AssertionError(f"{context}: {key} does not exist: {value}")


def _validate_attempt(cell: dict[str, Any], context: str) -> None:
    status = cell.get("status")
    if status not in CELL_STATUSES:
        raise AssertionError(f"{context}: invalid status {status!r}")
    if status == "passed":
        reference_ms = _positive(cell, "reference_ms", context)
        optimized_ms = _positive(cell, "optimized_ms", context)
        speedup = _positive(cell, "speedup", context)
        if not math.isclose(
            speedup, reference_ms / optimized_ms, rel_tol=5e-3, abs_tol=5e-3
        ):
            raise AssertionError(f"{context}: inconsistent speedup")
        for key in (
            "run_id",
            "git_commit",
            "evaluator_commit",
            "evaluation_mode",
            "artifact_sha256",
            "environment",
            "log",
            "summary",
            "measured_at_utc",
        ):
            _nonempty_string(cell, key, context)
        if cell["evaluator_commit"] != EVALUATOR_COMMIT:
            raise AssertionError(f"{context}: passed cell uses the wrong evaluator")
        if cell["evaluation_mode"] != EVALUATION_MODE:
            raise AssertionError(f"{context}: passed cell is not an official auto_bench result")
        if cell["official_pass"] is not True:
            raise AssertionError(f"{context}: passed cell must contain official_pass=true")
        for key in ("environment", "log", "summary"):
            _existing_file(cell, key, context)
    elif status not in {"not_run", "resource_blocked", "queued_observed", "running_observed"}:
        for key in ("detail", "run_id", "log"):
            _nonempty_string(cell, key, context)
        _existing_file(cell, "log", context)


def validate_coverage(
    payload: dict[str, Any],
    *,
    expected_tasks: tuple[str, ...],
    expected_chips: tuple[str, ...],
) -> None:
    if payload.get("schema_version") != 2:
        raise AssertionError("coverage schema_version must be 2")
    if payload.get("evaluator_commit") != EVALUATOR_COMMIT:
        raise AssertionError("coverage evaluator_commit does not match the pinned evaluator")
    tasks = payload.get("tasks")
    chips = payload.get("chips")
    matrix = payload.get("matrix")
    if tasks != list(expected_tasks):
        raise AssertionError("coverage task order/keys do not match the stable catalog")
    if not isinstance(chips, dict) or tuple(chips) != expected_chips:
        raise AssertionError("coverage chip order/keys do not match the stable catalog")
    if not isinstance(matrix, dict) or tuple(matrix) != expected_chips:
        raise AssertionError("matrix chip order/keys do not match the stable catalog")

    for chip_key, row in matrix.items():
        if not isinstance(row, dict) or tuple(row) != expected_tasks:
            raise AssertionError(f"{chip_key}: row must contain the 10 stable tasks in order")
        for task_key, cell in row.items():
            context = f"{chip_key}/{task_key}"
            if not isinstance(cell, dict):
                raise AssertionError(f"{context}: cell must be an object")
            _validate_attempt(cell, context)
            if "last_attempt" in cell:
                last_attempt = cell["last_attempt"]
                if not isinstance(last_attempt, dict):
                    raise AssertionError(f"{context}: last_attempt must be an object")
                if last_attempt.get("status") == "passed":
                    raise AssertionError(f"{context}: last_attempt cannot duplicate a PASS")
                _validate_attempt(last_attempt, f"{context}/last_attempt")


def write_coverage(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_tasks: tuple[str, ...],
    expected_chips: tuple[str, ...],
) -> None:
    validate_coverage(payload, expected_tasks=expected_tasks, expected_chips=expected_chips)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def update_cell(
    payload: dict[str, Any],
    chip_key: str,
    task_key: str,
    result: dict[str, Any],
) -> None:
    current = payload["matrix"][chip_key][task_key]
    if result["status"] == "passed":
        payload["matrix"][chip_key][task_key] = result
        return
    if current.get("status") == "passed":
        current["last_attempt"] = result
    else:
        payload["matrix"][chip_key][task_key] = result
