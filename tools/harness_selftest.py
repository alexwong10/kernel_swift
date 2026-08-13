"""Offline unit checks for evaluator parsing, preparation and ledger updates."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from coverage_lib import EVALUATOR_COMMIT, load_coverage, update_cell, validate_coverage  # noqa: E402
from build_submission import build_submission  # noqa: E402
from evaluator_contract import parse_official_pass  # noqa: E402
from prepare_case import prepare_source  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402
from run_all import failure_status  # noqa: E402


def evaluator_parser() -> None:
    output = "PASS accuracy; v0=1.2500 ms, v1=5.0e-1 ms, speedup=2.5000x"
    metrics = parse_official_pass(output)
    if metrics != {"reference_ms": 1.25, "optimized_ms": 0.5, "speedup": 2.5}:
        raise AssertionError(f"unexpected parsed metrics: {metrics}")
    if parse_official_pass("PASS accuracy without timing") is not None:
        raise AssertionError("partial PASS must not produce metrics")
    if parse_official_pass("PASS accuracy; v0=1 ms, v1=1 ms, speedup=9x") is not None:
        raise AssertionError("inconsistent official speedup must not produce metrics")
    expected = {
        "timeout": failure_status("", True, None),
        "import": failure_status("Failed to load definitions", False, 1),
        "compile": failure_status("CompilationError", False, 1),
        "accuracy": failure_status("FAIL accuracy", False, 1),
        "runtime": failure_status("traceback", False, 1),
    }
    if expected != {
        "timeout": "infrastructure_failed",
        "import": "import_failed",
        "compile": "compile_failed",
        "accuracy": "accuracy_failed",
        "runtime": "runtime_failed",
    }:
        raise AssertionError(f"unexpected failure classification: {expected}")


def case_preparation_scope() -> None:
    source = (
        "from __future__ import annotations\n"
        "import torch\n"
        "label = 'cuda'\n"
        "device = 'cuda'\n"
        "x = torch.empty(1, device='cuda')\n"
        "y = torch.device('cuda')\n"
    )
    with tempfile.TemporaryDirectory(prefix=".case-preparation-", dir=ROOT) as temp:
        temp_root = Path(temp)
        source_path = temp_root / "source.py"
        prepared_path = temp_root / "prepared.py"
        source_path.write_text(source, encoding="utf-8")
        manifest = prepare_source(source_path, prepared_path, "ascend_a2_910b")
        prepared = prepared_path.read_text(encoding="utf-8")
        if "label = 'cuda'" not in prepared:
            raise AssertionError("non-device string was rewritten")
        if "device = 'npu'" not in prepared or "device='npu'" not in prepared:
            raise AssertionError("device assignment/keyword was not rewritten")
        if "torch.device('npu')" not in prepared or "import torch_npu" not in prepared:
            raise AssertionError("torch.device/bootstrap was not prepared")
        contexts = {item["context"] for item in manifest["replacements"]}
        if contexts != {"device_assignment", "device_keyword", "torch.device"}:
            raise AssertionError(f"unexpected rewrite contexts: {contexts}")


def upload_builder_scope() -> None:
    with tempfile.TemporaryDirectory(prefix=".upload-builder-", dir=ROOT) as temp:
        temp_root = Path(temp)
        ascend_path = temp_root / "ascend.py"
        ascend_manifest = build_submission(
            "02_fused_moe", "ascend_a2_910b", ascend_path
        )
        ascend_source = ascend_path.read_text(encoding="utf-8")
        if "import torch_npu" not in ascend_source or not any(
            token in ascend_source for token in ("device='npu'", 'device="npu"')
        ):
            raise AssertionError("known runtime device adaptation missing")
        if ascend_manifest["device_adaptation"]["status"] != "rewritten":
            raise AssertionError("known runtime adaptation was not recorded")

        # MetaX has a target-runner device profile (``cuda``) even though its
        # environment is not verified yet.  Use a genuinely unresolved chip
        # here so this check continues to exercise the safety rule that the
        # builder must not invent a device for an unknown runtime.
        unresolved_path = temp_root / "unresolved.py"
        unresolved_manifest = build_submission(
            "02_fused_moe", "thead_810e", unresolved_path
        )
        unresolved_source = unresolved_path.read_text(encoding="utf-8")
        if not any(
            token in unresolved_source
            for token in ("device='cuda'", 'device="cuda"')
        ):
            raise AssertionError("unconfirmed runtime must not guess a device")
        if unresolved_manifest["device_adaptation"]["status"] != "unconfirmed_runtime":
            raise AssertionError("unconfirmed runtime status was not recorded")


def coverage_updates() -> None:
    payload = load_coverage(ROOT / "results" / "coverage.json")
    working = copy.deepcopy(payload)
    with tempfile.TemporaryDirectory(prefix=".coverage-selftest-", dir=ROOT) as temp:
        temp_root = Path(temp)
        environment = temp_root / "environment.json"
        log = temp_root / "evaluator.log"
        summary = temp_root / "summary.json"
        later_log = temp_root / "later.log"
        for path in (environment, log, summary, later_log):
            path.write_text("selftest\n", encoding="utf-8")
        passed = {
            "status": "passed",
            "task_key": TASK_KEYS[0],
            "chip_key": CHIP_KEYS[0],
            "run_id": "selftest",
            "reference_ms": 2.0,
            "optimized_ms": 1.0,
            "speedup": 2.0,
            "git_commit": "selftest",
            "evaluator_commit": EVALUATOR_COMMIT,
            "evaluation_mode": "official_auto_bench",
            "official_pass": True,
            "artifact_sha256": "selftest",
            "environment": str(environment),
            "log": str(log),
            "summary": str(summary),
            "measured_at_utc": "2026-08-06T00:00:00+00:00",
        }
        update_cell(working, CHIP_KEYS[0], TASK_KEYS[0], passed)
        update_cell(
            working,
            CHIP_KEYS[0],
            TASK_KEYS[0],
            {
                "status": "runtime_failed",
                "run_id": "selftest-later",
                "detail": "expected self-test failure",
                "log": str(later_log),
                "measured_at_utc": "2026-08-06T00:01:00+00:00",
            },
        )
        cell = working["matrix"][CHIP_KEYS[0]][TASK_KEYS[0]]
        if cell["status"] != "passed" or cell["last_attempt"]["status"] != "runtime_failed":
            raise AssertionError("a later failure must not erase the last official PASS")
        validate_coverage(working, expected_tasks=TASK_KEYS, expected_chips=CHIP_KEYS)

        invalid = copy.deepcopy(working)
        invalid["matrix"][CHIP_KEYS[0]][TASK_KEYS[0]]["speedup"] = 99.0
        try:
            validate_coverage(invalid, expected_tasks=TASK_KEYS, expected_chips=CHIP_KEYS)
        except AssertionError:
            pass
        else:
            raise AssertionError("inconsistent speedup must fail coverage validation")


def main() -> None:
    evaluator_parser()
    case_preparation_scope()
    upload_builder_scope()
    coverage_updates()
    print("PASS harness self-test")


if __name__ == "__main__":
    main()
