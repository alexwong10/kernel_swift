"""Validate the complete coverage ledger and report verified completion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402
from coverage_lib import load_coverage, validate_coverage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "results" / "coverage.json")
    args = parser.parse_args()
    payload = load_coverage(args.path.resolve())
    validate_coverage(payload, expected_tasks=TASK_KEYS, expected_chips=CHIP_KEYS)
    passed = sum(
        cell["status"] == "passed"
        for row in payload["matrix"].values()
        for cell in row.values()
    )
    print(f"PASS coverage schema: {len(CHIP_KEYS)} chips x {len(TASK_KEYS)} tasks")
    print(f"verified passed cells: {passed}/100")


if __name__ == "__main__":
    main()

