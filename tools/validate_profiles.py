"""Validate the complete 10-chip by 10-task profile catalog."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import (  # noqa: E402
    CHIP_KEYS,
    TASK_KEYS,
    get_operator_profile,
    load_chip_profile,
    load_environment_manifest,
    validate_all_profiles,
)


def main() -> None:
    validate_all_profiles()
    runtime_verified = sum(
        load_chip_profile(chip)["runtime"]["verified"] is True for chip in CHIP_KEYS
    )
    environment_verified = sum(
        load_environment_manifest(chip)["status"] == "verified" for chip in CHIP_KEYS
    )
    operator_verified = sum(
        get_operator_profile(task, chip)["verified"] is True
        for task in TASK_KEYS
        for chip in CHIP_KEYS
    )
    print(f"PASS profiles: {len(CHIP_KEYS)} chips x {len(TASK_KEYS)} tasks")
    print(f"verified runtime profiles: {runtime_verified}/{len(CHIP_KEYS)}")
    print(f"verified environment locks: {environment_verified}/{len(CHIP_KEYS)}")
    print(f"verified operator profiles: {operator_verified}/{len(CHIP_KEYS) * len(TASK_KEYS)}")


if __name__ == "__main__":
    main()
