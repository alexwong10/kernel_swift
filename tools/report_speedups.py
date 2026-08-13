"""Report only official speedups recorded in the coverage ledger.

This command deliberately ignores ``offline_target_chip_validation`` summaries
so that derived one-shot timings cannot be mistaken for competition scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coverage_lib import load_coverage, validate_coverage  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402


def report(payload: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chip_key in CHIP_KEYS:
        values: list[float] = []
        task_speedups: dict[str, float] = {}
        for task_key in TASK_KEYS:
            cell = payload["matrix"][chip_key][task_key]
            if cell["status"] != "passed":
                continue
            value = float(cell["speedup"])
            values.append(value)
            task_speedups[task_key] = value
        geometric_mean = (
            math.exp(sum(math.log(value) for value in values) / len(values))
            if values
            else None
        )
        rows.append(
            {
                "chip_key": chip_key,
                "official_passed": len(values),
                "official_total": len(TASK_KEYS),
                "geomean_speedup": geometric_mean,
                "task_speedups": task_speedups,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report official KernelSwift speedups")
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "results" / "coverage.json")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = load_coverage(args.path.resolve())
    validate_coverage(payload, expected_tasks=TASK_KEYS, expected_chips=CHIP_KEYS)
    rows = report(payload)
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        mean = row["geomean_speedup"]
        mean_text = "-" if mean is None else f"{float(mean):.4f}x"
        print(
            f"{row['chip_key']}: {row['official_passed']}/{row['official_total']} "
            f"official PASS, geomean speedup={mean_text}"
        )


if __name__ == "__main__":
    main()
