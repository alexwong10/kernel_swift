"""Create one self-contained file intended for direct KernelSwift upload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_submission import build_submission  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single-file artifact with no repository-local imports."
    )
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--task", choices=TASK_KEYS, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "upload_artifacts")
    args = parser.parse_args()
    output = args.output_root.resolve() / args.chip / f"{args.task}.py"
    manifest = build_submission(args.task, args.chip, output)
    print(f"UPLOAD_FILE {output}")
    print(f"SHA256 {manifest['artifact_sha256']}")
    print("This file is self-contained; upload it as the candidate ModelNew file.")


if __name__ == "__main__":
    main()
