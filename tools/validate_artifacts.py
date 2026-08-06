"""Build and statically validate all 100 chip-specific standalone artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_submission import build_submission  # noqa: E402
from profile_runtime import CHIP_KEYS, TASK_KEYS  # noqa: E402
from static_validate import (  # noqa: E402
    find_class,
    find_method,
    function_args,
    function_defaults,
    top_level_functions,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_interface(reference: Path, artifact: Path) -> None:
    ref_tree = ast.parse(reference.read_text(encoding="utf-8"), filename=str(reference))
    artifact_source = artifact.read_text(encoding="utf-8")
    artifact_tree = ast.parse(artifact_source, filename=str(artifact))
    compile(artifact_source, str(artifact), "exec")

    for node in ast.walk(artifact_tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "common",
            "profile_runtime",
        }:
            raise AssertionError(f"{artifact}: forbidden local import {node.module}")
        if isinstance(node, ast.Import) and any(
            alias.name in {"common", "profile_runtime"} for alias in node.names
        ):
            raise AssertionError(f"{artifact}: forbidden local import")

    ref_cls = find_class(ref_tree, "Model")
    artifact_cls = find_class(artifact_tree, "ModelNew")
    for method_name in ("__init__", "forward"):
        ref_method = find_method(ref_cls, method_name)
        artifact_method = find_method(artifact_cls, method_name)
        if function_args(ref_method) != function_args(artifact_method):
            raise AssertionError(f"{artifact}:{method_name} arguments differ")
        if function_defaults(ref_method) != function_defaults(artifact_method):
            raise AssertionError(f"{artifact}:{method_name} defaults differ")
    required = {"get_init_inputs", "get_inputs"}
    if not required <= top_level_functions(artifact_tree):
        raise AssertionError(f"{artifact}: missing competition input helpers")


def main() -> None:
    count = 0
    with tempfile.TemporaryDirectory(prefix=".artifact-validation-", dir=ROOT) as temp:
        output_root = Path(temp)
        for chip_key in CHIP_KEYS:
            for task_key in TASK_KEYS:
                artifact = output_root / chip_key / f"{task_key}.py"
                manifest_path = artifact.with_suffix(".manifest.json")
                manifest = build_submission(
                    task_key,
                    chip_key,
                    artifact,
                    manifest_path=manifest_path,
                )
                validate_interface(
                    ROOT / "reference" / f"{task_key}.py",
                    artifact,
                )
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                if saved != manifest:
                    raise AssertionError(f"{chip_key}/{task_key}: manifest round-trip differs")
                if manifest["task_key"] != task_key or manifest["chip_key"] != chip_key:
                    raise AssertionError(f"{chip_key}/{task_key}: manifest identity differs")
                if manifest["artifact_sha256"] != sha256_file(artifact):
                    raise AssertionError(f"{chip_key}/{task_key}: artifact hash differs")
                profile = manifest["profile"]
                if profile["task_key"] != task_key or profile["chip_key"] != chip_key:
                    raise AssertionError(f"{chip_key}/{task_key}: baked profile differs")
                count += 1
    expected = len(CHIP_KEYS) * len(TASK_KEYS)
    if count != expected:
        raise AssertionError(f"validated {count} artifacts, expected {expected}")
    print(f"PASS standalone artifacts: {count}/{expected}")


if __name__ == "__main__":
    main()
