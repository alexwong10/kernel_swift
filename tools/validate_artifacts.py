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
from profile_runtime import (  # noqa: E402
    CHIP_KEYS,
    TASK_KEYS,
    get_operator_profile,
    load_chip_profile,
)
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
    artifact_classes = [
        node for node in artifact_tree.body if isinstance(node, ast.ClassDef)
    ]
    model_classes = [node for node in artifact_classes if node.name == "Model"]
    model_new_classes = [node for node in artifact_classes if node.name == "ModelNew"]
    if len(model_classes) != 1 or model_new_classes:
        raise AssertionError(
            f"{artifact}: KernelSwift upload must expose exactly one Model and no ModelNew"
        )
    artifact_cls = model_classes[0]
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


def validate_persisted_artifacts(root: Path) -> None:
    """Validate the files that will actually be selected in the upload UI."""
    count = 0
    for chip_key in CHIP_KEYS:
        chip_profile = load_chip_profile(chip_key)
        for task_key in TASK_KEYS:
            artifact = root / chip_key / f"{task_key}.py"
            manifest_path = artifact.with_suffix(".manifest.json")
            if not artifact.is_file() or not manifest_path.is_file():
                raise AssertionError(f"missing persisted artifact pair: {chip_key}/{task_key}")
            validate_interface(ROOT / "reference" / f"{task_key}.py", artifact)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["task_key"] != task_key or manifest["chip_key"] != chip_key:
                raise AssertionError(f"{chip_key}/{task_key}: persisted manifest identity differs")
            if manifest["artifact_sha256"] != sha256_file(artifact):
                raise AssertionError(f"{chip_key}/{task_key}: persisted artifact hash differs")
            profile = manifest["profile"]
            expected_profile = get_operator_profile(task_key, chip_key)
            if profile != expected_profile:
                raise AssertionError(f"{chip_key}/{task_key}: persisted profile is stale")
            if manifest["chip_profile"] != chip_profile:
                raise AssertionError(f"{chip_key}/{task_key}: persisted chip profile is stale")
            for relative, expected_hash in manifest["sources"].items():
                source = ROOT / Path(relative)
                if not source.is_file() or sha256_file(source) != expected_hash:
                    raise AssertionError(f"{chip_key}/{task_key}: source hash is stale for {relative}")
            count += 1
    expected = len(CHIP_KEYS) * len(TASK_KEYS)
    if count != expected:
        raise AssertionError(f"validated {count} persisted artifacts, expected {expected}")
    print(f"PASS persisted artifacts: {count}/{expected}")


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
    validate_persisted_artifacts(ROOT / "upload_artifacts")


if __name__ == "__main__":
    main()
