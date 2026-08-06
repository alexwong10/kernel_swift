"""Prepare device-only temporary copies of a reference and artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import CHIP_KEYS, load_chip_profile  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeviceLiteralRewriter(ast.NodeTransformer):
    def __init__(self, aliases: set[str], target: str):
        self.aliases = aliases
        self.target = target
        self.replacements: list[dict[str, Any]] = []

    def _rewrite(self, node: ast.expr, context: str) -> ast.expr:
        if isinstance(node.value, str) and node.value in self.aliases:
            self.replacements.append(
                {
                    "line": getattr(node, "lineno", None),
                    "from": node.value,
                    "to": self.target,
                    "context": context,
                }
            )
            return ast.copy_location(ast.Constant(value=self.target), node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        node = self.generic_visit(node)
        if node.arg == "device" and isinstance(node.value, ast.Constant):
            node.value = self._rewrite(node.value, "device_keyword")
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & {"device", "torch_device"} and isinstance(node.value, ast.Constant):
            node.value = self._rewrite(node.value, "device_assignment")
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.target, ast.Name)
            and node.target.id in {"device", "torch_device"}
            and isinstance(node.value, ast.Constant)
        ):
            node.value = self._rewrite(node.value, "device_assignment")
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        is_torch_device = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "device"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        )
        if is_torch_device and node.args and isinstance(node.args[0], ast.Constant):
            node.args[0] = self._rewrite(node.args[0], "torch.device")
        return node


def prepare_source(source_path: Path, output_path: Path, chip_key: str) -> dict[str, Any]:
    chip = load_chip_profile(chip_key)
    runtime = chip["runtime"]
    target = runtime["torch_device"]
    if not isinstance(target, str) or not target:
        raise ValueError(f"{chip_key}: torch_device is unconfirmed; case preparation fails closed")
    aliases = set(runtime["source_device_aliases"])
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    bootstrap_imports = runtime.get("bootstrap_imports", [])
    if not isinstance(bootstrap_imports, list) or not all(
        isinstance(name, str) and name for name in bootstrap_imports
    ):
        raise ValueError(f"{chip_key}: bootstrap_imports must be a list of module names")
    insertion = 0
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom):
        if tree.body[insertion].module != "__future__":
            break
        insertion += 1
    tree.body[insertion:insertion] = [
        ast.Import(names=[ast.alias(name=name)]) for name in bootstrap_imports
    ]
    rewriter = DeviceLiteralRewriter(aliases, target)
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    payload = (ast.unparse(tree) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {
        "source": str(source_path),
        "prepared": str(output_path),
        "source_sha256": sha256_file(source_path),
        "prepared_sha256": hashlib.sha256(payload).hexdigest(),
        "chip_key": chip_key,
        "target_device": target,
        "bootstrap_imports": bootstrap_imports,
        "replacements": rewriter.replacements,
    }


def prepare_pair(
    reference: Path,
    artifact: Path,
    output_dir: Path,
    chip_key: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "chip_key": chip_key,
        "reference": prepare_source(reference, output_dir / "reference.py", chip_key),
        "artifact": prepare_source(artifact, output_dir / "artifact.py", chip_key),
    }
    (output_dir / "case_preparation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_pair(
        args.reference.resolve(),
        args.artifact.resolve(),
        args.output_dir.resolve(),
        args.chip,
    )
    print(f"PREPARED {args.chip}: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
