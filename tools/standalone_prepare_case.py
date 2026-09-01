"""Prepare a self-contained package case for the pinned evaluator."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


class DeviceLiteralRewriter(ast.NodeTransformer):
    """Rewrite only device literals used by the competition model interface."""

    def __init__(self, aliases: set[str], target: str):
        self.aliases = aliases
        self.target = target
        self.replacements: list[dict[str, Any]] = []

    def _rewrite(self, node: ast.expr, context: str) -> ast.expr:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in self.aliases:
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


class ArtifactEntrypointRewriter(ast.NodeTransformer):
    """Rename the package upload entrypoint Model -> ModelNew for auto_bench."""

    def __init__(self) -> None:
        self.renamed = False

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name == "Model":
            if self.renamed:
                raise ValueError("artifact contains more than one Model class")
            node.name = "ModelNew"
            self.renamed = True
        return self.generic_visit(node)


def _runtime_from_manifest(chip_manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = chip_manifest.get("runtime")
    if not isinstance(runtime, dict):
        chip_profile = chip_manifest.get("chip_profile")
        if not isinstance(chip_profile, dict):
            raise ValueError("package runtime configuration is missing")
        runtime = chip_profile.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("package runtime configuration is missing runtime")
    target = runtime.get("torch_device")
    if not isinstance(target, str) or not target:
        raise ValueError("package manifest has no target torch_device")
    aliases = runtime.get("source_device_aliases", [])
    imports = runtime.get("bootstrap_imports", [])
    if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
        raise ValueError("source_device_aliases must be a string list")
    if not isinstance(imports, list) or not all(isinstance(item, str) for item in imports):
        raise ValueError("bootstrap_imports must be a string list")
    return {
        "target": target,
        "aliases": set(aliases),
        "bootstrap_imports": imports,
    }


def prepare_source(
    source_path: Path,
    output_path: Path,
    runtime: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    insertion = 0
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom):
        if tree.body[insertion].module != "__future__":
            break
        insertion += 1
    tree.body[insertion:insertion] = [
        ast.Import(names=[ast.alias(name=name)])
        for name in runtime["bootstrap_imports"]
    ]
    rewriter = DeviceLiteralRewriter(runtime["aliases"], runtime["target"])
    tree = rewriter.visit(tree)
    entrypoint: dict[str, Any] = {"role": role, "renamed": False}
    if role == "artifact":
        class_rewriter = ArtifactEntrypointRewriter()
        tree = class_rewriter.visit(tree)
        if not class_rewriter.renamed:
            raise ValueError(f"{source_path}: artifact does not define Model")
        entrypoint.update({"renamed": True, "from": "Model", "to": "ModelNew"})
    elif role != "reference":
        raise ValueError(f"unknown role: {role}")
    ast.fix_missing_locations(tree)
    payload = ast.unparse(tree) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return {
        "source": str(source_path),
        "prepared": str(output_path),
        "source_sha256": __import__("hashlib").sha256(source.encode()).hexdigest(),
        "prepared_sha256": __import__("hashlib").sha256(payload.encode()).hexdigest(),
        "target_device": runtime["target"],
        "bootstrap_imports": runtime["bootstrap_imports"],
        "replacements": rewriter.replacements,
        "entrypoint": entrypoint,
    }


def prepare_pair(
    reference: Path,
    artifact: Path,
    output_dir: Path,
    chip_manifest: dict[str, Any],
) -> dict[str, Any]:
    runtime = _runtime_from_manifest(chip_manifest)
    payload = {
        "schema_version": 1,
        "chip_key": chip_manifest.get("chip_key"),
        "reference": prepare_source(
            reference, output_dir / "reference.py", runtime, role="reference"
        ),
        "artifact": prepare_source(
            artifact, output_dir / "artifact.py", runtime, role="artifact"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_preparation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
