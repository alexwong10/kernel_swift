"""Build a self-contained, chip-profile-baked operator submission file."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import (  # noqa: E402
    CHIP_KEYS,
    TASK_KEYS,
    get_operator_profile,
    load_chip_profile,
)


BUILDER_VERSION = 5


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _split_module(tree: ast.Module) -> tuple[list[ast.stmt], list[ast.stmt]]:
    imports: list[ast.stmt] = []
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_operator_profile":
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_KS_PROFILE"
            for target in node.targets
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.module in {"common", "profile_runtime"}:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            body.append(node)
    return imports, body


class _InlineProfileCallRewriter(ast.NodeTransformer):
    """Replace source profile helper calls with the baked dictionary constant."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "get_operator_profile":
            return ast.copy_location(
                ast.Name(id="_KS_BAKED_PROFILE", ctx=ast.Load()), node
            )
        return node


def _baked_profile_nodes(profile: dict[str, Any]) -> list[ast.stmt]:
    return ast.parse(f"_KS_BAKED_PROFILE = {profile!r}\n").body


def _uses_common(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and node.module == "common" for node in tree.body
    )


class _DeviceLiteralRewriter(ast.NodeTransformer):
    """Make known chip artifacts construct inputs on their target runtime."""

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
        if node.arg == "device":
            node.value = self._rewrite(node.value, "device_keyword")
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & {"device", "torch_device"}:
            node.value = self._rewrite(node.value, "device_assignment")
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id in {
            "device",
            "torch_device",
        } and node.value is not None:
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
        if is_torch_device and node.args:
            node.args[0] = self._rewrite(node.args[0], "torch.device")
        return node


def _prepare_task_devices(
    tree: ast.Module, chip_profile: dict[str, Any]
) -> tuple[ast.Module, dict[str, Any]]:
    runtime = chip_profile.get("runtime", {})
    target = runtime.get("torch_device")
    aliases = runtime.get("source_device_aliases", [])
    if not isinstance(target, str) or not target or not isinstance(aliases, list):
        return tree, {
            "target_device": target,
            "source_aliases": aliases,
            "replacements": [],
            "status": "unconfirmed_runtime",
        }
    alias_set = {value for value in aliases if isinstance(value, str) and value}
    if not alias_set:
        return tree, {
            "target_device": target,
            "source_aliases": [],
            "replacements": [],
            "status": "no_source_aliases",
        }
    rewriter = _DeviceLiteralRewriter(alias_set, target)
    rewritten = rewriter.visit(tree)
    ast.fix_missing_locations(rewritten)
    return rewritten, {
        "target_device": target,
        "source_aliases": sorted(alias_set),
        "replacements": rewriter.replacements,
        "status": "rewritten" if rewriter.replacements else "no_matches",
    }


def _runtime_imports(chip_profile: dict[str, Any]) -> list[ast.stmt]:
    runtime = chip_profile.get("runtime", {})
    imports = runtime.get("bootstrap_imports", [])
    if not isinstance(imports, list):
        return []
    return [
        ast.Import(names=[ast.alias(name=name)])
        for name in imports
        if isinstance(name, str) and name
    ]


def _rename_submission_class(module: ast.Module) -> None:
    """KernelSwift upload entrypoint is Model; canonical sources use ModelNew."""
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one canonical ModelNew class, found {len(matches)}")
    matches[0].name = "Model"


def build_submission(
    task_key: str,
    chip_key: str,
    output_path: Path,
    *,
    manifest_path: Path | None = None,
    profile_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task_key not in TASK_KEYS:
        raise ValueError(f"unknown task_key: {task_key}")
    if chip_key not in CHIP_KEYS:
        raise ValueError(f"unknown chip_key: {chip_key}")

    task_path = ROOT / "triton_kernels" / f"{task_key}.py"
    common_path = ROOT / "triton_kernels" / "common.py"
    profile_path = ROOT / "profiles" / "operators" / f"{task_key}.json"
    chip_path = ROOT / "profiles" / "chips" / f"{chip_key}.json"
    task_tree = ast.parse(task_path.read_text(encoding="utf-8"), filename=str(task_path))
    chip_profile = load_chip_profile(chip_key)
    task_tree, device_adaptation = _prepare_task_devices(task_tree, chip_profile)
    task_tree = _InlineProfileCallRewriter().visit(task_tree)
    ast.fix_missing_locations(task_tree)
    task_imports, task_body = _split_module(task_tree)

    common_imports: list[ast.stmt] = []
    common_body: list[ast.stmt] = []
    include_common = _uses_common(task_tree)
    if include_common:
        common_tree = ast.parse(
            common_path.read_text(encoding="utf-8"), filename=str(common_path)
        )
        common_tree = _InlineProfileCallRewriter().visit(common_tree)
        ast.fix_missing_locations(common_tree)
        common_imports, common_body = _split_module(common_tree)

    profile = get_operator_profile(task_key, chip_key)
    if profile_override is not None:
        if not isinstance(profile_override, dict):
            raise ValueError("profile_override must be an object")
        profile = copy.deepcopy(profile)
        if "variant" in profile_override:
            variant = profile_override["variant"]
            if not isinstance(variant, str) or not variant:
                raise ValueError("profile override variant must be a non-empty string")
            profile["variant"] = variant
        if "config" in profile_override:
            config = profile_override["config"]
            if not isinstance(config, dict):
                raise ValueError("profile override config must be an object")
            profile["config"].update(copy.deepcopy(config))
        profile["verified"] = False
        profile["candidate_override"] = copy.deepcopy(profile_override)
    module = ast.Module(
        body=[
            *_runtime_imports(chip_profile),
            *task_imports,
            *common_imports,
            *_baked_profile_nodes(profile),
            *common_body,
            *task_body,
        ],
        type_ignores=[],
    )
    _rename_submission_class(module)
    ast.fix_missing_locations(module)
    generated = ast.unparse(module) + "\n"
    header = (
        "# Generated by tools/build_submission.py; do not edit.\n"
        f"# task_key={task_key} chip_key={chip_key} builder_version={BUILDER_VERSION}\n"
    )
    payload = (header + generated).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "builder_version": BUILDER_VERSION,
        "task_key": task_key,
        "chip_key": chip_key,
        "source_commit": git_commit(),
        "artifact": str(output_path),
        "artifact_sha256": sha256_bytes(payload),
        "sources": {
            task_path.relative_to(ROOT).as_posix(): sha256_file(task_path),
            profile_path.relative_to(ROOT).as_posix(): sha256_file(profile_path),
            chip_path.relative_to(ROOT).as_posix(): sha256_file(chip_path),
        },
        "profile": profile,
        "chip_profile": chip_profile,
        "device_adaptation": device_adaptation,
    }
    if include_common:
        manifest["sources"][common_path.relative_to(ROOT).as_posix()] = sha256_file(
            common_path
        )
    manifest_path = manifest_path or output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASK_KEYS)
    parser.add_argument("--chip", choices=CHIP_KEYS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--variant",
        help="explicit unverified variant override for a diagnostic candidate build",
    )
    parser.add_argument(
        "--config-json",
        help="JSON object merged into the selected profile config for a diagnostic candidate build",
    )
    args = parser.parse_args()
    if args.all == bool(args.task or args.chip):
        raise SystemExit("choose either --all or both --task and --chip")
    if not args.all and (not args.task or not args.chip):
        raise SystemExit("--task and --chip are required together")
    if args.all and (args.variant or args.config_json):
        raise SystemExit("candidate profile overrides require a single --task/--chip pair")
    profile_override: dict[str, Any] | None = None
    if args.variant or args.config_json:
        profile_override = {}
        if args.variant:
            profile_override["variant"] = args.variant
        if args.config_json:
            try:
                config = json.loads(args.config_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"--config-json must be valid JSON: {exc}") from exc
            if not isinstance(config, dict):
                raise SystemExit("--config-json must decode to a JSON object")
            profile_override["config"] = config

    pairs = (
        [(task, chip) for chip in CHIP_KEYS for task in TASK_KEYS]
        if args.all
        else [(args.task, args.chip)]
    )
    for task_key, chip_key in pairs:
        output = args.output_root / chip_key / f"{task_key}.py"
        build_submission(task_key, chip_key, output, profile_override=profile_override)
        print(f"BUILT {chip_key}/{task_key}")


if __name__ == "__main__":
    main()
