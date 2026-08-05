"""Static competition-interface checks that do not require PyTorch/Triton."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
OPTIMIZED = ROOT / "triton_kernels"


def function_args(node: ast.FunctionDef) -> tuple[str, ...]:
    args = [*node.args.posonlyargs, *node.args.args]
    names = [arg.arg for arg in args if arg.arg != "self"]
    names.extend(arg.arg for arg in node.args.kwonlyargs)
    return tuple(names)


def function_defaults(node: ast.FunctionDef) -> tuple[str, ...]:
    positional = tuple(ast.dump(value, include_attributes=False) for value in node.args.defaults)
    keyword_only = tuple(
        "<required>" if value is None else ast.dump(value, include_attributes=False)
        for value in node.args.kw_defaults
    )
    return positional + keyword_only


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls.name} missing {name}")


def top_level_functions(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def main() -> None:
    reference_files = sorted(REFERENCE.glob("[0-9][0-9]_*.py"))
    optimized_files = sorted(OPTIMIZED.glob("[0-9][0-9]_*.py"))
    if len(reference_files) != 10 or len(optimized_files) != 10:
        raise AssertionError("expected exactly ten reference and ten optimized files")

    for ref_path, opt_path in zip(reference_files, optimized_files):
        if ref_path.name != opt_path.name:
            raise AssertionError(f"file mismatch: {ref_path.name} vs {opt_path.name}")
        ref_tree = ast.parse(ref_path.read_text(encoding="utf-8"), filename=str(ref_path))
        opt_tree = ast.parse(opt_path.read_text(encoding="utf-8"), filename=str(opt_path))
        ref_cls = find_class(ref_tree, "Model")
        opt_cls = find_class(opt_tree, "ModelNew")
        for method_name in ("__init__", "forward"):
            ref_method = find_method(ref_cls, method_name)
            opt_method = find_method(opt_cls, method_name)
            ref_args = function_args(ref_method)
            opt_args = function_args(opt_method)
            if ref_args != opt_args:
                raise AssertionError(
                    f"{opt_path.name}:{method_name} args {opt_args} != reference {ref_args}"
                )
            ref_defaults = function_defaults(ref_method)
            opt_defaults = function_defaults(opt_method)
            if ref_defaults != opt_defaults:
                raise AssertionError(
                    f"{opt_path.name}:{method_name} defaults differ from reference"
                )
        functions = top_level_functions(opt_tree)
        for required in ("get_init_inputs", "get_inputs"):
            if required not in functions:
                raise AssertionError(f"{opt_path.name} missing {required}")
        if any(isinstance(node, ast.Try) for node in ast.walk(opt_cls)):
            raise AssertionError(f"{opt_path.name} contains forbidden fallback-style try/except")
        print(f"PASS {opt_path.name}")


if __name__ == "__main__":
    main()
