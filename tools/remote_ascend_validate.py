"""Run one self-contained submission against its reference on an Ascend NPU.

This runner is intentionally small and process-per-task friendly: the first
candidate forward includes Triton-Ascend compilation, while the reference is
run with the candidate's initialized state_dict and the same input tensors.
It is not an official KernelSwift evaluator and therefore never updates the
coverage ledger.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def move_to_device(value: Any, device: str):
    """Move the nested get_inputs result to NPU (CPU reference inputs are common)."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def leaves(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    else:
        raise TypeError(f"unsupported output leaf: {type(value)!r}")


def compare_outputs(candidate: Any, reference: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    candidate_leaves = list(leaves(candidate))
    reference_leaves = list(leaves(reference))
    if len(candidate_leaves) != len(reference_leaves):
        raise AssertionError(
            f"output leaf count differs: {len(candidate_leaves)} != {len(reference_leaves)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    for index, (got, expected) in enumerate(zip(candidate_leaves, reference_leaves)):
        if tuple(got.shape) != tuple(expected.shape):
            raise AssertionError(
                f"output[{index}] shape differs: {tuple(got.shape)} != {tuple(expected.shape)}"
            )
        got_f = got.float()
        expected_f = expected.float()
        if not torch.isfinite(got_f).all() or not torch.isfinite(expected_f).all():
            raise AssertionError(f"output[{index}] contains non-finite values")
        diff = (got_f - expected_f).abs()
        max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
        denom = expected_f.abs().clamp_min(1e-12)
        max_rel = max(max_rel, float((diff / denom).max().item()) if diff.numel() else 0.0)
        if not torch.allclose(got_f, expected_f, atol=atol, rtol=rtol):
            raise AssertionError(
                f"output[{index}] mismatch: max_abs={float(diff.max().item())} "
                f"max_rel={float((diff / denom).max().item())}"
            )
    return {"max_abs": max_abs, "max_rel": max_rel, "leaves": len(candidate_leaves)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Import the bridge before torch so the torch device backend is registered.
    import torch_npu  # noqa: F401
    import torch

    torch.npu.set_device(0)
    candidate_module = load_module(args.artifact / f"{args.task}.py", f"candidate_{args.task}")
    reference_module = load_module(args.reference / f"{args.task}.py", f"reference_{args.task}")

    torch.manual_seed(args.seed)
    candidate = candidate_module.Model(*candidate_module.get_init_inputs()).eval().to("npu")
    reference = reference_module.Model(*reference_module.get_init_inputs()).eval().to("npu")
    reference.load_state_dict(candidate.state_dict(), strict=True)

    inputs = move_to_device(candidate_module.get_inputs(), "npu")
    # The random augmentation task consumes framework RNG in forward. Reset the
    # seed immediately before both calls so candidate/reference see identical
    # rand/randn streams while reusing the same input tensors.
    torch.manual_seed(args.seed + 1)
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        candidate_output = candidate(*inputs)
    torch.npu.synchronize()
    candidate_ms = (time.perf_counter() - start) * 1000.0

    torch.manual_seed(args.seed + 1)
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        reference_output = reference(*inputs)
    torch.npu.synchronize()
    reference_ms = (time.perf_counter() - start) * 1000.0

    comparison = compare_outputs(
        candidate_output,
        reference_output,
        atol=args.atol,
        rtol=args.rtol,
    )
    return {
        "task": args.task,
        "device": "npu:0",
        "torch": torch.__version__,
        "candidate_ms": candidate_ms,
        "reference_ms": reference_ms,
        "comparison": comparison,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:  # keep a machine-readable failure record per task
        result = {"task": args.task, "status": "failed", "error": repr(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
