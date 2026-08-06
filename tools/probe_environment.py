"""Capture a fail-closed runtime snapshot for one competition chip."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_runtime import (  # noqa: E402
    CHIP_KEYS,
    load_chip_profile,
    load_environment_manifest,
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment(chip_key: str) -> dict[str, Any]:
    profile = load_chip_profile(chip_key)
    environment_manifest = load_environment_manifest(chip_key)
    runtime = profile["runtime"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "chip_key": chip_key,
        "profile": profile,
        "environment_manifest": environment_manifest,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "bootstrap": [],
        "checks": {},
        "ready": False,
        "reproducible": False,
    }
    for name in (
        "torch",
        "triton",
        "torch-npu",
        "torch-mlu",
        "torch-gcu",
        "torch-musa",
        "torch-sdaa",
    ):
        version = _version(name)
        if version is not None:
            result["packages"][name] = version

    target = runtime["torch_device"]
    module_name = runtime["accelerator_module"]
    if not target or not module_name:
        result["checks"]["profile"] = {
            "ok": False,
            "detail": "torch_device/accelerator_module is unconfirmed",
        }
        return result

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        result["checks"]["torch_import"] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        return result

    for bootstrap in runtime["bootstrap_imports"]:
        try:
            importlib.import_module(bootstrap)
            result["bootstrap"].append({"module": bootstrap, "ok": True})
        except Exception as exc:
            result["bootstrap"].append(
                {"module": bootstrap, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            )
            return result

    accelerator = getattr(torch, module_name, None)
    if accelerator is None:
        result["checks"]["accelerator_module"] = {
            "ok": False,
            "detail": f"torch.{module_name} is missing",
        }
        return result
    required = ("is_available", "manual_seed_all", "synchronize")
    missing = [name for name in required if not callable(getattr(accelerator, name, None))]
    if missing:
        result["checks"]["accelerator_api"] = {
            "ok": False,
            "detail": f"missing methods: {missing}",
        }
        return result
    try:
        available = bool(accelerator.is_available())
    except Exception as exc:
        result["checks"]["availability"] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        return result
    if not available:
        result["checks"]["availability"] = {"ok": False, "detail": "runtime is unavailable"}
        return result

    try:
        device = torch.device(target)
        device_count = int(accelerator.device_count()) if hasattr(accelerator, "device_count") else None
        accelerator.synchronize()
    except Exception as exc:
        result["checks"]["device_sync"] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        return result
    device_name = None
    if callable(getattr(accelerator, "get_device_name", None)):
        try:
            device_name = accelerator.get_device_name(0)
        except Exception:
            device_name = None
    result["checks"].update(
        {
            "profile": {"ok": True},
            "torch_import": {"ok": True, "version": getattr(torch, "__version__", "unknown")},
            "accelerator_api": {"ok": True},
            "availability": {"ok": True},
            "device_sync": {"ok": True},
        }
    )
    result["device"] = {
        "type": device.type,
        "index": device.index,
        "count": device_count,
        "name": device_name,
    }
    result["ready"] = True
    expected_packages = environment_manifest["packages"]
    version_mismatches = {
        name: {"expected": expected, "actual": result["packages"].get(name)}
        for name, expected in expected_packages.items()
        if result["packages"].get(name) != expected
    }
    python_matches = environment_manifest["python"] == platform.python_version()
    result["checks"]["environment_lock"] = {
        "ok": environment_manifest["status"] == "verified"
        and python_matches
        and not version_mismatches,
        "status": environment_manifest["status"],
        "python_matches": python_matches,
        "package_mismatches": version_mismatches,
    }
    result["reproducible"] = result["checks"]["environment_lock"]["ok"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", choices=CHIP_KEYS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    payload = probe_environment(args.chip)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.require_ready and not payload["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
