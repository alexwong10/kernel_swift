"""Read-only probe for a MetaX C500 validation host."""

from __future__ import annotations

import importlib.util
import json
import platform
import socket
import subprocess
import sys


def main() -> None:
    result: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    try:
        completed = subprocess.run(
            ["mx-smi", "-L"], text=True, capture_output=True, check=False
        )
        result["mx_smi_returncode"] = completed.returncode
        result["mx_smi"] = completed.stdout + completed.stderr
    except Exception as exc:
        result["mx_smi_error"] = repr(exc)

    modules = {}
    for name in ("torch", "torch_musa", "torch_musa.core", "triton", "triton_musa"):
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = str(spec) if spec is not None else None
        except Exception as exc:
            modules[name] = f"error: {exc!r}"
    result["modules"] = modules

    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_is_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
        if hasattr(torch, "musa"):
            result["musa_is_available"] = bool(torch.musa.is_available())
            result["musa_device_count"] = int(torch.musa.device_count())
            if torch.musa.is_available():
                result["musa_device_name"] = torch.musa.get_device_name(0)
    except Exception as exc:
        result["torch_error"] = repr(exc)

    try:
        import torch_musa  # type: ignore

        result["torch_musa"] = getattr(torch_musa, "__version__", "unknown")
    except Exception as exc:
        result["torch_musa_error"] = repr(exc)

    try:
        import triton

        result["triton"] = triton.__version__
    except Exception as exc:
        result["triton_error"] = repr(exc)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
