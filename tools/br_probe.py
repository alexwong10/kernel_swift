"""Legacy read-only probe for a Biren BR106M validation host.

The active catalog uses ``nvidia_h200``; this helper remains only to interpret
historical Biren evidence and is not part of the current scoring path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys


def command(name: str, *args: str) -> dict[str, object]:
    try:
        completed = subprocess.run([name, *args], text=True, capture_output=True, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> None:
    result: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "path": sys.path,
        "env": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith(("BR", "BIREN", "SUPA", "CUDA", "TRITON", "LD_"))
        },
        "brsmi_gpu": command("brsmi", "gpu", "list"),
        "brsw_inner_version": command("brsw", "-i"),
        "brcc_version": command("brcc", "--version"),
        "pip_triton": command("python3", "-m", "pip", "show", "triton"),
    }
    modules: dict[str, object] = {}
    for name in ("torch", "torch_br", "torch_biren", "triton", "triton_br", "biren", "bpex"):
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = str(spec) if spec is not None else None
        except Exception as exc:
            modules[name] = f"error: {exc!r}"
    result["modules"] = modules
    try:
        try:
            import torch_br  # type: ignore

            result["torch_br_version"] = getattr(torch_br, "__version__", "unknown")
            try:
                from torch_br.utils.utils import has_triton_on_supa  # type: ignore

                result["torch_br_has_triton_on_supa"] = bool(has_triton_on_supa())
            except Exception as exc:
                result["torch_br_triton_probe_error"] = repr(exc)
        except Exception as exc:
            result["torch_br_import_error"] = repr(exc)
        import torch

        result["torch"] = torch.__version__
        result["torch_accelerator_attrs"] = [
            name for name in dir(torch) if "br" in name.lower() or "bire" in name.lower()
        ]
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            x = torch.randn((4, 4), device="cuda")
            result["cuda_tensor"] = {"device": str(x.device), "sum": float(x.float().sum().item())}
            torch.cuda.synchronize()
        for name in ("br", "biren", "supa"):
            backend = getattr(torch, name, None)
            if backend is not None:
                result[f"torch_{name}_available"] = bool(backend.is_available())
                result[f"torch_{name}_device_count"] = int(backend.device_count())
        supa = getattr(torch, "supa", None)
        if supa is not None and supa.is_available():
            try:
                x = torch.randn((4, 4), device="supa")
                result["supa_tensor"] = {
                    "device": str(x.device),
                    "dtype": str(x.dtype),
                    "sum": float(x.float().sum().item()),
                    "contiguous": bool(x.is_contiguous()),
                }
                supa.synchronize()
            except Exception as exc:
                result["supa_tensor_error"] = repr(exc)
    except Exception as exc:
        result["torch_error"] = repr(exc)
    try:
        import triton

        result["triton"] = triton.__version__
    except Exception as exc:
        result["triton_error"] = repr(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
