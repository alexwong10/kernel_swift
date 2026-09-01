"""Load explicit KernelSwift chip and operator profiles.

Canonical module sources use this loader during local/vendor evaluation.  Final
single-file artifacts replace it with a baked profile, so a submitted operator
does not depend on environment variables or repository files.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "profiles"
ENVIRONMENT_ROOT = ROOT / "environments"

CHIP_KEYS = (
    "ascend_a2_910b",
    "metax_c500",
    "iluvatar_bi150",
    "enflame_s60",
    "thead_810e",
    "cambricon_mlu590_m9d",
    "hygon_bw1000",
    "moore_threads_ph100",
    "kunlun_p800",
    "nvidia_h200",
)

TASK_KEYS = (
    "01_grouped_topk",
    "02_fused_moe",
    "03_flex_attention",
    "04_splade_sparse_pooler",
    "05_music_flamingo_rotary_embedding",
    "06_mm_encoder_attention",
    "07_mhc_post",
    "08_hc_split_sinkhorn",
    "09_centre_random_augmentation",
    "10_head_compute_mix_bwd",
)


class ProfileError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid JSON profile {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProfileError(f"profile must be an object: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_chip_key(chip_key: str | None = None) -> str:
    resolved = chip_key or os.environ.get("KERNELSWIFT_CHIP_KEY")
    if not resolved:
        raise ProfileError(
            "KERNELSWIFT_CHIP_KEY is required for canonical sources; "
            "build a chip-specific artifact for submission"
        )
    if resolved not in CHIP_KEYS:
        raise ProfileError(f"unknown chip_key {resolved!r}; expected one of {CHIP_KEYS}")
    return resolved


def load_chip_profile(chip_key: str) -> dict[str, Any]:
    chip_key = resolve_chip_key(chip_key)
    payload = _read_json(PROFILE_ROOT / "chips" / f"{chip_key}.json")
    if payload.get("schema_version") != 1 or payload.get("chip_key") != chip_key:
        raise ProfileError(f"invalid chip profile identity: {chip_key}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ProfileError(f"{chip_key}: runtime must be an object")
    required_runtime = {
        "family",
        "torch_device",
        "accelerator_module",
        "bootstrap_imports",
        "source_device_aliases",
        "evaluator_support",
        "verified",
    }
    missing = required_runtime - set(runtime)
    if missing:
        raise ProfileError(f"{chip_key}: runtime missing fields {sorted(missing)}")
    return payload


def load_environment_manifest(chip_key: str) -> dict[str, Any]:
    chip_key = resolve_chip_key(chip_key)
    payload = _read_json(ENVIRONMENT_ROOT / chip_key / "manifest.json")
    if payload.get("schema_version") != 1 or payload.get("chip_key") != chip_key:
        raise ProfileError(f"invalid environment manifest identity: {chip_key}")
    if payload.get("status") not in {"unverified", "verified"}:
        raise ProfileError(f"{chip_key}: environment status must be unverified or verified")
    required = {
        "source",
        "image",
        "python",
        "packages",
        "driver",
        "compiler",
        "captured_at_utc",
        "evidence",
    }
    missing = required - set(payload)
    if missing:
        raise ProfileError(f"{chip_key}: environment manifest missing {sorted(missing)}")
    image = payload["image"]
    if not isinstance(image, dict) or set(image) != {"identifier", "digest"}:
        raise ProfileError(f"{chip_key}: image must contain identifier and digest")
    if not isinstance(payload["packages"], dict):
        raise ProfileError(f"{chip_key}: packages must be an object")
    if payload["status"] == "verified":
        for key in (
            "source",
            "python",
            "driver",
            "compiler",
            "captured_at_utc",
            "evidence",
        ):
            if not isinstance(payload[key], str) or not payload[key]:
                raise ProfileError(f"{chip_key}: verified environment requires {key}")
        if not isinstance(image["identifier"], str) or not image["identifier"]:
            raise ProfileError(f"{chip_key}: verified environment requires image.identifier")
        if image["digest"] is None:
            if payload.get("image_digest_status") != "not_exposed_by_provider":
                raise ProfileError(
                    f"{chip_key}: null image.digest requires image_digest_status=not_exposed_by_provider"
                )
        elif not isinstance(image["digest"], str) or not image["digest"]:
            raise ProfileError(f"{chip_key}: image.digest must be a non-empty string or null")
        for package in ("torch", "triton"):
            if not isinstance(payload["packages"].get(package), str):
                raise ProfileError(f"{chip_key}: verified environment requires {package} pin")
    return payload


def get_operator_profile(task_key: str, chip_key: str | None = None) -> dict[str, Any]:
    if task_key not in TASK_KEYS:
        raise ProfileError(f"unknown task_key {task_key!r}; expected one of {TASK_KEYS}")
    resolved_chip = resolve_chip_key(chip_key)
    payload = _read_json(PROFILE_ROOT / "operators" / f"{task_key}.json")
    if payload.get("schema_version") != 1 or payload.get("task_key") != task_key:
        raise ProfileError(f"invalid operator profile identity: {task_key}")
    chips = payload.get("chips")
    if not isinstance(chips, dict) or set(chips) != set(CHIP_KEYS):
        raise ProfileError(f"{task_key}: chips must contain exactly the 10 stable chip keys")
    default = payload.get("default")
    cell = chips[resolved_chip]
    if not isinstance(default, dict) or not isinstance(cell, dict):
        raise ProfileError(f"{task_key}/{resolved_chip}: invalid default or cell")
    overrides = cell.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ProfileError(f"{task_key}/{resolved_chip}: overrides must be an object")
    merged = _deep_merge(default, overrides)
    if not isinstance(merged.get("variant"), str) or not merged["variant"]:
        raise ProfileError(f"{task_key}/{resolved_chip}: variant must be non-empty")
    if not isinstance(merged.get("config"), dict):
        raise ProfileError(f"{task_key}/{resolved_chip}: config must be an object")
    merged.update(
        {
            "schema_version": 1,
            "task_key": task_key,
            "chip_key": resolved_chip,
            "verified": cell.get("verified") is True,
        }
    )
    return merged


def validate_all_profiles() -> None:
    chip_files = {path.stem for path in (PROFILE_ROOT / "chips").glob("*.json")}
    operator_files = {path.stem for path in (PROFILE_ROOT / "operators").glob("*.json")}
    if chip_files != set(CHIP_KEYS):
        raise ProfileError("chip profile filenames do not match the stable chip catalog")
    if operator_files != set(TASK_KEYS):
        raise ProfileError("operator profile filenames do not match the task catalog")
    for chip_key in CHIP_KEYS:
        load_chip_profile(chip_key)
        load_environment_manifest(chip_key)
    for task_key in TASK_KEYS:
        for chip_key in CHIP_KEYS:
            get_operator_profile(task_key, chip_key)
