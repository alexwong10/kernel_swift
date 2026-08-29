"""Build the local, task-oriented KernelSwift track submission package.

The package is a repository-local presentation of the active artifacts.  It
does not submit anything to KernelSwift, and it keeps official ledger results
separate from target-runner evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission_package" / "赛道一"

TASK_LABELS = {
    "01_grouped_topk": "GroupedTopk",
    "02_fused_moe": "FusedMoE",
    "03_flex_attention": "FlexAttention",
    "04_splade_sparse_pooler": "SPLADESparsePooler",
    "05_music_flamingo_rotary_embedding": "MusicFlamingoRotaryEmbedding",
    "06_mm_encoder_attention": "MMEncoderAttention",
    "07_mhc_post": "mhc_post",
    "08_hc_split_sinkhorn": "hc_split_sinkhorn",
    "09_centre_random_augmentation": "CentreRandomAugmentation",
    "10_head_compute_mix_bwd": "head_compute_mix_bwd",
}

CHIP_LABELS = {
    "ascend_a2_910b": "Huawei Ascend A2 910B",
    "metax_c500": "MetaX C500",
    "iluvatar_bi150": "Iluvatar BI150",
    "enflame_s60": "Enflame S60",
    "thead_810e": "T-Head Zhenwu 810E",
    "cambricon_mlu590_m9d": "Cambricon MLU590-M9D",
    "hygon_bw1000": "Hygon BW1000",
    "moore_threads_ph100": "Moore Threads PH100",
    "kunlun_p800": "Kunlunxin P800",
    "nvidia_h200": "NVIDIA H200",
}

EVIDENCE = {
    "ascend_a2_910b": {
        "kind": "target_runner_only",
        "path": "results/ascend_a2_910b/f595d21/summary.json",
        "package_path": "evidence/ascend_a2_910b_summary.json",
        "note": "Ascend910B3 target-runner correctness/compile evidence; not official KernelSwift coverage.",
    },
    "metax_c500": {
        "kind": "official_diagnostic",
        "path": "results/metax_c500/official_auto_bench_20260829_current_worktree.json",
        "package_path": "evidence/metax_c500_official_auto_bench_20260829.json",
        "note": "Latest DLBlas official auto_bench diagnostic: 10/10 PASS accuracy with latency and speedup; the unlocked/manual environment is not promoted to the formal scoring ledger.",
    },
    "iluvatar_bi150": {
        "kind": "official_ledger",
        "path": "results/coverage.json",
        "package_path": "evidence/coverage.json",
        "note": "The only current official coverage row: 10/10 BI150 cells with parsed PASS, latency and speedup.",
    },
    "enflame_s60": {
        "kind": "target_runner_only",
        "path": "results/enflame_s60/6f19171/summary.json",
        "package_path": "evidence/enflame_s60_summary.json",
        "note": "Official auto_bench-shaped target evidence; runtime lock is unverified, so it is not in the scoring ledger.",
    },
    "hygon_bw1000": {
        "kind": "target_runner_only",
        "path": "results/hygon_bw1000/20260829_official/official_summary.json",
        "package_path": "evidence/hygon_bw1000_official_summary.json",
        "note": "Fixed-evaluator target-runner evidence; coverage_ledger_updated=false and runtime/environment remain unverified.",
    },
    "moore_threads_ph100": {
        "kind": "target_runner_only",
        "path": "results/moore_threads_ph100/20260814_moore_batch_fb6427f/summary.json",
        "package_path": "evidence/moore_threads_ph100_summary.json",
        "note": "PH100 target-runner evidence on the observed MTT S5000 device; not official KernelSwift coverage.",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def target_result(source: Path, task_key: str) -> dict[str, Any] | None:
    if not source.is_file():
        return None
    payload = read_json(source)
    rows = payload.get("results") or payload.get("tasks") or []
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_task = row.get("task_key") or row.get("task")
        if row_task != task_key:
            continue
        optimized = row.get("optimized_ms", row.get("candidate_ms"))
        reference = row.get("reference_ms")
        speedup = row.get("speedup")
        if speedup is None and isinstance(reference, (int, float)) and isinstance(optimized, (int, float)) and optimized > 0:
            speedup = reference / optimized
        result: dict[str, Any] = {
            "status": row.get("status"),
            "device": row.get("device"),
            "device_name": row.get("device_name"),
            "reference_ms": reference,
            "optimized_ms": optimized,
            "speedup": speedup,
            "artifact_sha256": row.get("artifact_sha256"),
        }
        return {key: value for key, value in result.items() if value is not None}
    return None


def evidence_for(chip_key: str, task_key: str) -> dict[str, Any] | None:
    item = EVIDENCE.get(chip_key)
    if item is None:
        return None
    source = ROOT / item["path"]
    result = target_result(source, task_key)
    if result is None:
        return {"kind": item["kind"], "source": item["path"], "note": item["note"]}
    return {
        "kind": item["kind"],
        "source": item["path"],
        "note": item["note"],
        "result": result,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.3f}"


def build_run_scripts(run_dir: Path) -> None:
    write_text(
        run_dir / "run_official_eval.sh",
        """#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <chip_key> [task_key] [official_eval options...]" >&2
  exit 2
fi

chip_key="$1"
shift
task_args=()
if [[ $# -gt 0 && "$1" != --* ]]; then
  task_args=(--task "$1")
  shift
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec python3 "$repo_root/tools/official_eval.py" --chip "$chip_key" "${task_args[@]}" "$@"
""",
    )
    write_text(
        run_dir / "run_official_eval.ps1",
        """param(
  [Parameter(Mandatory = $true)][string]$ChipKey,
  [string]$TaskKey,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')).Path
$arguments = @('tools/official_eval.py', '--chip', $ChipKey)
if ($TaskKey) {
  $arguments += @('--task', $TaskKey)
}
if ($ExtraArgs) {
  $arguments += $ExtraArgs
}
& py -3 @arguments
exit $LASTEXITCODE
""",
    )
    write_text(
        run_dir / "validate_package.py",
        """from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHIPS = [
    'ascend_a2_910b', 'metax_c500', 'iluvatar_bi150', 'enflame_s60',
    'thead_810e', 'cambricon_mlu590_m9d', 'hygon_bw1000',
    'moore_threads_ph100', 'kunlun_p800', 'nvidia_h200',
]
TASKS = [
    '01_grouped_topk', '02_fused_moe', '03_flex_attention',
    '04_splade_sparse_pooler', '05_music_flamingo_rotary_embedding',
    '06_mm_encoder_attention', '07_mhc_post', '08_hc_split_sinkhorn',
    '09_centre_random_augmentation', '10_head_compute_mix_bwd',
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for task in TASKS:
        task_dir = ROOT / task
        for chip in CHIPS:
            code = task_dir / 'code' / f'{chip}.py'
            manifest = task_dir / 'manifests' / f'{chip}.json'
            environment = task_dir / 'environment' / f'{chip}.json'
            if not code.is_file() or not manifest.is_file() or not environment.is_file():
                raise SystemExit(f'missing package file: {task}/{chip}')
            payload = json.loads(manifest.read_text(encoding='utf-8'))
            if payload.get('task_key') != task or payload.get('chip_key') != chip:
                raise SystemExit(f'manifest identity mismatch: {task}/{chip}')
            if payload.get('artifact_sha256') != digest(code):
                raise SystemExit(f'artifact hash mismatch: {task}/{chip}')
            if 'biren_br106m' in code.read_text(encoding='utf-8'):
                raise SystemExit(f'legacy chip string in active package: {task}/{chip}')
    print('PASS package structure: 10 tasks x 10 chips')


if __name__ == '__main__':
    main()
""",
    )


def build() -> None:
    coverage = read_json(ROOT / "results" / "coverage.json")
    chips = list(CHIP_LABELS)
    tasks = list(TASK_LABELS)
    if list(coverage.get("matrix", {})) != chips:
        raise ValueError("coverage chip order does not match active package catalog")
    if coverage.get("tasks") != tasks:
        raise ValueError("coverage task order does not match active package catalog")

    if OUTPUT.exists() and not (OUTPUT / "package_manifest.json").is_file():
        raise ValueError(
            f"refusing to overwrite a non-generated package directory: {OUTPUT}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence_dir = OUTPUT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    copied_evidence: dict[str, str] = {}
    for item in EVIDENCE.values():
        source = ROOT / item["path"]
        target = OUTPUT / item["package_path"]
        if source.is_file() and str(source) not in copied_evidence:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_evidence[str(source)] = item["package_path"]

    package_entries: list[dict[str, Any]] = []
    official_passed = 0
    for task_key, task_label in TASK_LABELS.items():
        task_dir = OUTPUT / task_key
        code_dir = task_dir / "code"
        manifest_dir = task_dir / "manifests"
        environment_dir = task_dir / "environment"
        code_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(exist_ok=True)
        environment_dir.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "reference" / f"{task_key}.py", task_dir / "reference.py")

        rows: list[dict[str, Any]] = []
        for chip_key in chips:
            source_code = ROOT / "upload_artifacts" / chip_key / f"{task_key}.py"
            source_manifest = source_code.with_suffix(".manifest.json")
            source_environment = ROOT / "environments" / chip_key / "manifest.json"
            if not source_code.is_file() or not source_manifest.is_file() or not source_environment.is_file():
                raise FileNotFoundError(f"missing active artifact set: {chip_key}/{task_key}")
            shutil.copy2(source_code, code_dir / f"{chip_key}.py")
            shutil.copy2(source_manifest, manifest_dir / f"{chip_key}.json")
            shutil.copy2(source_environment, environment_dir / f"{chip_key}.json")

            cell = coverage["matrix"][chip_key][task_key]
            manifest = read_json(source_manifest)
            row: dict[str, Any] = {
                "chip_key": chip_key,
                "chip": CHIP_LABELS[chip_key],
                "official_ledger_status": cell["status"],
                "current_artifact_sha256": manifest["artifact_sha256"],
            }
            if cell["status"] == "passed":
                official_passed += 1
                row["ledger_artifact_sha256"] = cell.get("artifact_sha256")
                row["current_artifact_matches_ledger"] = (
                    cell.get("artifact_sha256") == manifest["artifact_sha256"]
                )
                for key in ("reference_ms", "optimized_ms", "speedup", "run_id", "summary", "log"):
                    if key in cell:
                        row[f"ledger_{key}"] = cell[key]
            evidence = evidence_for(chip_key, task_key)
            if evidence is not None:
                row["additional_evidence"] = evidence
            rows.append(row)

        result_payload = {
            "schema_version": 1,
            "task_key": task_key,
            "task": task_label,
            "coverage_source": "evidence/coverage.json",
            "official_ledger_passed": sum(row["official_ledger_status"] == "passed" for row in rows),
            "chips": rows,
        }
        (task_dir / "results.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        lines = [
            f"# 赛题 {task_key[:2]} · {task_label}",
            "",
            "本目录按赛道说明的单题提交结构整理：优化代码、README、环境配置、运行脚本和性能测试结果。",
            "代码文件均来自当前活动 `upload_artifacts/`，入口为自包含的 `class Model`；H200 未取得目标机环境，相关状态保持未验证。",
            "",
            "## 文件说明",
            "",
            "- `code/<chip_key>.py`：10 款当前有效芯片的单文件提交物。",
            "- `manifests/<chip_key>.json`：对应 artifact 的本地审计 manifest，含 profile 和 SHA-256。",
            "- `environment/<chip_key>.json`：对应芯片环境配置/锁定状态。",
            "- `reference.py`：本题参考接口，便于复核，不是提交入口。",
            "- `results.json`、`performance.md`：官方台账与目标 runner 证据分栏记录。",
            "",
            "## 运行",
            "",
            "```bash",
            f"bash ../run/run_official_eval.sh nvidia_h200 {task_key} --diagnostic",
            "```",
            "",
            "正式计分运行要求目标 runtime、环境锁、固定 evaluator 和干净 commit 均已验证；当前 H200 命令只能作为诊断入口。",
            "",
            "## 结果口径",
            "",
            "`performance.md` 中只有 `official_ledger_status=passed` 才是当前覆盖台账成绩；其他芯片的目标 runner 结果仅供复核，不能替代官方计分证据。",
        ]
        write_text(task_dir / "README.md", "\n".join(lines))

        performance_lines = [
            f"# {task_key} 性能与验证结果",
            "",
            "说明：`官方台账` 是可计分的固定 evaluator 结果；`目标 runner` 是已有真实设备/诊断结果，保留原始数值但不提升 coverage。",
            "",
            "| 芯片 | 官方台账 | 官方 ref ms | 官方 opt ms | 官方 speedup | 目标 runner状态 | 目标 ref ms | 目标 opt ms | 目标 ratio |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for row in rows:
            extra = row.get("additional_evidence", {})
            target = extra.get("result", {}) if isinstance(extra, dict) else {}
            performance_lines.append(
                "| {chip} | {ledger} | {ref} | {opt} | {speed} | {target_status} | {target_ref} | {target_opt} | {target_speed} |".format(
                    chip=row["chip"],
                    ledger=row["official_ledger_status"],
                    ref=format_number(row.get("ledger_reference_ms")),
                    opt=format_number(row.get("ledger_optimized_ms")),
                    speed=format_number(row.get("ledger_speedup")),
                    target_status=target.get("status", "—"),
                    target_ref=format_number(target.get("reference_ms")),
                    target_opt=format_number(target.get("optimized_ms")),
                    target_speed=format_number(target.get("speedup")),
                )
            )
        performance_lines.extend(
            [
                "",
                "证据来源见每个 `results.json` 的 `additional_evidence.source`，以及包根目录 `evidence/`。",
                "BI150 的正式单元引用包内 `evidence/coverage.json`；H200 当前为 `not_run` / 未验证。",
            ]
        )
        mismatches = [
            row
            for row in rows
            if row.get("official_ledger_status") == "passed"
            and row.get("current_artifact_matches_ledger") is False
        ]
        if mismatches:
            performance_lines.append(
                "注意：正式台账 artifact 与当前活动产物 SHA-256 不一致；台账成绩仍按历史 run bundle 保留，当前产物需在目标环境重新验证后才能复用该成绩。"
            )
        write_text(task_dir / "performance.md", "\n".join(performance_lines))
        package_entries.append(
            {
                "task_key": task_key,
                "task": task_label,
                "path": task_key,
                "chips": len(chips),
                "files": {"code": 10, "manifests": 10, "environment": 10},
            }
        )

    build_run_scripts(OUTPUT / "run")
    current = current_commit()
    root_readme = [
        "# 赛道一作品提交包（本地整理版）",
        "",
        "本目录按《赛道说明.md》4.1 的作品结构，按赛题分别整理代码、README、环境配置、运行脚本和测试结果。",
        "本轮只完成仓库内文件整理，不执行 KernelSwift 平台、PR 或邮件提交。正式压缩包名称仍需在提交前补充参赛选手/团队名称、赛道和 UID。",
        "",
        "## 当前状态",
        "",
        f"- 活动芯片：10 款；活动键来自 `profile_runtime.py`，第 10 个芯片为 `nvidia_h200`。",
        f"- 官方覆盖台账：{official_passed}/100；当前只有 BI150 的 10/10 单元进入正式台账。",
        "- MetaX C500：最新官方 auto_bench 诊断结果已归档到 `evidence/metax_c500_official_auto_bench_20260829.json`，10/10 PASS；因环境未锁定，仍不计入正式 coverage 台账。",
        "- H200：已完成 profile、环境占位、10×10 operator profile 和自包含产物接入；没有目标 H200 环境，因此 10 个单元均为未验证/未运行。",
        "- 旧 `biren_br106m`：活动 profile、环境和上传产物已迁入仓库 legacy 区；历史 results 原样保留，不进入本包。",
        "- BI150 台账成绩按其历史 run bundle 和 artifact SHA-256 保留；若与当前重建产物哈希不同，当前产物不自动继承该成绩，需目标环境回归。",
        "",
        "## 目录",
        "",
        "- `01_grouped_topk/` … `10_head_compute_mix_bwd/`：每道赛题一个目录。",
        "- 每题包含 `code/`、`manifests/`、`environment/`、`README.md`、`reference.py`、`results.json` 和 `performance.md`。",
        "- `run/`：本地作品包校验脚本和固定 evaluator 运行入口；H200 只能先使用 `--diagnostic`。",
        "- `evidence/`：coverage 台账和已有目标 runner 摘要的副本，便于离线审计。",
        "",
        "## 提交前检查",
        "",
        "```bash",
        "python3 run/validate_package.py",
        "```",
        "",
        "目标芯片正式测试仍需在对应硬件上使用 `run/run_official_eval.sh` 或 `run/run_official_eval.ps1`，并按固定 evaluator、200 warmup / 500 repeat 和真实环境证据更新 coverage；不使用其他芯片结果替代 H200。",
    ]
    write_text(OUTPUT / "README.md", "\n".join(root_readme))
    manifest = {
        "schema_version": 1,
        "package_kind": "kernelswift_track_submission_local",
        "track": "赛道一",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current,
        "active_chip_count": len(chips),
        "task_count": len(tasks),
        "official_ledger_passed": official_passed,
        "official_ledger_total": len(chips) * len(tasks),
        "legacy_excluded": ["biren_br106m"],
        "tasks": package_entries,
        "evidence": copied_evidence,
    }
    (OUTPUT / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS package: {len(tasks)} tasks x {len(chips)} chips")
    print(f"OFFICIAL_LEDGER {official_passed}/{len(tasks) * len(chips)}")
    print(f"OUTPUT {OUTPUT}")


if __name__ == "__main__":
    build()
