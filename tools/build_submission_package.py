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
        "kind": "official_diagnostic",
        "path": "results/ascend_a2_910b/61a98c6/summary.json",
        "package_path": "evidence/ascend_a2_910b_summary.json",
        "note": "Latest fixed-evaluator Ascend910B2C evidence: 10/10 PASS accuracy with latency and speedup; retained as target-chip evidence and not promoted to the formal scoring ledger.",
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
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_dir/official_eval.py" --chip "$chip_key" "${task_args[@]}" "$@"
""",
    )
    write_text(
        run_dir / "run_official_eval.ps1",
        """param(
  [Parameter(Mandatory = $true)][string]$ChipKey,
  [string]$TaskKey,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$arguments = @('--chip', $ChipKey)
if ($TaskKey) {
  $arguments += @('--task', $TaskKey)
}
if ($ExtraArgs) {
  $arguments += $ExtraArgs
}
& py -3 (Join-Path $scriptDir 'official_eval.py') @arguments
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
EVALUATOR_SHA256 = '357751a12552d1712ad5f66caa4e0fbd79d940b58a99342f83144fdfc9abb5db'
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
    runtime_dir = ROOT / 'run'
    evaluator = runtime_dir / 'auto_bench.py'
    if not evaluator.is_file() or digest(evaluator) != EVALUATOR_SHA256:
        raise SystemExit('package evaluator is missing or not the pinned auto_bench.py')
    required_runtime = (
        'official_eval.py', 'prepare_case.py', 'chip_runtime.json',
        'evaluator_manifest.json', 'run_official_eval.sh', 'run_official_eval.ps1',
    )
    for name in required_runtime:
        path = runtime_dir / name
        if not path.is_file():
            raise SystemExit(f'missing package runtime file: {path}')
    for name in ('official_eval.py', 'prepare_case.py', 'run_official_eval.sh', 'run_official_eval.ps1'):
        source = (runtime_dir / name).read_text(encoding='utf-8')
        if any(token in source for token in ('tools/', 'upload_artifacts', 'ROOT.parent')):
            raise SystemExit(f'package runtime depends on repository root: {runtime_dir / name}')
    runtime = json.loads((runtime_dir / 'chip_runtime.json').read_text(encoding='utf-8'))
    if set(runtime) != set(CHIPS):
        raise SystemExit('package chip runtime configuration is incomplete')
    evaluator_info = json.loads((runtime_dir / 'evaluator_manifest.json').read_text(encoding='utf-8'))
    if evaluator_info.get('evaluator_sha256') != EVALUATOR_SHA256:
        raise SystemExit('evaluator manifest SHA-256 does not match the pinned evaluator')
    for task in TASKS:
        reference = runtime_dir / 'reference' / f'{task}.py'
        if not reference.is_file():
            raise SystemExit(f'missing package reference input: {task}')
        for chip in CHIPS:
            code = ROOT / task / 'code' / f'{chip}.py'
            environment = ROOT / task / 'environment' / f'{chip}.json'
            if not code.is_file() or not environment.is_file():
                raise SystemExit(f'missing package file: {task}/{chip}')
            if 'biren_br106m' in code.read_text(encoding='utf-8'):
                raise SystemExit(f'legacy chip string in active package: {task}/{chip}')
    obsolete = [ROOT / 'evidence', ROOT / 'package_manifest.json']
    for task in TASKS:
        obsolete.extend([
            ROOT / task / 'manifests',
            ROOT / task / 'reference.py',
            ROOT / task / 'results.json',
        ])
    present = [path for path in obsolete if path.exists()]
    if present:
        raise SystemExit('non-official generated files remain: ' + ', '.join(str(p) for p in present))
    print('PASS package structure: 10 tasks x 10 chips')
    print('PASS package runtime: local evaluator and preparation are self-contained')


if __name__ == '__main__':
    main()
""",
    )
    shutil.copy2(
        ROOT / "tools" / "standalone_package_eval.py",
        run_dir / "official_eval.py",
    )
    shutil.copy2(
        ROOT / "tools" / "standalone_prepare_case.py",
        run_dir / "prepare_case.py",
    )
    reference_dir = run_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for task_key in TASK_LABELS:
        shutil.copy2(
            ROOT / "reference" / f"{task_key}.py",
            reference_dir / f"{task_key}.py",
        )
    runtime_config: dict[str, Any] = {}
    for chip_key in CHIP_LABELS:
        chip_profile = read_json(ROOT / "profiles" / "chips" / f"{chip_key}.json")
        runtime = chip_profile.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"missing runtime profile: {chip_key}")
        runtime_config[chip_key] = {
            key: runtime[key]
            for key in ("torch_device", "source_device_aliases", "bootstrap_imports")
        }
    write_text(
        run_dir / "chip_runtime.json",
        json.dumps(runtime_config, ensure_ascii=False, indent=2),
    )
    evaluator = ROOT / "third_party" / "DLBlas" / "benchmarks" / "ks" / "auto_bench.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"missing pinned evaluator source: {evaluator}")
    evaluator_hash = sha256(evaluator)
    expected_hash = "357751a12552d1712ad5f66caa4e0fbd79d940b58a99342f83144fdfc9abb5db"
    if evaluator_hash != expected_hash:
        raise ValueError(f"pinned evaluator hash mismatch: {evaluator_hash}")
    shutil.copy2(evaluator, run_dir / "auto_bench.py")
    write_text(
        run_dir / "evaluator_manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "evaluator": "DLBlas benchmarks/ks/auto_bench.py",
                "evaluator_commit": "9b5b3627a0f2e5e543ad9d05bf051308bafbd12c",
                "evaluator_sha256": evaluator_hash,
                "source_url": "https://github.com/DeepLink-org/DLBlas/blob/9b5b3627a0f2e5e543ad9d05bf051308bafbd12c/benchmarks/ks/auto_bench.py",
                "runtime_entrypoint": "run/official_eval.py",
                "platform_submission": False,
                "formal_coverage_update": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def build() -> None:
    coverage = read_json(ROOT / "results" / "coverage.json")
    chips = list(CHIP_LABELS)
    tasks = list(TASK_LABELS)
    if list(coverage.get("matrix", {})) != chips:
        raise ValueError("coverage chip order does not match active package catalog")
    if coverage.get("tasks") != tasks:
        raise ValueError("coverage task order does not match active package catalog")

    if OUTPUT.exists():
        allowed = set(tasks) | {"run", "README.md", "evidence", "package_manifest.json"}
        unexpected = [item.name for item in OUTPUT.iterdir() if item.name not in allowed]
        if unexpected:
            raise ValueError(
                f"refusing to overwrite unexpected package entries: {unexpected}"
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Remove only files/directories generated by the previous internal-audit
    # package layout. The official package keeps the task materials and the
    # shared local runtime only.
    obsolete = [OUTPUT / "evidence", OUTPUT / "package_manifest.json"]
    for task_key in tasks:
        obsolete.extend(
            [
                OUTPUT / task_key / "manifests",
                OUTPUT / task_key / "reference.py",
                OUTPUT / task_key / "results.json",
            ]
        )
    for path in obsolete:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for cache in OUTPUT.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)

    official_passed = 0
    for task_key, task_label in TASK_LABELS.items():
        task_dir = OUTPUT / task_key
        code_dir = task_dir / "code"
        environment_dir = task_dir / "environment"
        code_dir.mkdir(parents=True, exist_ok=True)
        environment_dir.mkdir(exist_ok=True)

        rows: list[dict[str, Any]] = []
        for chip_key in chips:
            source_code = ROOT / "upload_artifacts" / chip_key / f"{task_key}.py"
            source_manifest = source_code.with_suffix(".manifest.json")
            source_environment = ROOT / "environments" / chip_key / "manifest.json"
            if not source_code.is_file() or not source_manifest.is_file() or not source_environment.is_file():
                raise FileNotFoundError(f"missing active artifact set: {chip_key}/{task_key}")
            shutil.copy2(source_code, code_dir / f"{chip_key}.py")
            source_env_payload = read_json(source_environment)
            # The package describes the runtime without disclosing the
            # connection target used to collect the metadata. In the source
            # manifests, that target is embedded in "source"; do not copy it
            # (or any future connection fields) into the submission package.
            package_env = {
                key: source_env_payload[key]
                for key in (
                    "schema_version",
                    "chip_key",
                    "python",
                    "packages",
                    "driver",
                    "compiler",
                    "captured_at_utc",
                )
                if key in source_env_payload
            }
            write_text(
                environment_dir / f"{chip_key}.json",
                json.dumps(package_env, ensure_ascii=False, indent=2),
            )

            cell = coverage["matrix"][chip_key][task_key]
            artifact_manifest = read_json(source_manifest)
            row: dict[str, Any] = {
                "chip_key": chip_key,
                "chip": CHIP_LABELS[chip_key],
                "official_ledger_status": cell["status"],
                "current_artifact_sha256": artifact_manifest["artifact_sha256"],
            }
            if cell["status"] == "passed":
                official_passed += 1
                row["ledger_artifact_sha256"] = cell.get("artifact_sha256")
                row["current_artifact_matches_ledger"] = (
                    cell.get("artifact_sha256") == artifact_manifest["artifact_sha256"]
                )
                for key in ("reference_ms", "optimized_ms", "speedup", "run_id"):
                    if key in cell:
                        row[f"ledger_{key}"] = cell[key]
            evidence = evidence_for(chip_key, task_key)
            if evidence is not None:
                row["additional_evidence"] = evidence
            rows.append(row)

        lines = [
            f"# 赛题 {task_key[:2]} · {task_label}",
            "",
            "本目录按赛道说明 4.2 整理该赛题的算子优化代码、README、环境配置和性能测试结果。",
            "",
            "## 文件说明",
            "",
            "- `code/<chip_key>.py`：对应芯片的单文件算子优化代码，入口为 `class Model`。",
            "- `environment/<chip_key>.json`：对应芯片的环境配置文件。",
            "- `performance.md`：该赛题的性能测试结果。",
            "- `../run/`：所有赛题共用的运行脚本、固定 evaluator 和运行所需参考输入。",
            "",
            "## 运行",
            "",
            "```bash",
            f"bash ../run/run_official_eval.sh nvidia_h200 {task_key}",
            "```",
            "",
            "运行脚本只在本地目标环境执行固定 evaluator，不执行平台提交。",
        ]
        write_text(task_dir / "README.md", "\n".join(lines))

        performance_lines = [
            f"# {task_key} 性能测试结果",
            "",
            "表格保留官方评测和目标设备测试的原始性能数据；未记录的数据以 `—` 表示。",
            "",
            "| 芯片 | 官方测试状态 | 官方 ref ms | 官方 opt ms | 官方 speedup | 目标测试状态 | 目标 ref ms | 目标 opt ms | 目标 speedup |",
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
        write_text(task_dir / "performance.md", "\n".join(performance_lines))

    build_run_scripts(OUTPUT / "run")
    root_readme = [
        "# 赛道一作品提交包",
        "",
        "本包按《赛道说明》4.1、4.2 整理，包含 10 道赛题的算子优化代码、README、环境配置、运行脚本和性能测试结果。",
        "",
        "## 目录",
        "",
        "- `01_grouped_topk/` … `10_head_compute_mix_bwd/`：每道赛题的官方作品材料。",
        "- 每道赛题包含 `code/`、`README.md`、`environment/` 和 `performance.md`。",
        "- `run/`：共享的固定 evaluator、参考输入、case preparation、运行脚本和包校验脚本。",
        "",
        "## 本地运行",
        "",
        "```bash",
        "python3 run/validate_package.py",
        "bash run/run_official_eval.sh <chip_key> [task_key]",
        "```",
        "",
        "`run/auto_bench.py` 为固定版本 DLBlas evaluator；运行入口只执行本地评测，不提交平台、不修改仓库 coverage。目标机器仍需安装对应的 PyTorch、Triton 和芯片运行时。",
    ]
    write_text(OUTPUT / "README.md", "\n".join(root_readme))
    print(f"PASS package: {len(tasks)} tasks x {len(chips)} chips")
    print(f"OFFICIAL_LEDGER {official_passed}/{len(tasks) * len(chips)}")
    print(f"OUTPUT {OUTPUT}")


if __name__ == "__main__":
    build()
