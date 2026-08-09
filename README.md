# KernelSwift 2026 Triton 赛道提交集

本目录包含 `赛道说明.md` 中全部 10 道题的参考实现、profile 驱动的 canonical Triton
`ModelNew`、单文件构建与评测工具，以及 10 款计分芯片的完整覆盖台账。适配基础设施已经落地，
但所有厂商 profile 和环境锁仍未上芯验证，官方覆盖仍为 0/100。

## 目录

```text
reference/              从赛道说明机械提取的 10 个 Model 参考实现
triton_kernels/         10 个 canonical ModelNew 和共享 Triton 内核
profiles/chips/         10 个稳定芯片键、runtime 与能力 profile
profiles/operators/     10 题默认配置和完整 10×10 cell override
environments/           每芯片可复现环境锁；当前全部 unverified
tools/                  探针、设备准备、构建、评测与校验工具
results/coverage.json   schema v2 算子 × 芯片验收台账
KernelSwift平台使用记录.md
离线审查记录.md
赛道说明.md
```

## 赛题

| 编号 | 算子 | 实现策略 |
|---:|---|---|
| 01 | GroupedTopk | 单 token 融合 softmax/sigmoid、分组选择和 top-k |
| 02 | FusedMoE | 路由、gate/up、down、top-k reduction 四阶段 Triton |
| 03 | FlexAttention | 融合因果 attention，避免显式注意力矩阵 |
| 04 | SPLADESparsePooler | Triton GEMM、精确 GELU+LayerNorm、log1p-ReLU、分段池化 |
| 05 | MusicFlamingoRotaryEmbedding | 融合 batch/time 频率与 sin/cos |
| 06 | MMEncoderAttention | 融合非因果 attention |
| 07 | mhc_post | 融合 residual mixing 和 bf16 输出 |
| 08 | hc_split_sinkhorn | matrix reduce 与 HC=4 全标量 Sinkhorn 两种路径 |
| 09 | CentreRandomAugmentation | 保持 PyTorch RNG 顺序，Triton 融合中心化、四元数旋转和位移 |
| 10 | head_compute_mix_bwd | sigmoid backward、分块 partial 与确定性二次归约 |

## 本地静态检查

```powershell
python tools/extract_references.py
python tools/static_validate.py
python tools/numeric_emulation_validate.py
python tools/validate_profiles.py
python tools/harness_selftest.py
python tools/validate_artifacts.py
python tools/validate_coverage.py
python -m compileall -q reference triton_kernels tools
```

`static_validate.py` 会检查 10 对文件、`Model`/`ModelNew` 的 `__init__` 和 `forward` 参数、必需的输入函数，并拒绝 `ModelNew` 中的回退式 `try/except`。
`numeric_emulation_validate.py` 在不依赖 PyTorch/Triton 的情况下检查内核使用的索引、
归约和公式；它只能发现代数错误，不能证明 Triton 可编译或芯片精度通过。逐算子风险见
[`离线审查记录.md`](./离线审查记录.md)。

`validate_artifacts.py` 在临时目录生成并校验全部 100 份芯片固化单文件：接口一致、无
`common/profile_runtime` 本地导入、`Model` 上传入口、profile 身份和 SHA-256 一致。构建器还会
对已确认 runtime profile 的芯片改写输入构造中的设备别名，并把替换记录写入 manifest；这同样
不等价于目标编译器验证。

单独构建某个提交文件：

```bash
python tools/build_upload.py \
  --chip ascend_a2_910b \
  --task 05_music_flamingo_rotary_embedding \
  --output-root upload_artifacts
```

将命令输出的 `upload_artifacts/<chip_key>/<task_key>.py` 直接上传到 KernelSwift 新建任务页；
不要上传 `triton_kernels/<task>.py`，也不要只上传 `common.py` 或 `profile_runtime.py`。生成文件会
以 `Model` 作为平台入口类，内联 03/04/06 所需的 `common.py`，并固化芯片 profile，不依赖
仓库目录、环境变量或本地模块。

## 厂商环境评测

先取得官方 DLBlas 仓库，然后在目标加速卡环境运行：

```bash
python tools/run_all.py \
  --bench /path/to/DLBlas/benchmarks/ks/auto_bench.py \
  --chip ascend_a2_910b \
  --warmup 200 \
  --repeat 500
```

单题快速冒烟：

```bash
python tools/run_all.py \
  --bench /path/to/DLBlas/benchmarks/ks/auto_bench.py \
  --chip ascend_a2_910b \
  --task 05_music_flamingo_rotary_embedding \
  --warmup 5 \
  --repeat 20 \
  --no-update-coverage
```

runner 只接受稳定芯片键，并校验 `auto_bench.py` 与固定 commit
`9b5b3627a0f2e5e543ad9d05bf051308bafbd12c` 的文件内容完全一致。诊断运行在 profile/环境尚未
verified 时必须使用 `--no-update-coverage`；正式台账运行要求干净 Git commit、verified runtime
profile 和环境锁。每次执行写入
`results/<chip_key>/<source_commit>/<UTC timestamp>-<suffix>/`，解析到官方 PASS、两侧延迟和
speedup 后才原子更新 `coverage.json`。

## 跨芯片原则

- 核心内核只使用常见 Triton 原语，避免 CUDA 专属 API、CUDA Graph 和硬编码 warp-size。
- 当前官方 `auto_bench.py` 只会把源码中的 `"npu"` 重写为检测到的目标设备；目标为
  GCU 时还会额外执行 `"cuda" -> "gcu"`。它不会在纯 NPU/MLU 环境自动执行
  `"cuda" -> "npu"/"mlu"`。`tools/build_submission.py` 对已有 runtime profile 的芯片
  在生成上传文件时执行同样的显式设备改写，并记录 `device_adaptation`；runtime 未确认的
  芯片保留原始字面量并拒绝伪造适配结论。
- variant、tile 和 `num_warps` 已由稳定 `chip_key` 对应的 operator profile 选择并固化进单文件；
  当前配置尚未经 10 个厂商编译器验证，不能把“可选择”写成“已适配”。
- 昆仑芯 P800、壁仞 BR106M 需自备资源；KernelSwift 当前下拉框也未显示摩尔线程 PH100。

## 当前验证边界

工作区所在 Windows 主机没有 PyTorch、Triton 或加速卡。当前证据包括接口/配置/台账校验、
runner 自测、已有 NumPy 公式检查、Python 编译和 100/100 单文件静态构建回归；NumPy 检查不
覆盖 SPLADE 完整数值链，任何一项也不能证明 Triton 后端可编译。没有任何芯片的官方 PASS，
实际覆盖为 **0/100**。任何“已覆盖芯片”声明必须以目标芯片上的固定版本官方评测日志和完整
环境证据为准。
