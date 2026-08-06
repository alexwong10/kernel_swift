# KernelSwift 2026 Triton 赛道提交集

本目录包含 `赛道说明.md` 中全部 10 道题的参考实现、公共 Triton `ModelNew` 原型、基础批量
评测脚本，以及 10 款计分芯片的覆盖台账。当前代码尚未形成厂商 profile 或单文件提交产物。

## 目录

```text
reference/            从赛道说明机械提取的 10 个 Model 参考实现
triton_kernels/       10 个 ModelNew 实现和共享 Triton 内核
tools/                提取、静态校验、批量运行工具
results/coverage.json 算子 × 芯片验收台账
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
| 08 | hc_split_sinkhorn | 单 program 完成 split、sigmoid 和 20 次 Sinkhorn |
| 09 | CentreRandomAugmentation | 保持 PyTorch RNG 顺序，Triton 融合中心化、四元数旋转和位移 |
| 10 | head_compute_mix_bwd | 融合 sigmoid backward 与分通道 reduction |

## 本地静态检查

```powershell
python tools/extract_references.py
python tools/static_validate.py
python tools/numeric_emulation_validate.py
python -m compileall -q reference triton_kernels tools
```

`static_validate.py` 会检查 10 对文件、`Model`/`ModelNew` 的 `__init__` 和 `forward` 参数、必需的输入函数，并拒绝 `ModelNew` 中的回退式 `try/except`。
`numeric_emulation_validate.py` 在不依赖 PyTorch/Triton 的情况下检查内核使用的索引、
归约和公式；它只能发现代数错误，不能证明 Triton 可编译或芯片精度通过。逐算子风险见
[`离线审查记录.md`](./离线审查记录.md)。

## 厂商环境评测

先取得官方 DLBlas 仓库，然后在目标加速卡环境运行：

```bash
python tools/run_all.py \
  --bench /path/to/DLBlas/benchmarks/ks/auto_bench.py \
  --chip ascend-a2 \
  --warmup 200 \
  --repeat 500
```

单题快速冒烟：

```bash
python tools/run_all.py \
  --bench /path/to/DLBlas/benchmarks/ks/auto_bench.py \
  --chip ascend-a2 \
  --only 01 \
  --warmup 5 \
  --repeat 20
```

脚本会把原始日志和摘要写入 `results/<chip>/`。正式结果通过后，再把 `results/coverage.json` 对应项从 `not_run` 更新为 `passed`，并填写 reference/optimized 延迟、speedup、运行时版本和设备信息。

## 跨芯片原则

- 核心内核只使用常见 Triton 原语，避免 CUDA 专属 API、CUDA Graph 和硬编码 warp-size。
- 当前官方 `auto_bench.py` 只会把源码中的 `"npu"` 重写为检测到的目标设备；目标为
  GCU 时还会额外执行 `"cuda" -> "gcu"`。它不会在纯 NPU/MLU 环境自动执行
  `"cuda" -> "npu"/"mlu"`。因此包含硬编码 `"cuda"` 的 reference 和优化文件，在昇腾、
  寒武纪及其他非 CUDA 兼容运行时上必须先通过经审计的设备适配步骤，不能假定评测器会处理。
- 当前 tile、`num_warps` 和 stage 仍是源码中的公共硬编码，未经 10 个厂商编译器验证；目标方案
  是用稳定 `chip_key` 选择经实测的 runtime/operator profile 和单文件 artifact。
- 昆仑芯 P800、壁仞 BR106M 需自备资源；KernelSwift 当前下拉框也未显示摩尔线程 PH100。

## 当前验证边界

工作区所在 Windows 主机没有 PyTorch、Triton 或加速卡。当前证据仅包括接口静态检查、已有
NumPy 公式检查和 Python 编译；NumPy 检查不覆盖 SPLADE 完整数值链。没有任何芯片的官方
PASS，实际覆盖为 **0/100**。任何“已覆盖芯片”声明必须以目标芯片上的固定版本官方评测日志
和完整环境证据为准。
