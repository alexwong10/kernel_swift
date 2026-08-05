# KernelSwift 2026 Triton 赛道提交集

本目录包含 `赛道说明.md` 中全部 10 道题的参考实现、便携 Triton `ModelNew` 实现、批量评测脚本，以及 10 款计分芯片的覆盖台账。

## 目录

```text
reference/            从赛道说明机械提取的 10 个 Model 参考实现
triton_kernels/       10 个 ModelNew 实现和共享 Triton 内核
tools/                提取、静态校验、批量运行工具
results/coverage.json 算子 × 芯片验收台账
KernelSwift平台使用记录.md
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
python -m compileall -q reference triton_kernels tools
```

`static_validate.py` 会检查 10 对文件、`Model`/`ModelNew` 的 `__init__` 和 `forward` 参数、必需的输入函数，并拒绝 `ModelNew` 中的回退式 `try/except`。

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
- `get_inputs()` 中的 `"cuda"` 与参考题保持一致；官方评测器会按当前后端重写到 `npu`、`mlu`、`gcu` 等设备。
- 默认块大小优先保证 10 个厂商编译器可接受；每张卡的最优 `num_warps`、tile 和 stage 数应在该卡实测后单独调节。
- 昆仑芯 P800、壁仞 BR106M 需自备资源；KernelSwift 当前下拉框也未显示摩尔线程 PH100。

## 当前验证边界

工作区所在 Windows 主机没有 PyTorch、Triton 或加速卡，因此这里只能完成源码、接口和语法静态检查。任何“已覆盖芯片”声明必须以目标芯片上的官方 `auto_bench.py` 日志为证，不能用静态检查替代。

