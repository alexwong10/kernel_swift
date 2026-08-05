import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _split_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    rows: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    hc_offsets = tl.arange(0, HC)
    matrix_offsets = tl.arange(0, HC * HC)
    row_base = row * ((2 + HC) * HC)
    scale0 = tl.load(scale_ptr)
    scale1 = tl.load(scale_ptr + 1)
    scale2 = tl.load(scale_ptr + 2)

    pre_x = tl.load(mixes_ptr + row_base + hc_offsets).to(tl.float32)
    post_x = tl.load(mixes_ptr + row_base + HC + hc_offsets).to(tl.float32)
    pre_base = tl.load(base_ptr + hc_offsets).to(tl.float32)
    post_base = tl.load(base_ptr + HC + hc_offsets).to(tl.float32)
    pre = tl.sigmoid(pre_x * scale0 + pre_base) + EPS
    post = 2.0 * tl.sigmoid(post_x * scale1 + post_base)

    raw = tl.load(mixes_ptr + row_base + 2 * HC + matrix_offsets).to(tl.float32)
    matrix_base = tl.load(base_ptr + 2 * HC + matrix_offsets).to(tl.float32)
    matrix = tl.reshape(raw * scale2 + matrix_base, (HC, HC))
    row_max = tl.max(matrix, axis=1)
    matrix = tl.exp(matrix - row_max[:, None])
    matrix = matrix / tl.sum(matrix, axis=1)[:, None] + EPS
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + EPS)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + EPS)

    tl.store(pre_ptr + row * HC + hc_offsets, pre)
    tl.store(post_ptr + row * HC + hc_offsets, post)
    tl.store(comb_ptr + row * HC * HC + matrix_offsets, tl.reshape(matrix, (HC * HC,)))


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(self, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        batch, seq_len, _ = mixes.shape
        hc = self.hc_mult
        pre = torch.empty((batch, seq_len, hc), device=mixes.device, dtype=torch.float32)
        post = torch.empty_like(pre)
        comb = torch.empty(
            (batch, seq_len, hc, hc), device=mixes.device, dtype=torch.float32
        )
        _split_sinkhorn_kernel[(batch * seq_len,)](
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            batch * seq_len,
            HC=hc,
            EPS=self.eps,
            ITERS=self.sinkhorn_iters,
            num_warps=1,
        )
        return pre, post, comb


def get_init_inputs():
    return [4, 20, 1e-6]


def get_inputs():
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32)
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32)
    hc_base = torch.randn(mix_hc, dtype=torch.float32) * 0.1
    return [mixes, hc_scale, hc_base]

