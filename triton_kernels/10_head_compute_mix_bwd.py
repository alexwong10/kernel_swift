import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _head_mix_bwd_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_parts_ptr,
    grad_base_ptr,
    rows: tl.constexpr,
    mhc_mult: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mix = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    valid = row < rows
    offset = row * mhc_mult + mix
    x = tl.load(input_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + mix).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(x * scale + base)
    grad_z = grad_out * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input_ptr + offset, grad_z * scale, mask=valid)
    tl.store(grad_base_ptr + mix, tl.sum(grad_z, axis=0))
    tl.store(grad_scale_parts_ptr + mix, tl.sum(grad_z * x, axis=0))


@triton.jit
def _sum_scale_parts_kernel(parts_ptr, out_ptr, mhc_mult: tl.constexpr):
    offsets = tl.arange(0, mhc_mult)
    tl.store(out_ptr, tl.sum(tl.load(parts_ptr + offsets).to(tl.float32), axis=0))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        mhc_mult = input_mix.shape[-1]
        rows = input_mix.numel() // mhc_mult
        grad_input = torch.empty_like(input_mix)
        grad_scale_parts = torch.empty(
            mhc_mult, device=input_mix.device, dtype=torch.float32
        )
        grad_base = torch.empty_like(mhc_base)
        block = triton.next_power_of_2(rows)
        _head_mix_bwd_kernel[(mhc_mult,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            grad_scale_parts,
            grad_base,
            rows,
            mhc_mult,
            BLOCK=block,
            num_warps=8,
        )
        grad_scale = torch.empty_like(mhc_scale)
        _sum_scale_parts_kernel[(1,)](grad_scale_parts, grad_scale, mhc_mult=mhc_mult)
        return grad_input, grad_scale, grad_base


def get_inputs():
    input_mix = torch.randn(2, 1024, 4, dtype=torch.float32)
    mhc_scale = torch.randn(1, dtype=torch.float32)
    mhc_base = torch.randn(4, dtype=torch.float32)
    grad_out = torch.randn(2, 1024, 4, dtype=torch.float32)
    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []
