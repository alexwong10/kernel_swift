import torch
import torch.nn as nn
import triton
import triton.language as tl

from common import gelu_layer_norm, triton_linear


@triton.jit
def _pool_logits_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    vocab_size: tl.constexpr,
    POOL_MAX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    sequence = tl.program_id(0)
    vocab_block = tl.program_id(1)
    prefix_offsets = tl.arange(0, 8)
    lens = tl.load(
        seq_lens_ptr + prefix_offsets,
        mask=prefix_offsets < batch_size,
        other=0,
    ).to(tl.int32)
    start = tl.sum(tl.where(prefix_offsets < sequence, lens, 0), axis=0)
    length = tl.load(seq_lens_ptr + sequence).to(tl.int32)
    token = tl.arange(0, BLOCK_T)
    vocab = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    valid = (token[:, None] < length) & (vocab[None, :] < vocab_size)
    values = tl.load(
        logits_ptr + (start + token[:, None]) * vocab_size + vocab[None, :],
        mask=valid,
        other=-float("inf") if POOL_MAX else 0.0,
    ).to(tl.float32)
    if POOL_MAX:
        pooled = tl.max(values, axis=0)
    else:
        pooled = tl.sum(values, axis=0)
    tl.store(
        out_ptr + sequence * vocab_size + vocab,
        pooled,
        mask=vocab < vocab_size,
    )


class ModelNew(nn.Module):
    def __init__(
        self, hidden_size: int = 768, vocab_size: int = 30522, pooling: str = "max"
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        dense = triton_linear(hidden_states, self.dense.weight, self.dense.bias)
        normalized = gelu_layer_norm(
            dense,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.layer_norm.eps,
        )
        logits = triton_linear(
            normalized,
            self.decoder.weight,
            self.decoder.bias,
            log1p_relu=True,
        )
        batch_size = seq_lens.numel()
        vocab_size = logits.shape[1]
        pooled = torch.empty(
            (batch_size, vocab_size), device=logits.device, dtype=logits.dtype
        )
        block_v = 128
        _pool_logits_kernel[(batch_size, triton.cdiv(vocab_size, block_v))](
            logits,
            seq_lens,
            pooled,
            batch_size,
            vocab_size,
            POOL_MAX=self.pooling == "max",
            BLOCK_T=32,
            BLOCK_V=block_v,
            num_warps=4,
        )
        return [pooled[index] for index in range(batch_size)]


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(83, 768, device="cuda")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]
