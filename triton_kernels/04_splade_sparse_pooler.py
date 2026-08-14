import torch
import torch.nn as nn
import triton
import triton.language as tl

from common import gelu_layer_norm, triton_decoder_pool, triton_linear

_KS_PROFILE = {
    "task_key": "04_splade_sparse_pooler",
    "chip_key": "portable_default",
    "variant": "staged_portable",
    "config": {
        "linear_block_m": 16,
        "linear_block_n": 64,
        "linear_block_k": 32,
        "linear_num_warps": 4,
        "layer_norm_num_warps_small": 4,
        "layer_norm_num_warps_large": 8,
        "pool_block_t": 32,
        "pool_block_v": 128,
        "pool_num_warps": 4,
    },
}


def get_operator_profile(task_key, chip_key=None):
    if task_key != _KS_PROFILE["task_key"]:
        raise ValueError(f"unsupported task profile: {task_key}")
    return _KS_PROFILE


@triton.jit
def _pool_logits_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    total_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
    POOL_MAX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    sequence = tl.program_id(0)
    vocab_block = tl.program_id(1)
    start = 0
    for index in range(batch_size):
        item_length = tl.load(seq_lens_ptr + index).to(tl.int32)
        start += tl.where(index < sequence, item_length, 0)
    length = tl.load(seq_lens_ptr + sequence).to(tl.int32)
    token = tl.arange(0, BLOCK_T)
    vocab = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    if POOL_MAX:
        pooled = tl.full((BLOCK_V,), -float("inf"), tl.float32)
    else:
        pooled = tl.zeros((BLOCK_V,), tl.float32)
    for token_start in range(0, total_tokens, BLOCK_T):
        current_token = token_start + token
        # The reference inputs satisfy sum(seq_lens) == total_tokens, but keep
        # the total-token guard so malformed lengths cannot read past logits.
        valid = (
            (current_token[:, None] < length)
            & (current_token[:, None] < total_tokens)
            & (vocab[None, :] < vocab_size)
        )
        values = tl.load(
            logits_ptr
            + (start + current_token[:, None]) * vocab_size
            + vocab[None, :],
            mask=valid,
            other=-float("inf") if POOL_MAX else 0.0,
        ).to(tl.float32)
        if POOL_MAX:
            pooled = tl.maximum(pooled, tl.max(values, axis=0))
        else:
            pooled += tl.sum(values, axis=0)
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
        profile = get_operator_profile("04_splade_sparse_pooler")
        if profile["variant"] not in {"staged_portable", "fused_pool"}:
            raise ValueError(f"unsupported SPLADE variant: {profile['variant']}")
        self._ks_variant = profile["variant"]
        config = profile["config"]
        self._ks_linear_config = {
            "block_m": int(config["linear_block_m"]),
            "block_n": int(config["linear_block_n"]),
            "block_k": int(config["linear_block_k"]),
            "num_warps": int(config["linear_num_warps"]),
        }
        self._ks_layer_norm_config = {
            "num_warps_small": int(config["layer_norm_num_warps_small"]),
            "num_warps_large": int(config["layer_norm_num_warps_large"]),
        }
        self._ks_pool_config = {
            "block_t": int(config["pool_block_t"]),
            "block_v": int(config["pool_block_v"]),
            "num_warps": int(config["pool_num_warps"]),
        }
        self._ks_fused_config = {
            "block_t": int(config.get("fused_block_t", 16)),
            "block_v": int(config.get("fused_block_v", 64)),
            "block_k": int(config.get("fused_block_k", 16)),
            "max_seq": int(config.get("fused_max_seq", 32)),
            "num_warps": int(config.get("fused_num_warps", 8)),
        }

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        dense = triton_linear(
            hidden_states,
            self.dense.weight,
            self.dense.bias,
            config=self._ks_linear_config,
        )
        normalized = gelu_layer_norm(
            dense,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.layer_norm.eps,
            config=self._ks_layer_norm_config,
        )
        if self._ks_variant == "fused_pool":
            pooled = triton_decoder_pool(
                normalized,
                self.decoder.weight,
                self.decoder.bias,
                seq_lens,
                pooling=self.pooling,
                config=self._ks_fused_config,
            )
            return [pooled[index] for index in range(pooled.shape[0])]
        logits = triton_linear(
            normalized,
            self.decoder.weight,
            self.decoder.bias,
            log1p_relu=True,
            config=self._ks_linear_config,
        )
        batch_size = seq_lens.numel()
        vocab_size = logits.shape[1]
        pooled = torch.empty(
            (batch_size, vocab_size), device=logits.device, dtype=logits.dtype
        )
        block_v = self._ks_pool_config["block_v"]
        _pool_logits_kernel[(batch_size, triton.cdiv(vocab_size, block_v))](
            logits,
            seq_lens,
            pooled,
            batch_size,
            logits.shape[0],
            vocab_size,
            POOL_MAX=self.pooling == "max",
            BLOCK_T=self._ks_pool_config["block_t"],
            BLOCK_V=block_v,
            num_warps=self._ks_pool_config["num_warps"],
        )
        return [pooled[index] for index in range(batch_size)]


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(83, 768, device="cuda")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]
