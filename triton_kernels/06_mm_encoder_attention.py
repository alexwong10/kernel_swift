import torch
import torch.nn as nn

from common import triton_attention

_KS_PROFILE = {
    "task_key": "06_mm_encoder_attention",
    "chip_key": "portable_default",
    "variant": "full_row_diagnostic",
    "config": {"num_warps": 8},
}


def get_operator_profile(task_key, chip_key=None):
    if task_key != _KS_PROFILE["task_key"]:
        raise ValueError(f"unsupported task profile: {task_key}")
    return _KS_PROFILE


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size**0.5)
        profile = get_operator_profile("06_mm_encoder_attention")
        if profile["variant"] != "full_row_diagnostic":
            raise ValueError(f"unsupported MMEncoderAttention variant: {profile['variant']}")
        self._ks_attention_config = profile["config"]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, q_len = query.shape[:2]
        kv_len = key.shape[1]
        if self.num_heads != self.num_kv_heads:
            raise ValueError("num_heads != num_kv_heads requires a separately verified variant")
        q = query.view(batch_size, q_len, self.num_heads, self.head_size)
        k = key.view(batch_size, kv_len, self.num_kv_heads, self.head_size)
        v = value.view(batch_size, kv_len, self.num_kv_heads, self.head_size)
        out = triton_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal=False,
            config=self._ks_attention_config,
        )
        return out.reshape(batch_size, q_len, self.num_heads * self.head_size)


def get_inputs():
    batch_size, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(batch_size, seq_len, hidden, dtype=dtype, device="cuda")
    key = torch.randn(batch_size, seq_len, hidden, dtype=dtype, device="cuda")
    value = torch.randn(batch_size, seq_len, hidden, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
