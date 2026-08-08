import torch
import torch.nn as nn

from common import triton_attention

_KS_PROFILE = {
    "task_key": "03_flex_attention",
    "chip_key": "portable_default",
    "variant": "full_row_diagnostic",
    "config": {"num_warps": 8},
}


def get_operator_profile(task_key, chip_key=None):
    if task_key != _KS_PROFILE["task_key"]:
        raise ValueError(f"unsupported task profile: {task_key}")
    return _KS_PROFILE


class ModelNew(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        head_size: int = 64,
        scale: float = None,
        num_kv_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size**0.5)
        self.num_kv_heads = num_kv_heads
        profile = get_operator_profile("03_flex_attention")
        if profile["variant"] != "full_row_diagnostic":
            raise ValueError(f"unsupported FlexAttention variant: {profile['variant']}")
        self._ks_attention_config = profile["config"]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        num_tokens = query.shape[0]
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            # This path is not used by the official case; keeping the view
            # expansion here preserves the reference interface.
            key = key[:, :, None, :].expand(-1, -1, repeat, -1).reshape(
                num_tokens, self.num_heads, self.head_size
            )
            value = value[:, :, None, :].expand(-1, -1, repeat, -1).reshape(
                num_tokens, self.num_heads, self.head_size
            )
        q = query.unsqueeze(0)
        k = key.unsqueeze(0)
        v = value.unsqueeze(0)
        out = triton_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal=True,
            config=self._ks_attention_config,
        )
        return out.reshape(num_tokens, self.num_heads * self.head_size)


def get_inputs():
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]
