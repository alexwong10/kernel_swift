import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


_KS_PROFILE = {
    "task_key": "05_music_flamingo_rotary_embedding",
    "chip_key": "portable_default",
    "variant": "tiled_frequency",
    "config": {"block_channel": 128, "num_warps": 4},
}


def get_operator_profile(task_key, chip_key=None):
    if task_key != _KS_PROFILE["task_key"]:
        raise ValueError(f"unsupported task profile: {task_key}")
    return _KS_PROFILE


@triton.jit
def _music_rope_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    channel = offsets % (2 * dim)
    position = (offsets // (2 * dim)) % seq_len
    batch = offsets // (seq_len * 2 * dim)
    timestamp = tl.load(timestamps_ptr + batch * seq_len + position, mask=mask, other=0.0)
    local_channel = channel % dim
    inv = tl.load(inv_freq_ptr + local_channel // 2, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (batch.to(tl.float32) / max_seq_len) * inv
    time_freq = tl.load(
        position_angles_ptr + position * dim + local_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq = tl.where(channel < dim, batch_freq, time_freq)
    angle = -timestamp.to(tl.float32) * 6.283185307179586
    phase = freq * angle
    tl.store(cos_ptr + offsets, tl.cos(phase), mask=mask)
    tl.store(sin_ptr + offsets, tl.sin(phase), mask=mask)


@triton.jit
def _music_rope_tiled_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute one batch/position channel tile with shared scalar indexing."""
    batch = tl.program_id(0)
    position = tl.program_id(1)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_count = 2 * dim
    mask = channel < channel_count

    timestamp = tl.load(timestamps_ptr + batch * seq_len + position).to(tl.float32)
    local_channel = channel % dim
    inv = tl.load(inv_freq_ptr + local_channel // 2, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (batch.to(tl.float32) / max_seq_len) * inv
    time_freq = tl.load(
        position_angles_ptr + position * dim + local_channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq = tl.where(channel < dim, batch_freq, time_freq)
    phase = freq * (-timestamp * 6.283185307179586)
    output = (batch * seq_len + position) * channel_count + channel
    tl.store(cos_ptr + output, tl.cos(phase), mask=mask)
    tl.store(sin_ptr + output, tl.sin(phase), mask=mask)


@triton.jit
def _music_rope_paired_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Compute the two repeated channels of each rotary frequency together.

    ``position_angles`` is built with ``repeat_interleave(2)`` and the first
    half of the output uses the same repeated ``inv_freq`` layout.  Thus the
    phase for channels ``2p`` and ``2p + 1`` is identical in both halves of
    the output.  Keeping one phase per pair cuts the expensive sin/cos calls
    in half without changing the output ordering or arithmetic.
    """
    batch = tl.program_id(0)
    position = tl.program_id(1)
    pair = tl.program_id(2) * BLOCK_P + tl.arange(0, BLOCK_P)
    pair_count = dim // 2
    mask = pair < pair_count
    channel = pair * 2

    timestamp = tl.load(timestamps_ptr + batch * seq_len + position).to(tl.float32)
    inv = tl.load(inv_freq_ptr + pair, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (batch.to(tl.float32) / max_seq_len) * inv
    time_freq = tl.load(
        position_angles_ptr + position * dim + channel,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    angle = -timestamp * 6.283185307179586
    batch_phase = batch_freq * angle
    time_phase = time_freq * angle

    base = (batch * seq_len + position) * (2 * dim)
    batch_output = base + channel
    time_output = base + dim + channel
    batch_cos = tl.cos(batch_phase)
    batch_sin = tl.sin(batch_phase)
    time_cos = tl.cos(time_phase)
    time_sin = tl.sin(time_phase)
    tl.store(cos_ptr + batch_output, batch_cos, mask=mask)
    tl.store(cos_ptr + batch_output + 1, batch_cos, mask=mask)
    tl.store(sin_ptr + batch_output, batch_sin, mask=mask)
    tl.store(sin_ptr + batch_output + 1, batch_sin, mask=mask)
    tl.store(cos_ptr + time_output, time_cos, mask=mask)
    tl.store(cos_ptr + time_output + 1, time_cos, mask=mask)
    tl.store(sin_ptr + time_output, time_sin, mask=mask)
    tl.store(sin_ptr + time_output + 1, time_sin, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        self.register_buffer("position_angles", position_angles.repeat_interleave(2, dim=-1))
        profile = get_operator_profile("05_music_flamingo_rotary_embedding")
        if profile["variant"] not in {
            "fused_elementwise",
            "tiled_frequency",
            "paired_tiled_frequency",
        }:
            raise ValueError(f"unsupported MusicFlamingo variant: {profile['variant']}")
        self._ks_variant = profile["variant"]
        self._ks_config = profile["config"]

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        batch_size = timestamps.shape[0]
        dim = self.position_angles.shape[1]
        shape = (batch_size, seq_len, 2 * dim)
        cos = torch.empty(shape, device=timestamps.device, dtype=torch.float32)
        sin = torch.empty_like(cos)
        if self._ks_variant == "paired_tiled_frequency":
            if dim % 2 != 0:
                raise ValueError("paired_tiled_frequency requires an even dim")
            block_pair = int(self._ks_config.get("block_pair", 32))
            if block_pair <= 0:
                raise ValueError("block_pair must be positive")
            _music_rope_paired_kernel[
                (
                    batch_size,
                    seq_len,
                    triton.cdiv(dim // 2, block_pair),
                )
            ](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                seq_len,
                dim,
                self.max_seq_len,
                BLOCK_P=block_pair,
                num_warps=int(self._ks_config["num_warps"]),
            )
        elif self._ks_variant == "tiled_frequency":
            block_channel = int(self._ks_config["block_channel"])
            if block_channel <= 0:
                raise ValueError("block_channel must be positive")
            _music_rope_tiled_kernel[
                (
                    batch_size,
                    seq_len,
                    triton.cdiv(2 * dim, block_channel),
                )
            ](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                seq_len,
                dim,
                self.max_seq_len,
                BLOCK_C=block_channel,
                num_warps=int(self._ks_config["num_warps"]),
            )
        else:
            total = cos.numel()
            block = int(self._ks_config["block"])
            _music_rope_kernel[(triton.cdiv(total, block),)](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                seq_len,
                dim,
                self.max_seq_len,
                total,
                BLOCK=block,
                num_warps=int(self._ks_config["num_warps"]),
            )
        return cos, sin


def get_inputs():
    batch_size, seq_len = 4, 32
    timestamps = torch.rand(batch_size, seq_len, device="cuda")
    return [timestamps, seq_len]


def get_init_inputs():
    return [64, 256, 10000.0]
