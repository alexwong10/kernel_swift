import torch
import torch.nn as nn
import triton
import triton.language as tl


# Portable default for direct KernelSwift upload.  The artifact builder bakes
# the selected chip profile over this definition for formal cell runs.
_KS_PROFILE = {
    "task_key": "02_fused_moe",
    "chip_key": "portable_default",
    "variant": "tiled_dot_fp16_reference",
    "config": {
        "route_num_warps": 1,
        "gate_up_num_warps": 4,
        "down_num_warps": 4,
        "reduce_block": 256,
        "reduce_num_warps": 4,
    },
}


def get_operator_profile(task_key, chip_key=None):
    if task_key != _KS_PROFILE["task_key"]:
        raise ValueError(f"unsupported task profile: {task_key}")
    return _KS_PROFILE


@triton.jit
def _moe_route_kernel(
    logits_ptr,
    ids_ptr,
    weights_ptr,
    num_experts: tl.constexpr,
    TOP_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK_E)
    valid = expert < num_experts
    logits = tl.load(logits_ptr + token * num_experts + expert, mask=valid, other=-float("inf"))
    logits = logits.to(tl.float32)
    scores = tl.exp(logits - tl.max(logits, axis=0))
    scores = scores / tl.sum(tl.where(valid, scores, 0.0), axis=0)
    remaining = tl.where(valid, scores, -float("inf"))
    selected_sum = 0.0
    for rank in range(TOP_K):
        expert_id = tl.argmax(remaining, axis=0)
        weight = tl.max(remaining, axis=0)
        tl.store(ids_ptr + token * TOP_K + rank, expert_id)
        tl.store(weights_ptr + token * TOP_K + rank, weight)
        selected_sum += weight
        remaining = tl.where(expert == expert_id, -float("inf"), remaining)
    rank_offsets = tl.arange(0, TOP_K)
    selected = tl.load(weights_ptr + token * TOP_K + rank_offsets)
    if RENORMALIZE:
        selected /= selected_sum
    tl.store(weights_ptr + token * TOP_K + rank_offsets, selected)


@triton.jit
def _moe_gate_up_kernel(
    x_ptr,
    ids_ptr,
    w1_ptr,
    act_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    token = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(ids_ptr + token * TOP_K + rank)
    h = tl.arange(0, BLOCK_H)
    i = tl.arange(0, BLOCK_I)
    h_mask = h < hidden_size
    i_mask = i < intermediate_size
    # The reference first casts the parameter tensors to the input dtype and
    # performs a GEMM.  Keep the operands in fp16 here so the dot path has the
    # same input rounding as ``x_e @ w1[e].T`` rather than an fp32 outer
    # product followed by a reduction.
    x = tl.load(x_ptr + token * hidden_size + h, mask=h_mask, other=0.0)
    expert_base = expert * 2 * intermediate_size * hidden_size
    gate_w = tl.load(
        w1_ptr + expert_base + i[:, None] * hidden_size + h[None, :],
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    up_w = tl.load(
        w1_ptr
        + expert_base
        + (intermediate_size + i[:, None]) * hidden_size
        + h[None, :],
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    # Triton-Ascend does not accept constexpr indexing of a singleton dot
    # dimension (``[:, 0]``).  Reducing that dimension is algebraically the
    # same operation and keeps the tiled dot path portable across forks.
    gate = tl.sum(tl.dot(gate_w, x[:, None]), axis=1).to(tl.float16)
    up = tl.sum(tl.dot(up_w, x[:, None]), axis=1).to(tl.float16)
    act = gate * tl.sigmoid(gate) * up
    tl.store(
        act_ptr + (token * TOP_K + rank) * intermediate_size + i,
        act.to(tl.float16),
        mask=i_mask,
    )


@triton.jit
def _moe_down_kernel(
    act_ptr,
    ids_ptr,
    route_weights_ptr,
    w2_ptr,
    contribution_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    token = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(ids_ptr + token * TOP_K + rank)
    # ``topk_weights`` is converted to the hidden-state dtype by the
    # reference before the expert output is weighted.
    route_weight = tl.load(route_weights_ptr + token * TOP_K + rank).to(tl.float16)
    h = tl.arange(0, BLOCK_H)
    i = tl.arange(0, BLOCK_I)
    h_mask = h < hidden_size
    i_mask = i < intermediate_size
    act = tl.load(
        act_ptr + (token * TOP_K + rank) * intermediate_size + i,
        mask=i_mask,
        other=0.0,
    )
    expert_base = expert * hidden_size * intermediate_size
    w2 = tl.load(
        w2_ptr + expert_base + h[:, None] * intermediate_size + i[None, :],
        mask=h_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    down = tl.sum(tl.dot(w2, act[:, None]), axis=1).to(tl.float16) * route_weight
    tl.store(
        contribution_ptr + (token * TOP_K + rank) * hidden_size + h,
        down.to(tl.float16),
        mask=h_mask,
    )


@triton.jit
def _moe_gate_up_scalar_kernel(
    x_ptr,
    ids_ptr,
    w1_ptr,
    act_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Elementwise GEMV fallback for backends whose dot N dimension is < 16."""
    token = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(ids_ptr + token * TOP_K + rank)
    h = tl.arange(0, BLOCK_H)
    i = tl.arange(0, BLOCK_I)
    h_mask = h < hidden_size
    i_mask = i < intermediate_size
    x = tl.load(x_ptr + token * hidden_size + h, mask=h_mask, other=0.0).to(tl.float16)
    expert_base = expert * 2 * intermediate_size * hidden_size
    gate_w = tl.load(
        w1_ptr + expert_base + i[:, None] * hidden_size + h[None, :],
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    up_w = tl.load(
        w1_ptr
        + expert_base
        + (intermediate_size + i[:, None]) * hidden_size
        + h[None, :],
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    # Some vendor Triton forks reject tl.dot for a [I,H] x [H,1] GEMV because
    # the non-batch N dimension is one.  The broadcasted product is equivalent
    # and keeps the reduction in the same fp16-input/fp32-accumulation domain.
    gate = tl.sum(gate_w * x[None, :], axis=1).to(tl.float16)
    up = tl.sum(up_w * x[None, :], axis=1).to(tl.float16)
    act = gate * tl.sigmoid(gate.to(tl.float32)).to(tl.float16) * up
    tl.store(
        act_ptr + (token * TOP_K + rank) * intermediate_size + i,
        act.to(tl.float16),
        mask=i_mask,
    )


@triton.jit
def _moe_down_scalar_kernel(
    act_ptr,
    ids_ptr,
    route_weights_ptr,
    w2_ptr,
    contribution_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Elementwise GEMV fallback matching ``_moe_down_kernel``."""
    token = tl.program_id(0)
    rank = tl.program_id(1)
    expert = tl.load(ids_ptr + token * TOP_K + rank)
    route_weight = tl.load(route_weights_ptr + token * TOP_K + rank).to(tl.float16)
    h = tl.arange(0, BLOCK_H)
    i = tl.arange(0, BLOCK_I)
    h_mask = h < hidden_size
    i_mask = i < intermediate_size
    act = tl.load(
        act_ptr + (token * TOP_K + rank) * intermediate_size + i,
        mask=i_mask,
        other=0.0,
    ).to(tl.float16)
    expert_base = expert * hidden_size * intermediate_size
    w2 = tl.load(
        w2_ptr + expert_base + h[:, None] * intermediate_size + i[None, :],
        mask=h_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    down = tl.sum(w2 * act[None, :], axis=1).to(tl.float16) * route_weight
    tl.store(
        contribution_ptr + (token * TOP_K + rank) * hidden_size + h,
        down.to(tl.float16),
        mask=h_mask,
    )


@triton.jit
def _moe_route_pack_kernel(
    logits_ptr,
    weights_ptr,
    counts_ptr,
    packed_routes_ptr,
    num_experts: tl.constexpr,
    TOP_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOTAL_ROUTES: tl.constexpr,
):
    """Route and compact token/rank routes in one Triton launch.

    This is the grouped variant's only routing launch.  It preserves the
    reference top-k/renormalization arithmetic, then uses ``atomic_add`` only
    to reserve a unique per-expert slot for each selected route.  The verified
    scalar path keeps using ``_moe_route_kernel`` unchanged.
    """
    token = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    valid = expert_offsets < num_experts
    logits = tl.load(
        logits_ptr + token * num_experts + expert_offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.exp(logits - tl.max(logits, axis=0))
    scores = scores / tl.sum(tl.where(valid, scores, 0.0), axis=0)
    remaining = tl.where(valid, scores, -float("inf"))
    selected_sum = 0.0
    for rank in range(TOP_K):
        expert_id = tl.argmax(remaining, axis=0)
        weight = tl.max(remaining, axis=0)
        route = token * TOP_K + rank
        tl.store(weights_ptr + route, weight)
        slot = tl.atomic_add(counts_ptr + expert_id, 1)
        tl.store(packed_routes_ptr + expert_id * TOTAL_ROUTES + slot, route)
        selected_sum += weight
        remaining = tl.where(expert_offsets == expert_id, -float("inf"), remaining)
    rank_offsets = tl.arange(0, TOP_K)
    selected = tl.load(weights_ptr + token * TOP_K + rank_offsets)
    if RENORMALIZE:
        selected /= selected_sum
    tl.store(weights_ptr + token * TOP_K + rank_offsets, selected)


@triton.jit
def _moe_gate_up_grouped_kernel(
    x_ptr,
    packed_routes_ptr,
    counts_ptr,
    w1_ptr,
    act_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    TOTAL_ROUTES: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Batched gate/up dot for routes belonging to one expert.

    The route compaction makes the ``BLOCK_R`` dimension dense for each
    expert.  Invalid tail rows are masked, and the masked rows never store to
    ``act_ptr``.  Weight layout remains [2I, H], but is loaded as [H, I] so
    that the dot is [R, H] x [H, I].
    """
    expert = tl.program_id(0)
    route_block = tl.program_id(1)
    rows = route_block * BLOCK_R + tl.arange(0, BLOCK_R)
    count = tl.load(counts_ptr + expert)
    row_mask = rows < count
    packed_base = expert * TOTAL_ROUTES
    routes = tl.load(
        packed_routes_ptr + packed_base + rows,
        mask=row_mask,
        other=0,
    )
    tokens = routes // TOP_K
    h = tl.arange(0, BLOCK_H)
    i = tl.arange(0, BLOCK_I)
    h_mask = h < hidden_size
    i_mask = i < intermediate_size
    x = tl.load(
        x_ptr + tokens[:, None] * hidden_size + h[None, :],
        mask=row_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    expert_base = expert * 2 * intermediate_size * hidden_size
    gate_w = tl.load(
        w1_ptr + expert_base + i[None, :] * hidden_size + h[:, None],
        mask=h_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    up_w = tl.load(
        w1_ptr
        + expert_base
        + (intermediate_size + i[None, :]) * hidden_size
        + h[:, None],
        mask=h_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    gate = tl.dot(x, gate_w).to(tl.float16)
    up = tl.dot(x, up_w).to(tl.float16)
    act = gate * tl.sigmoid(gate.to(tl.float32)).to(tl.float16) * up
    tl.store(
        act_ptr + routes[:, None] * intermediate_size + i[None, :],
        act.to(tl.float16),
        mask=row_mask[:, None] & i_mask[None, :],
    )


@triton.jit
def _moe_down_grouped_kernel(
    packed_routes_ptr,
    counts_ptr,
    act_ptr,
    route_weights_ptr,
    w2_ptr,
    contribution_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOTAL_ROUTES: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Batched down projection for compacted routes of one expert."""
    expert = tl.program_id(0)
    route_block = tl.program_id(1)
    rows = route_block * BLOCK_R + tl.arange(0, BLOCK_R)
    count = tl.load(counts_ptr + expert)
    row_mask = rows < count
    packed_base = expert * TOTAL_ROUTES
    routes = tl.load(
        packed_routes_ptr + packed_base + rows,
        mask=row_mask,
        other=0,
    )
    i = tl.arange(0, BLOCK_I)
    h = tl.arange(0, BLOCK_H)
    i_mask = i < intermediate_size
    h_mask = h < hidden_size
    act = tl.load(
        act_ptr + routes[:, None] * intermediate_size + i[None, :],
        mask=row_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    expert_base = expert * hidden_size * intermediate_size
    # [I, H] view of the row-major [H, I] down matrix.
    down_w = tl.load(
        w2_ptr + expert_base + h[None, :] * intermediate_size + i[:, None],
        mask=i_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    down = tl.dot(act, down_w).to(tl.float16)
    route_weight = tl.load(route_weights_ptr + routes).to(tl.float16)
    down = down * route_weight[:, None]
    tl.store(
        contribution_ptr + routes[:, None] * hidden_size + h[None, :],
        down.to(tl.float16),
        mask=row_mask[:, None] & h_mask[None, :],
    )


@triton.jit
def _moe_reduce_kernel(
    contribution_ptr,
    out_ptr,
    total: tl.constexpr,
    hidden_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total
    token = offsets // hidden_size
    hidden = offsets % hidden_size
    result = tl.zeros((BLOCK,), tl.float32)
    for rank in range(TOP_K):
        value = tl.load(
            contribution_ptr + (token * TOP_K + rank) * hidden_size + hidden,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        result += value
    tl.store(out_ptr + offsets, result, mask=valid)


class ModelNew(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        profile = get_operator_profile("02_fused_moe")
        if profile["variant"] not in {
            "tiled_dot_fp16_reference",
            "scalar_elementwise_fallback",
            "expert_grouped_dot",
        }:
            raise ValueError(f"unsupported FusedMoE variant: {profile['variant']}")
        self._ks_variant = profile["variant"]
        self._ks_config = profile["config"]

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        ids = None
        if self._ks_variant != "expert_grouped_dot":
            ids = torch.empty(
                (num_tokens, self.top_k), device=hidden_states.device, dtype=torch.int32
            )
        route_weights = torch.empty(
            (num_tokens, self.top_k), device=hidden_states.device, dtype=hidden_states.dtype
        )
        block_e = triton.next_power_of_2(self.num_experts)
        total_routes = num_tokens * self.top_k
        if self._ks_variant == "expert_grouped_dot":
            # Allocate the compaction buffers before routing so the grouped
            # route kernel can produce ids, weights, and the expert segments
            # in one launch.
            packed_counts = torch.zeros(
                (self.num_experts,), device=hidden_states.device, dtype=torch.int32
            )
            packed_routes = torch.empty(
                (self.num_experts, total_routes),
                device=hidden_states.device,
                dtype=torch.int32,
            )
            _moe_route_pack_kernel[(num_tokens,)](
                router_logits,
                route_weights,
                packed_counts,
                packed_routes,
                self.num_experts,
                TOP_K=self.top_k,
                RENORMALIZE=self.renormalize,
                BLOCK_E=block_e,
                TOTAL_ROUTES=total_routes,
                num_warps=int(self._ks_config["route_num_warps"]),
            )
        else:
            _moe_route_kernel[(num_tokens,)](
                router_logits,
                ids,
                route_weights,
                self.num_experts,
                TOP_K=self.top_k,
                RENORMALIZE=self.renormalize,
                BLOCK_E=block_e,
                num_warps=int(self._ks_config["route_num_warps"]),
            )
        act = torch.empty(
            (num_tokens, self.top_k, self.intermediate_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        block_h = triton.next_power_of_2(self.hidden_size)
        block_i = triton.next_power_of_2(self.intermediate_size)
        if self._ks_variant == "expert_grouped_dot":
            # The route-pack kernel above produced only a Triton-generated
            # permutation.  No framework sort or PyTorch fallback is used.
            group_block_routes = int(self._ks_config["group_block_routes"])
            _moe_gate_up_grouped_kernel[
                (self.num_experts, triton.cdiv(total_routes, group_block_routes))
            ](
                hidden_states,
                packed_routes,
                packed_counts,
                self.w1,
                act,
                self.hidden_size,
                self.intermediate_size,
                TOP_K=self.top_k,
                TOTAL_ROUTES=total_routes,
                BLOCK_R=group_block_routes,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                num_warps=int(self._ks_config["group_gate_num_warps"]),
            )
        else:
            gate_up_kernel = (
                _moe_gate_up_scalar_kernel
                if self._ks_variant == "scalar_elementwise_fallback"
                else _moe_gate_up_kernel
            )
            gate_up_kernel[(num_tokens, self.top_k)](
                hidden_states,
                ids,
                self.w1,
                act,
                self.hidden_size,
                self.intermediate_size,
                TOP_K=self.top_k,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                num_warps=int(self._ks_config["gate_up_num_warps"]),
            )
        contributions = torch.empty(
            (num_tokens, self.top_k, self.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if self._ks_variant == "expert_grouped_dot":
            group_block_routes = int(self._ks_config["group_block_routes"])
            _moe_down_grouped_kernel[
                (self.num_experts, triton.cdiv(total_routes, group_block_routes))
            ](
                packed_routes,
                packed_counts,
                act,
                route_weights,
                self.w2,
                contributions,
                self.hidden_size,
                self.intermediate_size,
                TOTAL_ROUTES=total_routes,
                BLOCK_R=group_block_routes,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                num_warps=int(self._ks_config["group_down_num_warps"]),
            )
        else:
            down_kernel = (
                _moe_down_scalar_kernel
                if self._ks_variant == "scalar_elementwise_fallback"
                else _moe_down_kernel
            )
            down_kernel[(num_tokens, self.top_k)](
                act,
                ids,
                route_weights,
                self.w2,
                contributions,
                self.hidden_size,
                self.intermediate_size,
                TOP_K=self.top_k,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                num_warps=int(self._ks_config["down_num_warps"]),
            )
        out = torch.empty_like(hidden_states)
        total = out.numel()
        block = int(self._ks_config["reduce_block"])
        _moe_reduce_kernel[(triton.cdiv(total, block),)](
            contributions,
            out,
            total,
            self.hidden_size,
            TOP_K=self.top_k,
            BLOCK=block,
            num_warps=int(self._ks_config["reduce_num_warps"]),
        )
        return out


def get_inputs():
    hidden_states = torch.randn(83, 128, dtype=torch.float16, device="cuda")
    router_logits = torch.randn(83, 8, dtype=torch.float32, device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
