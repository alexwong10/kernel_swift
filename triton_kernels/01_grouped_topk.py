import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_experts: tl.constexpr,
    experts_per_group: tl.constexpr,
    num_groups: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_GROUPS: tl.constexpr,
    SOFTMAX: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    mask = expert < num_experts
    logits = tl.load(logits_ptr + token * num_experts + expert, mask=mask, other=-float("inf"))
    logits = logits.to(tl.float32)
    if SOFTMAX:
        shifted = logits - tl.max(logits, axis=0)
        scores = tl.exp(shifted)
        scores = scores / tl.sum(tl.where(mask, scores, 0.0), axis=0)
    else:
        scores = tl.sigmoid(logits)

    grouped = tl.reshape(scores, (num_groups, experts_per_group))
    group_scores = tl.max(grouped, axis=1)
    group_offsets = tl.arange(0, num_groups)
    eligible = tl.zeros((BLOCK,), tl.int1)
    remaining_groups = group_scores
    for _ in range(TOPK_GROUPS):
        group_id = tl.argmax(remaining_groups, axis=0)
        eligible = eligible | ((expert // experts_per_group) == group_id)
        remaining_groups = tl.where(group_offsets == group_id, -float("inf"), remaining_groups)

    candidates = tl.where(mask & eligible, scores, -float("inf"))
    weight_sum = 0.0
    for k in range(TOPK):
        expert_id = tl.argmax(candidates, axis=0)
        weight = tl.max(candidates, axis=0)
        tl.store(weights_ptr + token * TOPK + k, weight)
        tl.store(ids_ptr + token * TOPK + k, expert_id)
        weight_sum += weight
        candidates = tl.where(expert == expert_id, -float("inf"), candidates)

    out_k = tl.arange(0, TOPK)
    selected = tl.load(weights_ptr + token * TOPK + out_k)
    if RENORMALIZE:
        selected = selected / weight_sum
    selected *= ROUTED_SCALE
    tl.store(weights_ptr + token * TOPK + out_k, selected)


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        num_tokens, num_experts = gating_output.shape
        experts_per_group = num_experts // self.num_expert_group
        weights = torch.empty(
            (num_tokens, self.topk), device=gating_output.device, dtype=torch.float32
        )
        ids = torch.empty(
            (num_tokens, self.topk), device=gating_output.device, dtype=torch.int32
        )
        block = triton.next_power_of_2(num_experts)
        _grouped_topk_kernel[(num_tokens,)](
            gating_output,
            weights,
            ids,
            num_experts,
            experts_per_group,
            self.num_expert_group,
            TOPK=self.topk,
            TOPK_GROUPS=self.topk_group,
            SOFTMAX=self.scoring_func == "softmax",
            RENORMALIZE=self.renormalize,
            ROUTED_SCALE=self.routed_scaling_factor,
            BLOCK=block,
            num_warps=4,
        )
        return weights, ids


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]

