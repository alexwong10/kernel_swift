"""NumPy checks for indexing/reduction formulas used by the Triton kernels.

This does not replace accelerator execution.  It catches algebra, layout and
axis mistakes without requiring PyTorch or a Triton runtime on the host.
"""

from __future__ import annotations

import math

import numpy as np


RNG = np.random.default_rng(2026)


def assert_close(name: str, lhs, rhs, atol: float = 1e-5) -> None:
    if not np.allclose(lhs, rhs, atol=atol, rtol=atol):
        diff = float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
        raise AssertionError(f"{name}: max diff {diff}")
    print("PASS", name)


def grouped_topk() -> None:
    logits = RNG.normal(size=(5, 32)).astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    scores = np.exp(shifted)
    scores /= scores.sum(axis=1, keepdims=True)
    grouped = scores.reshape(5, 4, 8)
    group_ids = np.argsort(-grouped.max(axis=2), axis=1)[:, :2]
    mask = np.zeros_like(grouped, dtype=bool)
    for token in range(5):
        mask[token, group_ids[token]] = True
    candidates = np.where(mask.reshape(5, 32), scores, -np.inf)
    ref_ids = np.argsort(-candidates, axis=1)[:, :4]
    ref_weights = np.take_along_axis(candidates, ref_ids, axis=1)
    ref_weights /= ref_weights.sum(axis=1, keepdims=True)

    got_ids = np.empty_like(ref_ids)
    got_weights = np.empty_like(ref_weights)
    for token in range(5):
        remaining_groups = grouped[token].max(axis=1).copy()
        eligible = np.zeros(32, dtype=bool)
        for _ in range(2):
            best_group_score = remaining_groups.max()
            group = int(np.flatnonzero(remaining_groups == best_group_score)[0])
            eligible |= np.arange(32) // 8 == group
            remaining_groups[group] = -np.inf
        remaining = np.where(eligible, scores[token], -np.inf)
        for rank in range(4):
            best_expert_score = remaining.max()
            expert = int(np.flatnonzero(remaining == best_expert_score)[0])
            got_ids[token, rank] = expert
            got_weights[token, rank] = remaining[expert]
            remaining[expert] = -np.inf
        got_weights[token] /= got_weights[token].sum()
    if not np.array_equal(ref_ids, got_ids):
        raise AssertionError("GroupedTopk ids differ")
    assert_close("01_grouped_topk", ref_weights, got_weights)

    # The manual fallback deliberately uses lowest-id tie breaking.  This is an
    # internal invariant only; each target runtime must still compare it with
    # that runtime's torch.topk before selecting the fallback profile.
    tied = np.ones(16, dtype=np.float32)
    remaining_groups = np.full(4, 4.0, dtype=np.float32)
    eligible = np.zeros(16, dtype=bool)
    for _ in range(2):
        group = int(np.flatnonzero(remaining_groups == remaining_groups.max())[0])
        eligible |= np.arange(16) // 4 == group
        remaining_groups[group] = -np.inf
    candidates = np.where(eligible, tied, -np.inf)
    tied_ids = []
    for _ in range(4):
        expert = int(np.flatnonzero(candidates == candidates.max())[0])
        tied_ids.append(expert)
        candidates[expert] = -np.inf
    if tied_ids != [0, 1, 2, 3]:
        raise AssertionError(f"manual GroupedTopk tie order differs: {tied_ids}")
    print("PASS", "01_grouped_topk_manual_tie_order")


def fused_moe() -> None:
    tokens, experts, top_k, hidden, intermediate = 7, 4, 2, 9, 5
    x = RNG.normal(size=(tokens, hidden)).astype(np.float32)
    logits = RNG.normal(size=(tokens, experts)).astype(np.float32)
    w1 = RNG.normal(scale=0.02, size=(experts, 2 * intermediate, hidden)).astype(np.float32)
    w2 = RNG.normal(scale=0.02, size=(experts, hidden, intermediate)).astype(np.float32)
    scores = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores /= scores.sum(axis=1, keepdims=True)
    ids = np.argsort(-scores, axis=1)[:, :top_k]
    weights = np.take_along_axis(scores, ids, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    ref = np.zeros_like(x)
    for token in range(tokens):
        for rank in range(top_k):
            expert = ids[token, rank]
            gate_up = w1[expert] @ x[token]
            gate, up = np.split(gate_up, 2)
            act = gate / (1.0 + np.exp(-gate)) * up
            ref[token] += weights[token, rank] * (w2[expert] @ act)
    staged = np.empty((tokens, top_k, intermediate), dtype=np.float32)
    for token in range(tokens):
        for rank in range(top_k):
            expert = ids[token, rank]
            gate = w1[expert, :intermediate] @ x[token]
            up = w1[expert, intermediate:] @ x[token]
            staged[token, rank] = gate / (1.0 + np.exp(-gate)) * up
    got = np.zeros_like(x)
    for token in range(tokens):
        for rank in range(top_k):
            got[token] += weights[token, rank] * (w2[ids[token, rank]] @ staged[token, rank])
    assert_close("02_fused_moe", ref, got)


def splade_sparse_pooler() -> None:
    """Exercise the complete SPLADE pipeline with the tiled layout formulas.

    The host does not have a PyTorch/Triton runtime, so this mirrors the
    reference with NumPy and separately evaluates the same staged operations
    used by ``triton_kernels/04_splade_sparse_pooler.py``.  Small dimensions
    keep this check fast while still covering padded GEMM tiles and uneven
    sequence segments.
    """
    batch, hidden, vocab = 3, 7, 11
    seq_lens = np.array([2, 4, 1], dtype=np.int32)
    total_tokens = int(seq_lens.sum())
    hidden_states = RNG.normal(size=(total_tokens, hidden)).astype(np.float32)
    dense_w = RNG.normal(scale=0.02, size=(hidden, hidden)).astype(np.float32)
    dense_b = RNG.normal(scale=0.02, size=(hidden,)).astype(np.float32)
    decoder_w = RNG.normal(scale=0.02, size=(vocab, hidden)).astype(np.float32)
    decoder_b = RNG.normal(scale=0.02, size=(vocab,)).astype(np.float32)

    # Reference chain: Linear -> exact GELU -> LayerNorm -> Linear ->
    # log1p(relu) -> per-sequence pooling.
    dense_ref = hidden_states @ dense_w.T + dense_b
    gelu_ref = 0.5 * dense_ref * (
        1.0 + np.vectorize(math.erf)(dense_ref / math.sqrt(2.0))
    )
    mean = gelu_ref.mean(axis=1, keepdims=True)
    variance = ((gelu_ref - mean) ** 2).mean(axis=1, keepdims=True)
    normalized_ref = (gelu_ref - mean) / np.sqrt(variance + 1e-12)
    logits_ref = normalized_ref @ decoder_w.T + decoder_b
    activated_ref = np.log1p(np.maximum(logits_ref, 0.0))

    # Staged kernel emulation.  Use explicit padded tiles for the two GEMMs,
    # then the same row-wise reductions and sequence offset calculation as
    # the Triton implementation.
    def tiled_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out = np.zeros((x.shape[0], weight.shape[0]), dtype=np.float32)
        block_m, block_n, block_k = 4, 8, 4
        for m0 in range(0, x.shape[0], block_m):
            for n0 in range(0, weight.shape[0], block_n):
                acc = np.zeros(
                    (
                        min(block_m, x.shape[0] - m0),
                        min(block_n, weight.shape[0] - n0),
                    ),
                    dtype=np.float32,
                )
                for k0 in range(0, x.shape[1], block_k):
                    acc += x[
                        m0 : m0 + block_m, k0 : k0 + block_k
                    ] @ weight[
                        n0 : n0 + block_n, k0 : k0 + block_k
                    ].T
                acc += bias[n0:n0 + block_n]
                out[m0:m0 + block_m, n0:n0 + block_n] = acc
        return out

    dense_got = tiled_linear(hidden_states, dense_w, dense_b)
    gelu_got = 0.5 * dense_got * (
        1.0 + np.vectorize(math.erf)(dense_got / math.sqrt(2.0))
    )
    mean_got = gelu_got.mean(axis=1, keepdims=True)
    variance_got = ((gelu_got - mean_got) ** 2).mean(axis=1, keepdims=True)
    normalized_got = (gelu_got - mean_got) / np.sqrt(variance_got + 1e-12)
    logits_got = tiled_linear(normalized_got, decoder_w, decoder_b)
    activated_got = np.log1p(np.maximum(logits_got, 0.0))
    assert_close(
        "04_splade_dense_gelu_layernorm", normalized_ref, normalized_got, atol=2e-6
    )
    assert_close(
        "04_splade_logits_activation", activated_ref, activated_got, atol=2e-6
    )

    for pooling in ("max", "sum"):
        ref_segments = []
        got_segments = []
        offset = 0
        for length in seq_lens:
            ref_chunk = activated_ref[offset:offset + int(length)]
            got_chunk = activated_got[offset:offset + int(length)]
            reducer = np.max if pooling == "max" else np.sum
            ref_segments.append(reducer(ref_chunk, axis=0))
            got_segments.append(reducer(got_chunk, axis=0))
            offset += int(length)
        assert_close(
            f"04_splade_pool_{pooling}",
            np.stack(ref_segments),
            np.stack(got_segments),
            atol=2e-6,
        )


def attention() -> None:
    batch, heads, dim = 2, 3, 5
    scale = dim**-0.5
    for causal, q_len, kv_len, name in (
        (True, 7, 7, "03_flex_attention"),
        (False, 5, 7, "06_mm_encoder_attention_q5_kv7"),
    ):
        q = RNG.normal(size=(batch, q_len, heads, dim))
        k = RNG.normal(size=(batch, kv_len, heads, dim))
        v = RNG.normal(size=k.shape)
        scores = np.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if causal:
            scores = np.where(
                np.arange(kv_len)[None, None, None, :]
                <= np.arange(q_len)[None, None, :, None],
                scores,
                -np.inf,
            )
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        ref = np.einsum("bhqk,bkhd->bqhd", probs, v)
        got = np.empty_like(ref)
        for b in range(batch):
            for query in range(q_len):
                for head in range(heads):
                    valid = (
                        np.arange(kv_len) <= query
                        if causal
                        else np.ones(kv_len, dtype=bool)
                    )
                    row = (k[b, :, head] @ q[b, query, head]) * scale
                    row = np.where(valid, row, -np.inf)
                    p = np.exp(row - row.max())
                    p /= p.sum()
                    got[b, query, head] = (p[:, None] * v[b, :, head]).sum(axis=0)
        assert_close(name, ref, got)

    # The MM encoder reference supports grouped-query attention.  Exercise the
    # same head expansion used by the Triton wrapper instead of only checking
    # the equal-head case from the default input generator.
    q_len, kv_len, q_heads, kv_heads, dim = 4, 6, 8, 2, 5
    q = RNG.normal(size=(1, q_len, q_heads, dim))
    k_small = RNG.normal(size=(1, kv_len, kv_heads, dim))
    v_small = RNG.normal(size=k_small.shape)
    repeat = q_heads // kv_heads
    k = np.repeat(k_small, repeat, axis=2)
    v = np.repeat(v_small, repeat, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) * (dim ** -0.5)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    ref = np.einsum("bhqk,bkhd->bqhd", probs, v)
    got = np.empty_like(ref)
    for query in range(q_len):
        for head in range(q_heads):
            row = (k[0, :, head] @ q[0, query, head]) * (dim ** -0.5)
            p = np.exp(row - row.max())
            p /= p.sum()
            got[0, query, head] = (p[:, None] * v[0, :, head]).sum(axis=0)
    assert_close("06_mm_encoder_attention_gqa", ref, got)


def music_rope() -> None:
    batch, seq, dim, max_seq = 4, 8, 12, 64
    timestamps = RNG.random((batch, seq)).astype(np.float32)
    inv = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    pos = np.arange(max_seq, dtype=np.float32) / max_seq * (2 * math.pi)
    position_angles = np.repeat(pos[:, None] * inv[None, :], 2, axis=1)
    batch_freq = np.repeat((np.arange(batch) / max_seq)[:, None] * inv[None, :], 2, axis=1)
    freqs = np.concatenate(
        [np.broadcast_to(batch_freq[:, None, :], (batch, seq, dim)),
         np.broadcast_to(position_angles[None, :seq, :], (batch, seq, dim))],
        axis=-1,
    )
    phase_ref = freqs * (-timestamps * 2 * math.pi)[..., None]
    phase_flat = np.empty_like(phase_ref)
    for b in range(batch):
        for s in range(seq):
            for channel in range(2 * dim):
                local = channel % dim
                frequency = (b / max_seq) * inv[local // 2]
                if channel >= dim:
                    frequency = position_angles[s, local]
                phase_flat[b, s, channel] = frequency * (-timestamps[b, s] * 2 * math.pi)
    assert_close("05_music_flamingo_cos", np.cos(phase_ref), np.cos(phase_flat))
    assert_close("05_music_flamingo_sin", np.sin(phase_ref), np.sin(phase_flat))


def mhc_post() -> None:
    batch0, batch1, mult, hidden = 2, 3, 4, 7
    x = RNG.normal(size=(batch0, batch1, hidden))
    residual = RNG.normal(size=(batch0, batch1, mult, hidden))
    post = RNG.normal(size=(batch0, batch1, mult, 1))
    comb = RNG.normal(size=(batch0, batch1, mult, mult))
    ref = x[..., None, :] * post + np.einsum("abmn,abmc->abnc", comb, residual)
    got = np.empty_like(ref)
    for a in range(batch0):
        for b in range(batch1):
            for out_mix in range(mult):
                for h in range(hidden):
                    got[a, b, out_mix, h] = x[a, b, h] * post[a, b, out_mix, 0]
                    for in_mix in range(mult):
                        got[a, b, out_mix, h] += comb[a, b, in_mix, out_mix] * residual[a, b, in_mix, h]
    assert_close("07_mhc_post", ref, got)


def sinkhorn() -> None:
    rows, hc, eps, iterations = 6, 4, 1e-6, 20
    mixes = RNG.normal(size=(rows, (2 + hc) * hc)).astype(np.float32)
    scale = np.array([0.5, 0.25, 1.0], dtype=np.float32)
    base = RNG.normal(scale=0.1, size=((2 + hc) * hc,)).astype(np.float32)
    pre_ref = 1.0 / (1.0 + np.exp(-(mixes[:, :hc] * scale[0] + base[:hc]))) + eps
    post_ref = 2.0 / (1.0 + np.exp(-(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])))
    matrix = mixes[:, 2 * hc:].reshape(rows, hc, hc) * scale[2] + base[2 * hc:].reshape(1, hc, hc)
    matrix = np.exp(matrix - matrix.max(axis=-1, keepdims=True))
    matrix = matrix / matrix.sum(axis=-1, keepdims=True) + eps
    matrix = matrix / (matrix.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iterations - 1):
        matrix = matrix / (matrix.sum(axis=-1, keepdims=True) + eps)
        matrix = matrix / (matrix.sum(axis=-2, keepdims=True) + eps)
    pre_got = np.empty_like(pre_ref)
    post_got = np.empty_like(post_ref)
    matrix_got = np.empty_like(matrix)
    for row in range(rows):
        pre_got[row] = 1.0 / (1.0 + np.exp(-(mixes[row, :hc] * scale[0] + base[:hc]))) + eps
        post_got[row] = 2.0 / (1.0 + np.exp(-(mixes[row, hc:2 * hc] * scale[1] + base[hc:2 * hc])))
        m = (mixes[row, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc)
        m = np.exp(m - m.max(axis=1, keepdims=True))
        m = m / m.sum(axis=1, keepdims=True) + eps
        m = m / (m.sum(axis=0, keepdims=True) + eps)
        for _ in range(iterations - 1):
            m = m / (m.sum(axis=1, keepdims=True) + eps)
            m = m / (m.sum(axis=0, keepdims=True) + eps)
        matrix_got[row] = m
    assert_close("08_sinkhorn_pre", pre_ref, pre_got)
    assert_close("08_sinkhorn_post", post_ref, post_got)
    assert_close("08_sinkhorn_comb", matrix, matrix_got)


def centre_augmentation() -> None:
    atoms, samples = 17, 4
    coords = RNG.normal(size=(atoms, 3)).astype(np.float32)
    mask = np.ones(atoms, dtype=np.float32)
    u1, u2, u3 = (RNG.random(samples).astype(np.float32) for _ in range(3))
    translation = RNG.normal(size=(samples, 3)).astype(np.float32)
    center = (coords * mask[:, None]).sum(axis=0, keepdims=True) / mask.sum()
    centered = coords - center
    qx = np.sqrt(1 - u1) * np.sin(2 * math.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * math.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * math.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * math.pi * u3)
    matrices = np.stack(
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy),
         2 * (qx*qy + qw*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx),
         2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx*qx + qy*qy)],
        axis=-1,
    ).reshape(samples, 3, 3)
    ref = np.einsum("sij,aj->sai", matrices, centered) + translation[:, None, :]
    got = np.empty_like(ref)
    for sample in range(samples):
        for atom in range(atoms):
            got[sample, atom] = matrices[sample] @ centered[atom] + translation[sample]
    assert_close("09_centre_random_augmentation", ref, got)

    # The reference's centre-only branch returns centered coordinates before
    # applying the output mask.  A sparse mask therefore affects the center,
    # but masked atoms remain present in the returned coordinates.
    sparse_mask = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.float32)
    sparse_center = (
        (coords * sparse_mask[:, None]).sum(axis=0, keepdims=True)
        / (sparse_mask.sum() + 1e-12)
    )
    centre_only_ref = np.broadcast_to(
        coords - sparse_center, (samples, atoms, 3)
    ).copy()
    centre_only_got = np.empty_like(centre_only_ref)
    for sample in range(samples):
        for atom in range(atoms):
            centre_only_got[sample, atom] = coords[atom] - sparse_center[0]
    assert_close(
        "09_centre_random_augmentation_centre_only",
        centre_only_ref,
        centre_only_got,
    )


def head_mix_bwd() -> None:
    x = RNG.normal(size=(2, 9, 4)).astype(np.float32)
    scale = RNG.normal(size=(1,)).astype(np.float32)
    base = RNG.normal(size=(4,)).astype(np.float32)
    grad_out = RNG.normal(size=x.shape).astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-(x * scale + base)))
    grad_z = grad_out * sigmoid * (1.0 - sigmoid)
    ref_input = grad_z * scale
    ref_base = grad_z.sum(axis=(0, 1))
    ref_scale = np.array([(grad_z * x).sum()], dtype=np.float32)
    got_input = np.empty_like(x)
    block_rows = 3
    num_chunks = (x.shape[0] * x.shape[1] + block_rows - 1) // block_rows
    partial_base = np.zeros((num_chunks, 4), dtype=np.float32)
    partial_scale = np.zeros((num_chunks, 4), dtype=np.float32)
    rows = x.reshape(-1, 4)
    grads = grad_out.reshape(-1, 4)
    for chunk in range(num_chunks):
        begin = chunk * block_rows
        end = min(begin + block_rows, rows.shape[0])
        for mix in range(4):
            sig = 1.0 / (
                1.0 + np.exp(-(rows[begin:end, mix] * scale[0] + base[mix]))
            )
            gz = grads[begin:end, mix] * sig * (1 - sig)
            got_input.reshape(-1, 4)[begin:end, mix] = gz * scale[0]
            partial_base[chunk, mix] = gz.sum()
            partial_scale[chunk, mix] = (gz * rows[begin:end, mix]).sum()
    got_base = partial_base.sum(axis=0)
    assert_close("10_head_mix_grad_input", ref_input, got_input)
    assert_close("10_head_mix_grad_base", ref_base, got_base)
    assert_close("10_head_mix_grad_scale", ref_scale, np.array([partial_scale.sum()]))


def main() -> None:
    grouped_topk()
    fused_moe()
    splade_sparse_pooler()
    attention()
    music_rope()
    mhc_post()
    sinkhorn()
    centre_augmentation()
    head_mix_bwd()


if __name__ == "__main__":
    main()
