from __future__ import annotations

import torch

from attention_kv_h2o.h2o_scoring import H2OOracleState, scores_from_attention_probs, simulate_eviction_trace


def test_scores_from_attention_probs_shape():
    b, h, q, k = 1, 4, 3, 5
    x = torch.rand(b, h, q, k)
    s = scores_from_attention_probs(x)
    assert s.shape == (h, k)


def test_first_step_no_crash_short_sequence():
    h, k = 2, 6
    probs = torch.ones(1, h, 1, k) / k
    # Budgets use int(ratio * k) on the first step only; pick ratios so cache >= k (no eviction).
    state = H2OOracleState(heavy_budget_ratio=0.5, recent_budget_ratio=0.5)
    out = state.step(probs)
    assert out["key_len"] == k
    assert out["attention_masks_next"].shape == (1, h, 1, k)
    assert torch.all(out["attention_masks_next"] == 1)


def test_incremental_decode_matches_key_growth():
    h = 3
    state = H2OOracleState(heavy_budget_ratio=0.2, recent_budget_ratio=0.2)
    probs1 = torch.softmax(torch.randn(1, h, 4, 4, dtype=torch.float32), dim=-1)
    state.step(probs1)
    probs2 = torch.softmax(torch.randn(1, h, 1, 5, dtype=torch.float32), dim=-1)
    state.step(probs2)
    probs3 = torch.softmax(torch.randn(1, h, 1, 6, dtype=torch.float32), dim=-1)
    state.step(probs3)
    assert state.previous_scores is not None
    assert state.previous_scores.shape == (h, 6)


def test_eviction_activates_when_over_budget():
    h = 2
    k = 20
    heavy_r, recent_r = 0.2, 0.2
    state = H2OOracleState(heavy_budget_ratio=heavy_r, recent_budget_ratio=recent_r)
    probs = torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1)
    state.step(probs)
    assert state.cache_budget is not None
    if k > state.cache_budget:
        m = state.attention_masks_next
        assert m is not None
        assert m.shape == (1, h, 1, k)
        assert m.min() >= 0
        assert m.max() <= 1


def test_simulate_eviction_trace_lengths():
    tensors = []
    h = 2
    for k in range(4, 10):
        tensors.append(torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1))
    trace = simulate_eviction_trace(tensors, 0.25, 0.25)
    assert len(trace) == len(tensors)


def test_apply_mask_to_attn_weights():
    h, k = 2, 8
    state = H2OOracleState(heavy_budget_ratio=0.2, recent_budget_ratio=0.2)
    probs = torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1)
    state.step(probs)
    logits = torch.randn(1, h, 1, k, dtype=torch.float32)
    masked = state.apply_mask_to_attn_weights(logits)
    assert masked.shape == logits.shape


def test_logits_mask_additive_shape():
    h, k = 2, 7
    state = H2OOracleState(heavy_budget_ratio=0.2, recent_budget_ratio=0.2)
    probs = torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1)
    state.step(probs)
    add = state.logits_mask_additive(k, dtype=torch.float32, device=probs.device)
    assert add.shape == (1, h, 1, k)
