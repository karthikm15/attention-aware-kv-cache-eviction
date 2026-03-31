from __future__ import annotations

import torch

from attention_kv_h2o import FIFOKVCachePolicy, LRUKVCachePolicy


def test_fifo_keeps_last_cache_budget_tokens():
    h, k = 2, 6
    state = FIFOKVCachePolicy(cache_budget=3)
    probs = torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1)

    out = state.step(probs)
    mask = out["attention_masks_next"]
    assert mask is not None
    assert mask.shape == (1, h, 1, k)

    expected = torch.tensor([0, 0, 0, 1, 1, 1], dtype=mask.dtype, device=mask.device)
    for head_idx in range(h):
        assert torch.equal(mask[0, head_idx, 0], expected)


def test_fifo_mask_helpers_work():
    h, k = 2, 5
    state = FIFOKVCachePolicy(cache_budget=2)
    probs = torch.softmax(torch.randn(1, h, 1, k, dtype=torch.float32), dim=-1)
    state.step(probs)

    logits = torch.randn(1, h, 1, k, dtype=torch.float32)
    masked = state.apply_mask_to_attn_weights(logits)
    assert masked.shape == logits.shape

    add = state.logits_mask_additive(k, dtype=torch.float32, device=probs.device)
    assert add.shape == (1, h, 1, k)


def test_lru_keeps_recently_winning_keys():
    h = 1
    state = LRUKVCachePolicy(cache_budget=2)

    probs1 = torch.zeros(1, h, 1, 4, dtype=torch.float32)
    probs1[0, 0, 0, 0] = 1.0
    state.step(probs1)

    probs2 = torch.zeros(1, h, 1, 5, dtype=torch.float32)
    probs2[0, 0, 0, 3] = 1.0
    state.step(probs2)

    probs3 = torch.zeros(1, h, 1, 6, dtype=torch.float32)
    probs3[0, 0, 0, 1] = 1.0
    out = state.step(probs3)

    mask = out["attention_masks_next"]
    assert mask is not None
    assert mask.shape == (1, h, 1, 6)

    # Most recently accessed: position 1 (step 3), position 5 is new key with step 3.
    # Position 3 was accessed in step 2, so it's evicted. Cache keeps [1, 5].
    kept = torch.nonzero(mask[0, 0, 0] > 0.5, as_tuple=False).flatten().tolist()
    assert sorted(kept) == [1, 5]


def test_lru_reset_clears_state():
    state = LRUKVCachePolicy(cache_budget=2)
    probs = torch.softmax(torch.randn(1, 1, 1, 3, dtype=torch.float32), dim=-1)
    state.step(probs)

    state.reset()
    assert state.attention_masks_next is None
    assert state.previous_scores is None
    assert state.last_used_step is None
    assert state.step_index == 0
