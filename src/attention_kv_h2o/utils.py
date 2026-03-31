from __future__ import annotations

import torch


def scores_from_attention_probs(attn_weights: torch.Tensor) -> torch.Tensor:
    """Reduce softmax attention to per-head key scores for one forward pass."""
    if attn_weights.dim() != 4:
        raise ValueError(
            f"attn_weights must be 4D (B, H, Q, K), got shape {tuple(attn_weights.shape)}"
        )
    return attn_weights.sum(dim=0).sum(dim=1)
