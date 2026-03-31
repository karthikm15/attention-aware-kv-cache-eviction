from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from attention_kv_h2o.kv_cache_base import KVCachePolicy
from attention_kv_h2o.utils import scores_from_attention_probs


@dataclass
class FIFOKVCachePolicy(KVCachePolicy):
    """Keep only the newest keys when key length exceeds a fixed cache budget."""

    cache_budget: int
    previous_scores: Optional[torch.Tensor] = field(default=None, repr=False)
    attention_masks_next: Optional[torch.Tensor] = field(default=None, repr=False)

    def reset(self) -> None:
        self.previous_scores = None
        self.attention_masks_next = None

    def step(
        self,
        attn_weights: torch.Tensor,
        *,
        assert_single_batch: bool = True,
    ) -> dict[str, torch.Tensor | int | None]:
        if attn_weights.dim() != 4:
            raise ValueError(
                f"attn_weights must be 4D (B, H, Q, K), got {tuple(attn_weights.shape)}"
            )

        bsz, num_heads, q_len, kv_len = attn_weights.shape
        if assert_single_batch and bsz != 1:
            raise ValueError(f"FIFO path assumes batch size 1 for mask indexing; got {bsz}")

        if self.cache_budget < 0:
            raise ValueError(f"cache_budget must be >= 0, got {self.cache_budget}")

        current_scores_sum = scores_from_attention_probs(attn_weights)
        self.previous_scores = current_scores_sum.clone()

        dtype, device = self._dtype_device(attn_weights)
        keep = torch.zeros(num_heads, kv_len, dtype=dtype, device=device)

        if self.cache_budget >= kv_len:
            keep[:, :] = 1.0
        elif self.cache_budget > 0:
            keep[:, kv_len - self.cache_budget :] = 1.0

        self.previous_scores = self.previous_scores * keep
        self.attention_masks_next = keep.unsqueeze(0).unsqueeze(2)

        return {
            "policy": "fifo",
            "key_len": kv_len,
            "num_heads": num_heads,
            "query_len": q_len,
            "current_scores_sum": current_scores_sum,
            "previous_scores_out": self.previous_scores.clone(),
            "attention_masks_next": self.attention_masks_next.clone(),
            "cache_budget": self.cache_budget,
        }
