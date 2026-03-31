from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from attention_kv_h2o.kv_cache_base import KVCachePolicy
from attention_kv_h2o.utils import scores_from_attention_probs


@dataclass
class LRUKVCachePolicy(KVCachePolicy):
    """Keep keys with the most recent attention winner hits per head."""

    cache_budget: int
    previous_scores: Optional[torch.Tensor] = field(default=None, repr=False)
    attention_masks_next: Optional[torch.Tensor] = field(default=None, repr=False)
    last_used_step: Optional[torch.Tensor] = field(default=None, repr=False)
    step_index: int = field(default=0, repr=False)

    def reset(self) -> None:
        self.previous_scores = None
        self.attention_masks_next = None
        self.last_used_step = None
        self.step_index = 0

    def _ensure_last_used_shape(self, num_heads: int, kv_len: int, device: torch.device) -> None:
        if self.last_used_step is None:
            self.last_used_step = torch.full((num_heads, kv_len), -1, dtype=torch.long, device=device)
            return

        old_k = self.last_used_step.shape[-1]
        if kv_len < old_k:
            raise ValueError(f"key_len must be monotonic non-decreasing, got old_k={old_k}, kv_len={kv_len}")
        if kv_len > old_k:
            pad = torch.full((num_heads, kv_len - old_k), self.step_index, dtype=torch.long, device=device)
            self.last_used_step = torch.cat([self.last_used_step, pad], dim=-1)

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
            raise ValueError(f"LRU path assumes batch size 1 for mask indexing; got {bsz}")

        if self.cache_budget < 0:
            raise ValueError(f"cache_budget must be >= 0, got {self.cache_budget}")

        dtype, device = self._dtype_device(attn_weights)
        current_scores_sum = scores_from_attention_probs(attn_weights)
        self.previous_scores = current_scores_sum.clone()

        self._ensure_last_used_shape(num_heads, kv_len, device)

        winners = attn_weights.argmax(dim=-1)
        for head_idx in range(num_heads):
            unique_positions = winners[:, head_idx, :].reshape(-1).unique()
            self.last_used_step[head_idx, unique_positions] = self.step_index

        keep = torch.zeros(num_heads, kv_len, dtype=dtype, device=device)
        if self.cache_budget >= kv_len:
            keep[:, :] = 1.0
        elif self.cache_budget > 0:
            _, keep_idx = self.last_used_step.topk(k=self.cache_budget, dim=-1, largest=True)
            keep = keep.scatter(-1, keep_idx, 1.0)

        self.previous_scores = self.previous_scores * keep
        self.attention_masks_next = keep.unsqueeze(0).unsqueeze(2)

        out = {
            "policy": "lru",
            "key_len": kv_len,
            "num_heads": num_heads,
            "query_len": q_len,
            "current_scores_sum": current_scores_sum,
            "previous_scores_out": self.previous_scores.clone(),
            "attention_masks_next": self.attention_masks_next.clone(),
            "cache_budget": self.cache_budget,
        }
        self.step_index += 1
        return out
