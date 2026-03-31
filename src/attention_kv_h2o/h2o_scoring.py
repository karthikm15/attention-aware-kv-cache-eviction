"""
Heavy-Hitter Oracle (H2O) scoring isolated from model code.

Matches the accumulation and eviction-mask logic from FMInference/H2O
`h2o_hf/utils_hh/modify_llama.py`: per-head scores are the sum of attention
probability mass landing on each key position (summed over batch and query
positions). Scores accumulate across decode steps with the last key column
handled as in the reference (no carry into the brand-new key slot). When the
sequence exceeds the cache budget, each head keeps the most recent keys plus
the top heavy-budget keys by accumulated score among the non-recent prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from attention_kv_h2o.kv_cache_base import KVCachePolicy
from attention_kv_h2o.utils import scores_from_attention_probs


@dataclass
class H2OOracleState(KVCachePolicy):
    """
    Stateful H2O scorer + next-step attention mask (per head, per key).

    heavy_budget_ratio and recent_budget_ratio are fractions of the current
    key length used to set heavy_budget and recent_budget on the first step,
    matching int(ratio * seq_len) in the reference implementation.
    """

    heavy_budget_ratio: float
    recent_budget_ratio: float
    previous_scores: Optional[torch.Tensor] = field(default=None, repr=False)
    heavy_budget: Optional[int] = field(default=None, repr=False)
    recent_budget: Optional[int] = field(default=None, repr=False)
    cache_budget: Optional[int] = field(default=None, repr=False)
    cache_budget_records: list[int] = field(default_factory=list, repr=False)
    input_lengths: list[int] = field(default_factory=list, repr=False)
    attention_masks_next: Optional[torch.Tensor] = field(default=None, repr=False)

    def reset(self) -> None:
        self.previous_scores = None
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.cache_budget_records.clear()
        self.input_lengths.clear()
        self.attention_masks_next = None

    def step(
        self,
        attn_weights: torch.Tensor,
        *,
        assert_single_batch: bool = True,
    ) -> dict[str, torch.Tensor | int | None]:
        """
        One decoding (or prefill) step: update accumulated scores and build the
        mask applied to attention on the *next* forward pass.

        attn_weights: post-softmax probabilities, shape (B, H, Q, K).

        Returns a dict with:
          - key_len: int
          - current_scores_sum: (H, K) tensor before masking stored scores
          - previous_scores_out: (H, K) tensor stored for next step (masked)
          - attention_masks_next: (1, H, 1, K) float {0,1} for multiplying logits
          - heavy_budget, recent_budget, cache_budget: ints or None before init
        """
        if attn_weights.dim() != 4:
            raise ValueError(
                f"attn_weights must be 4D (B, H, Q, K), got {tuple(attn_weights.shape)}"
            )
        bsz, num_heads, q_len, kv_len = attn_weights.shape
        if assert_single_batch and bsz != 1:
            raise ValueError(
                f"H2O reference path assumes batch size 1 for mask indexing; got {bsz}"
            )

        current_scores_sum = scores_from_attention_probs(attn_weights)

        if self.previous_scores is not None:
            expected_prev_k = kv_len - 1
            if self.previous_scores.shape[-1] != expected_prev_k:
                raise ValueError(
                    "previous_scores last dim must be key_len - 1 when continuing a "
                    f"sequence; got prev_k={self.previous_scores.shape[-1]} "
                    f"and key_len={kv_len}"
                )
            if self.previous_scores.shape[0] != num_heads:
                raise ValueError(
                    f"previous_scores num_heads {self.previous_scores.shape[0]} != "
                    f"{num_heads}"
                )
            current_scores_sum = current_scores_sum.clone()
            current_scores_sum[:, :-1] = current_scores_sum[:, :-1] + self.previous_scores
        else:
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = int(self.heavy_budget) + int(self.recent_budget)
            self.cache_budget_records.append(int(self.cache_budget))
            self.input_lengths.append(int(kv_len))

        dtype, device = self._dtype_device(attn_weights)
        self.previous_scores = current_scores_sum.clone()

        attn_tokens_all = self.previous_scores.shape[-1]
        h = self.previous_scores.shape[0]

        if self.cache_budget is None:
            raise RuntimeError("cache_budget unset; first step must run with previous_scores=None")

        if attn_tokens_all > int(self.cache_budget):
            attn_mask = torch.ones(h, attn_tokens_all + 1, dtype=dtype, device=device)
            if self.recent_budget != 0:
                attn_mask[:, : -self.recent_budget] = 0
                selected_set = self.previous_scores[:, : -self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if self.heavy_budget != 0 and selected_set.shape[-1] > 0:
                effective_k = min(self.heavy_budget, selected_set.shape[-1])
                _, keep_topk = selected_set.topk(k=effective_k, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1.0)

            score_mask = attn_mask[:, :-1].clone()
            if self.recent_budget != 0:
                score_mask[:, -self.recent_budget :] = 1.0
            self.previous_scores = self.previous_scores * score_mask

            # Reference builds attn_mask with shape (H, K+1); attention logits are (B, H, Q, K).
            # Use the first K columns so the mask matches KV positions used in matmul.
            next_mask = attn_mask[:, :-1].clone().unsqueeze(0).unsqueeze(2)
            self.attention_masks_next = next_mask
        else:
            self.attention_masks_next = torch.ones(1, h, 1, attn_tokens_all, dtype=dtype, device=device)

        return {
            "key_len": attn_tokens_all,
            "num_heads": num_heads,
            "query_len": q_len,
            "current_scores_sum": current_scores_sum,
            "previous_scores_out": self.previous_scores.clone(),
            "attention_masks_next": self.attention_masks_next.clone(),
            "heavy_budget": self.heavy_budget,
            "recent_budget": self.recent_budget,
            "cache_budget": self.cache_budget,
        }


def simulate_eviction_trace(
    attention_probs_sequence: list[torch.Tensor],
    heavy_budget_ratio: float,
    recent_budget_ratio: float,
) -> list[dict[str, torch.Tensor | int | None]]:
    """
    Offline replay: run H2O steps over a list of attention probability tensors
    (one per forward pass), e.g. from recorded generation.
    """
    state = H2OOracleState(
        heavy_budget_ratio=heavy_budget_ratio,
        recent_budget_ratio=recent_budget_ratio,
    )
    trace: list[dict[str, torch.Tensor | int | None]] = []
    for probs in attention_probs_sequence:
        trace.append(state.step(probs))
    return trace
