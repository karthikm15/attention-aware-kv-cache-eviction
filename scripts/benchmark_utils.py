from __future__ import annotations

import argparse
import statistics

import torch

from attention_kv_h2o import FIFOKVCachePolicy, H2OOracleState, LRUKVCachePolicy
from attention_kv_h2o.kv_cache_base import KVCachePolicy


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def make_policy(name: str, args: argparse.Namespace) -> KVCachePolicy | None:
    if name == "cached_baseline":
        return None
    if name == "h2o":
        return H2OOracleState(
            heavy_budget_ratio=args.h2o_heavy_ratio,
            recent_budget_ratio=args.h2o_recent_ratio,
        )
    if name == "fifo":
        return FIFOKVCachePolicy(cache_budget=args.cache_budget)
    if name == "lru":
        return LRUKVCachePolicy(cache_budget=args.cache_budget)
    raise ValueError(f"Unknown policy: {name}")


def next_token_greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)


def collapse_policy_mask(next_mask: torch.Tensor, reduction: str) -> torch.Tensor:
    """Collapse per-head keep mask (1, H, 1, K) to global keep mask (K,)."""
    if next_mask.dim() != 4:
        raise ValueError(f"Expected next_mask 4D, got shape {tuple(next_mask.shape)}")
    if reduction == "all":
        keep_2d = (next_mask > 0.5).all(dim=1).squeeze(1)
    else:
        keep_2d = (next_mask > 0.5).any(dim=1).squeeze(1)
    if keep_2d.shape[0] != 1:
        raise ValueError(f"Expected batch=1 for keep mask, got shape {tuple(keep_2d.shape)}")
    keep_1d = keep_2d[0]
    if keep_1d.numel() > 0:
        # Always retain newest key to keep decode stable.
        keep_1d[-1] = True
    return keep_1d


def policy_next_attention_mask(
    policy: KVCachePolicy,
    attn_weights: torch.Tensor,
    mask_reduction: str,
) -> torch.Tensor:
    """Build next-step (B, K+1) attention mask from policy output."""
    info = policy.step(attn_weights, assert_single_batch=True)
    next_mask = info.get("attention_masks_next")
    if not isinstance(next_mask, torch.Tensor):
        raise RuntimeError("Policy did not return attention_masks_next tensor")
    keep_mask_1d = collapse_policy_mask(next_mask, reduction=mask_reduction)
    keep_2d = keep_mask_1d.unsqueeze(0).to(dtype=torch.long)
    # Next decode step adds one fresh token, always keep it.
    return torch.cat(
        [keep_2d, torch.ones((keep_2d.shape[0], 1), dtype=keep_2d.dtype, device=keep_2d.device)],
        dim=1,
    )


def summary_stats(values: list[float]) -> dict[str, float]:
    mean_v = statistics.mean(values)
    median_v = statistics.median(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    cv_pct = (100.0 * std_v / mean_v) if mean_v > 0 else 0.0
    return {
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "cv_pct": cv_pct,
        "min": min(values),
        "max": max(values),
    }
