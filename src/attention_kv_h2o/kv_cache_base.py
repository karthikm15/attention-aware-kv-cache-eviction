from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class KVCachePolicy(ABC):
    """Abstract interface for KV cache eviction policies."""

    attention_masks_next: Optional[torch.Tensor] = None

    @abstractmethod
    def reset(self) -> None:
        """Reset all policy state."""

    @abstractmethod
    def step(
        self,
        attn_weights: torch.Tensor,
        *,
        assert_single_batch: bool = True,
    ) -> dict[str, torch.Tensor | int | None]:
        """Consume one attention tensor and update next-step keep mask."""

    def _dtype_device(self, attn_weights: torch.Tensor) -> tuple[torch.dtype, torch.device]:
        return attn_weights.dtype, attn_weights.device

    def apply_mask_to_attn_weights(
        self,
        attn_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multiplicative keep mask to logits/scores as additive min-value drop mask."""
        if mask is None:
            mask = self.attention_masks_next
        if mask is None:
            return attn_weights
        if mask.shape[-1] != attn_weights.shape[-1]:
            raise ValueError(
                f"mask K ({mask.shape[-1]}) must match attn_weights K ({attn_weights.shape[-1]})"
            )
        min_val = torch.finfo(attn_weights.dtype).min
        return attn_weights * mask + (1.0 - mask) * min_val

    def logits_mask_additive(
        self,
        key_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive mask: 0 for keep, dtype min value for drop."""
        if self.attention_masks_next is None:
            return torch.zeros(1, 1, 1, key_len, dtype=dtype, device=device)
        h = self.attention_masks_next.shape[1]
        if self.attention_masks_next.shape[-1] != key_len:
            raise ValueError(
                f"attention_masks_next K ({self.attention_masks_next.shape[-1]}) != key_len ({key_len})"
            )
        min_val = torch.finfo(dtype).min
        m = self.attention_masks_next.to(dtype=dtype, device=device)
        return (1.0 - m) * min_val
