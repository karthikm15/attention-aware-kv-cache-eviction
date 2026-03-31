#!/usr/bin/env python3
"""
Benchmark generation throughput (tokens/second) for KV cache policies on GPT2.

Compares:
- no_cache: manual decode with use_cache=False
- cached_baseline: manual decode with use_cache=True and no eviction
- h2o / fifo / lru: cached decode with policy-driven KV pruning

Usage:
  python scripts/throughput_benchmark.py --model gpt2 --max-new-tokens 100 \
    --num-runs 10 --cache-budget 128
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from attention_kv_h2o.kv_cache_base import KVCachePolicy
from benchmark_utils import (
    make_policy,
    next_token_greedy,
    policy_next_attention_mask,
    summary_stats,
    torch_dtype_from_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KV cache eviction policies")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmup runs before benchmarking")
    parser.add_argument("--warmup-tokens", type=int, default=16, help="Tokens to generate in each warmup run")
    parser.add_argument("--cache-budget", type=int, default=128, help="Cache budget for FIFO/LRU")
    parser.add_argument("--h2o-heavy-ratio", type=float, default=0.1, help="Heavy hitter ratio for H2O")
    parser.add_argument("--h2o-recent-ratio", type=float, default=0.1, help="Recent ratio for H2O")
    parser.add_argument(
        "--mask-reduction",
        type=str,
        default="any",
        choices=("any", "all"),
        help="How to collapse per-head policy masks to a global KV keep set",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument(
        "--shuffle-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle policy evaluation order each run to reduce order bias",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffled order")
    return parser.parse_args()


def benchmark_no_cache_manual(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    """Manual decode with use_cache=False."""
    current_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_ids,
                use_cache=False,
                output_attentions=False,
            )
            next_id = next_token_greedy(outputs.logits[:, -1, :])
            current_ids = torch.cat([current_ids, next_id], dim=1)

    return max_new_tokens / (time.perf_counter() - start)


def benchmark_cached_manual(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    policy: Optional[KVCachePolicy],
    mask_reduction: str,
) -> float:
    """Manual cached decode with optional policy-driven attention masking."""
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    prompt_attention_mask = tok.attention_mask.to(device)
    next_attention_mask: Optional[torch.Tensor] = None

    start = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=prompt_attention_mask,
            output_attentions=policy is not None,
            use_cache=True,
        )
        past_key_values = prefill_out.past_key_values
        if policy is not None:
            prefill_attn = prefill_out.attentions[-1].to(dtype=dtype)
            next_attention_mask = policy_next_attention_mask(policy, prefill_attn, mask_reduction)

        next_id = next_token_greedy(prefill_out.logits[:, -1, :])

        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_id,
                past_key_values=past_key_values,
                attention_mask=next_attention_mask,
                output_attentions=policy is not None,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if policy is not None:
                decode_attn = out.attentions[-1].to(dtype=dtype)
                next_attention_mask = policy_next_attention_mask(policy, decode_attn, mask_reduction)
            next_id = next_token_greedy(out.logits[:, -1, :])

    return max_new_tokens / (time.perf_counter() - start)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    print(f"\nBenchmarking on {device} with {args.num_runs} runs each")
    print(f"Generating {args.max_new_tokens} tokens per run\n")

    policy_order = ["no_cache", "cached_baseline", "h2o", "fifo", "lru"]

    def eval_policy(name: str, max_new_tokens: int) -> float:
        if name == "no_cache":
            return benchmark_no_cache_manual(model, tokenizer, args.prompt, max_new_tokens, device)
        return benchmark_cached_manual(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens,
            device,
            dtype,
            policy=make_policy(name, args),
            mask_reduction=args.mask_reduction,
        )

    if args.warmup_runs > 0:
        print(
            f"Running {args.warmup_runs} warmup run(s) with {args.warmup_tokens} token(s) each (untimed)..."
        )
        for warm_i in range(args.warmup_runs):
            for name in policy_order:
                _ = eval_policy(name, max_new_tokens=args.warmup_tokens)
            print(f"Warmup {warm_i + 1}/{args.warmup_runs} complete")
        print()

    results: dict[str, list[float]] = {name: [] for name in policy_order}
    rng = random.Random(args.seed)

    for run_idx in range(args.num_runs):
        run_values: dict[str, float] = {}
        eval_order = list(policy_order)
        if args.shuffle_order:
            rng.shuffle(eval_order)

        print(f"Run {run_idx + 1}/{args.num_runs}...", end=" ", flush=True)
        for name in eval_order:
            tps = eval_policy(name, max_new_tokens=args.max_new_tokens)
            run_values[name] = tps
            results[name].append(tps)

        print(" ".join(f"{name}={run_values[name]:.2f}" for name in policy_order))

    print("\n" + "=" * 96)
    print("THROUGHPUT BENCHMARK RESULTS (tokens/second)")
    print("=" * 96)
    print(f"{'Policy':<18} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'CV%':<10} {'Min':<12} {'Max':<12}")
    print("-" * 96)

    for policy_name, values in results.items():
        stats = summary_stats(values)
        print(
            f"{policy_name:<18} {stats['mean']:<12.2f} {stats['median']:<12.2f} {stats['std']:<12.2f} "
            f"{stats['cv_pct']:<10.2f} {stats['min']:<12.2f} {stats['max']:<12.2f}"
        )

    print("=" * 96)
    print("\nConfig:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Cache Budget (FIFO/LRU): {args.cache_budget}")
    print(f"  Mask Reduction: {args.mask_reduction}")
    print(f"  H2O Ratios (heavy/recent): {args.h2o_heavy_ratio}/{args.h2o_recent_ratio}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Warmup Runs/Tokens: {args.warmup_runs}/{args.warmup_tokens}")
    print(f"  Shuffle Order: {args.shuffle_order} (seed={args.seed})")
    print(f"  Runs: {args.num_runs}")


if __name__ == "__main__":
    main()
