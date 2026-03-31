#!/usr/bin/env python3
"""
Benchmark perplexity for KV cache policies on GPT2.

Compares:
- no_cache: teacher-forced decode with use_cache=False
- cached_baseline: teacher-forced cached decode with no policy
- h2o / fifo / lru: teacher-forced cached decode with policy-driven attention masking

Usage:
    ./venv/bin/python scripts/perplexity_benchmark.py --model gpt2 --max-new-tokens 150 --teacher-prefill-tokens 64 --cache-budget 128 --mask-reduction any
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from attention_kv_h2o.kv_cache_base import KVCachePolicy
from benchmark_utils import make_policy, policy_next_attention_mask, summary_stats, torch_dtype_from_name


BENCHMARK_TEXT = (
    "The Time Traveller (for so it will be convenient to speak of him) was "
    "expounding a recondite matter to us. His pale grey eyes shone and "
    "twinkled, and his usually pale face was flushed and animated. The fire "
    "burned brightly, and the soft radiance of the incandescent lights in the "
    "lilies of silver caught the bubbles that flashed at our glasses. Our "
    "chairs, being his furniture, hugged us rather than held us, and there was "
    "that luxurious after-dinner atmosphere when thought runs gracefully free "
    "of the trammels of precision. And he put it to us in this way--marking the "
    "points with a lean forefinger--as we sat and lazily admired his earnestness "
    "over this new paradox (as we thought it) and his fecundity in inventing "
    "subtleties. 'You must follow me carefully. I shall have to controvert one "
    "or two ideas that are almost universally accepted. The geometry, for "
    "instance, they taught you at school is founded on a misconception.' 'Is "
    "not that rather a large thing to expect us to begin upon?' said Filby, an "
    "argumentative person with red hair. 'I do not mean to ask you to accept "
    "anything without reasonable ground for it. You will soon admit as much as "
    "I need from you. You know of course that a mathematical line, a line of "
    "thickness nil, has no real existence. They taught you that?'"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark perplexity for KV cache eviction policies")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Tokens to generate/evaluate")
    parser.add_argument(
        "--teacher-prefill-tokens",
        type=int,
        default=32,
        help="Teacher-forced mode: number of initial ground-truth tokens used to initialize cache/policy",
    )
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
    return parser.parse_args()


def _teacher_target_nll(logits: torch.Tensor, target_next_id: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_prob = log_probs.gather(-1, target_next_id).squeeze(-1)
    return float((-token_log_prob).item())


def _prepare_teacher_sequence(
    tokenizer: AutoTokenizer,
    text: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    eval_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    available = int(eval_ids.shape[1] - 1)
    if available < 1:
        raise ValueError("Teacher-forced mode needs at least 2 tokens in eval text")
    eval_steps = min(max_new_tokens, available)
    return eval_ids, eval_steps




def perplexity_no_cache_teacher_forced(
    model: torch.nn.Module,
    eval_ids: torch.Tensor,
    eval_steps: int,
) -> float:
    """Teacher-forced perplexity with use_cache=False on fixed reference tokens."""
    current_ids = eval_ids[:, :1]
    nll_sum = 0.0

    with torch.no_grad():
        for t in range(1, eval_steps + 1):
            outputs = model(
                input_ids=current_ids,
                use_cache=False,
                output_attentions=False,
            )
            target_next = eval_ids[:, t : t + 1]
            nll_sum += _teacher_target_nll(outputs.logits[:, -1, :], target_next)
            current_ids = torch.cat([current_ids, target_next], dim=1)

    return math.exp(nll_sum / eval_steps)


def perplexity_cached_teacher_forced(
    model: torch.nn.Module,
    eval_ids: torch.Tensor,
    eval_steps: int,
    dtype: torch.dtype,
    policy: KVCachePolicy | None,
    mask_reduction: str,
    teacher_prefill_tokens: int,
) -> float:
    """Teacher-forced perplexity with cached decode and optional policy masking."""
    prefill_len = max(1, min(int(teacher_prefill_tokens), eval_steps))
    input_ids = eval_ids[:, :prefill_len]
    prompt_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    next_attention_mask: torch.Tensor | None = None
    nll_sum = 0.0
    counted = 0

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=prompt_attention_mask,
            output_attentions=policy is not None,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        if policy is not None:
            attn = out.attentions[-1].to(dtype=dtype)
            next_attention_mask = policy_next_attention_mask(policy, attn, mask_reduction)

        # Score targets inside prefill window: tokens 1..prefill_len-1.
        if prefill_len > 1:
            prefill_logits = out.logits[:, : prefill_len - 1, :]
            prefill_targets = eval_ids[:, 1:prefill_len]
            log_probs = torch.log_softmax(prefill_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, prefill_targets.unsqueeze(-1)).squeeze(-1)
            nll_sum += float((-token_log_probs).sum().item())
            counted += prefill_len - 1

        # Score remaining targets from prefill_len..eval_steps.
        for t in range(prefill_len, eval_steps + 1):
            target_next = eval_ids[:, t : t + 1]
            nll_sum += _teacher_target_nll(out.logits[:, -1, :], target_next)
            counted += 1

            out = model(
                input_ids=target_next,
                past_key_values=past_key_values,
                attention_mask=next_attention_mask,
                output_attentions=policy is not None,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            if policy is not None:
                attn = out.attentions[-1].to(dtype=dtype)
                next_attention_mask = policy_next_attention_mask(policy, attn, mask_reduction)

    if counted <= 0:
        raise RuntimeError("No target tokens were scored in teacher-forced mode")
    return math.exp(nll_sum / counted)


def _evaluate_policy_once(
    name: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    dtype: torch.dtype,
    teacher_eval_ids: torch.Tensor,
    teacher_eval_steps: int,
) -> float:
    if name == "no_cache":
        return perplexity_no_cache_teacher_forced(model, teacher_eval_ids, teacher_eval_steps)
    return perplexity_cached_teacher_forced(
        model,
        teacher_eval_ids,
        teacher_eval_steps,
        dtype,
        policy=make_policy(name, args),
        mask_reduction=args.mask_reduction,
        teacher_prefill_tokens=args.teacher_prefill_tokens,
    )


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

    teacher_eval_ids, teacher_eval_steps = _prepare_teacher_sequence(
        tokenizer,
        BENCHMARK_TEXT,
        args.max_new_tokens,
        device,
    )

    print(f"\nBenchmarking perplexity on {device} (single deterministic run)")
    print(f"Teacher-forced eval on fixed targets for {teacher_eval_steps} token(s) per run\n")

    policy_order = ["no_cache", "cached_baseline", "h2o", "fifo", "lru"]
    results: dict[str, list[float]] = {name: [] for name in policy_order}
    run_values: dict[str, float] = {}

    print("Run 1/1...", end=" ", flush=True)
    for name in policy_order:
        ppl = _evaluate_policy_once(
            name,
            args,
            model,
            dtype,
            teacher_eval_ids,
            teacher_eval_steps,
        )
        run_values[name] = ppl
        results[name].append(ppl)

    print(" ".join(f"{name}={run_values[name]:.4f}" for name in policy_order))

    print("\n" + "=" * 96)
    print("PERPLEXITY BENCHMARK RESULTS (lower is better)")
    print("=" * 96)
    print(f"{'Policy':<18} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'CV%':<10} {'Min':<12} {'Max':<12}")
    print("-" * 96)

    for policy_name, values in results.items():
        stats = summary_stats(values)
        print(
            f"{policy_name:<18} {stats['mean']:<12.4f} {stats['median']:<12.4f} {stats['std']:<12.4f} "
            f"{stats['cv_pct']:<10.2f} {stats['min']:<12.4f} {stats['max']:<12.4f}"
        )

    print("=" * 96)
    print("\nConfig:")
    print(f"  Model: {args.model}")
    print("  Eval Mode: teacher-forced")
    print("  Eval Text Source: BENCHMARK_TEXT constant")
    print(f"  Evaluated Tokens/Run: {teacher_eval_steps}")
    print(f"  Device: {device}")
    print(f"  Cache Budget (FIFO/LRU): {args.cache_budget}")
    print(f"  Teacher Prefill Tokens: {args.teacher_prefill_tokens}")
    print(f"  Mask Reduction: {args.mask_reduction}")
    print(f"  H2O Ratios (heavy/recent): {args.h2o_heavy_ratio}/{args.h2o_recent_ratio}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print("  Runs: 1")


if __name__ == "__main__":
    main()
