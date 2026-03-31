#!/usr/bin/env python3
"""
Benchmark KV cache policies on long-context tasks.

Tests how efficiently each policy handles tasks with long context passages that require
generating answers to questions. This stresses the KV cache since context is processed
once (prefill) and then answer generation (decode) must access relevant tokens.

Usage:
    ./venv/bin/python scripts/long_context_benchmark.py --model gpt2 --max-answer-tokens 50 --cache-budget 64
"""

from __future__ import annotations

import argparse
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
from benchmark_utils import make_policy, policy_next_attention_mask, summary_stats, torch_dtype_from_name


def load_long_context_dataset(filepath: Path) -> list[dict[str, str]]:
    """Load long-context dataset in CONTEXT/QUESTION/--- format."""
    tasks = []
    current_context = None
    current_question = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("CONTEXT:"):
                current_context = line[len("CONTEXT:") :].strip()
            elif line.startswith("QUESTION:"):
                current_question = line[len("QUESTION:") :].strip()
            elif line.startswith("---"):
                if current_context and current_question:
                    tasks.append({"context": current_context, "question": current_question})
                    current_context = None
                    current_question = None

    if current_context and current_question:
        tasks.append({"context": current_context, "question": current_question})

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KV cache policies on long-context tasks")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="data/long_context.txt", help="Path to dataset file")
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to eval (None = all)")
    parser.add_argument("--max-answer-tokens", type=int, default=50, help="Max tokens to generate per answer")
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


def answer_question(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context: str,
    question: str,
    max_answer_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    policy: Optional[KVCachePolicy],
    mask_reduction: str,
) -> tuple[float, str]:
    """
    Answer a question given context using cached decode with optional policy masking.
    Returns (elapsed_time, generated_text).
    """
    # Format input: context + question
    prompt = f"{context} {question}"
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    prompt_attention_mask = tok.attention_mask.to(device)
    next_attention_mask: Optional[torch.Tensor] = None
    generated_ids = input_ids.clone()

    start = time.perf_counter()

    with torch.no_grad():
        # Prefill: process entire prompt
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

        # Decode: generate answer tokens
        next_id = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)

        for _ in range(max_answer_tokens - 1):
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

            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

    elapsed = time.perf_counter() - start
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return elapsed, generated_text


def evaluate_answer_quality(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context: str,
    question: str,
    answer: str,
    device: torch.device,
) -> float:
    """
    Evaluate answer quality by asking the model to rate it.
    Prompt the model: "Question: [Q] | Answer: [A] | Does this answer correctly address the question? Yes or No?"
    Returns the probability of "Yes" (0-1 scale).
    """
    # Create validation prompt
    validation_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Does this answer correctly address the question? "
    )

    input_ids = tokenizer(validation_prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[0, -1, :]

        # Get token IDs for "Yes" and "No"
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

        # Compute softmax over yes/no tokens
        yes_logit = logits[yes_id].item()
        no_logit = logits[no_id].item()

        # Probability of "Yes"
        yes_prob = torch.sigmoid(torch.tensor(yes_logit - no_logit)).item()

    return yes_prob


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

    print(f"Loading dataset from {args.dataset}")
    dataset_path = _REPO_ROOT / args.dataset
    if not dataset_path.exists():
        print(f"Error: dataset file not found at {dataset_path}")
        sys.exit(1)

    tasks = load_long_context_dataset(dataset_path)
    if args.num_tasks is not None:
        tasks = tasks[: args.num_tasks]

    print(f"Loaded {len(tasks)} tasks\n")

    policy_names = ["no_cache", "cached_baseline", "h2o", "fifo", "lru"]
    results: dict[str, list[tuple[float, float]]] = {name: [] for name in policy_names}

    print(f"Benchmarking on {device} with {len(tasks)} tasks")
    print(f"Max answer tokens: {args.max_answer_tokens}\n")

    for task_idx, task in enumerate(tasks):
        context = task["context"]
        question = task["question"]

        print(f"Task {task_idx + 1}/{len(tasks)}...", end=" ", flush=True)
        task_results = {}

        for policy_name in policy_names:
            policy = make_policy(policy_name, args) if policy_name != "no_cache" else None

            if policy_name == "no_cache":
                # For no_cache, use a simpler approach without caching
                prompt = f"{context} {question}"
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                generated_ids = input_ids.clone()

                start = time.perf_counter()
                with torch.no_grad():
                    next_id = torch.argmax(model(input_ids, use_cache=False).logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_id], dim=1)

                    for _ in range(args.max_answer_tokens - 1):
                        out = model(input_ids=generated_ids, use_cache=False)
                        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_id], dim=1)

                elapsed = time.perf_counter() - start
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                elapsed, generated_text = answer_question(
                    model,
                    tokenizer,
                    context,
                    question,
                    args.max_answer_tokens,
                    device,
                    dtype,
                    policy,
                    args.mask_reduction,
                )

            answer_quality = evaluate_answer_quality(model, tokenizer, context, question, generated_text, device)
            task_results[policy_name] = (elapsed, answer_quality)
            results[policy_name].append((elapsed, answer_quality))

        print(" | ".join(f"{name}={task_results[name][0]:.2f}s,q={task_results[name][1]:.2f}" for name in policy_names))

    print("\n" + "=" * 100)
    print("LONG-CONTEXT BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Policy':<18} {'Time (mean)':<14} {'Time (median)':<14} {'Quality (mean)':<14} {'Quality (median)':<14}")
    print("-" * 100)

    for policy_name in policy_names:
        times = [t for t, q in results[policy_name]]
        qualities = [q for t, q in results[policy_name]]

        time_mean = sum(times) / len(times) if times else 0.0
        time_median = sorted(times)[len(times) // 2] if times else 0.0
        quality_mean = sum(qualities) / len(qualities) if qualities else 0.0
        quality_median = sorted(qualities)[len(qualities) // 2] if qualities else 0.0

        print(
            f"{policy_name:<18} {time_mean:<14.4f} {time_median:<14.4f} {quality_mean:<14.4f} {quality_median:<14.4f}"
        )

    print("=" * 100)
    print("\nConfig:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Max Answer Tokens: {args.max_answer_tokens}")
    print(f"  Cache Budget (FIFO/LRU): {args.cache_budget}")
    print(f"  Mask Reduction: {args.mask_reduction}")
    print(f"  H2O Ratios (heavy/recent): {args.h2o_heavy_ratio}/{args.h2o_recent_ratio}")


if __name__ == "__main__":
    main()
