#!/usr/bin/env python3
"""
Phase-1 demo: run a small Hugging Face causal LM with output_attentions=True,
feed last-layer softmax attention into H2OOracleState each prefill/decode step.

Install (from repo root): pip install -e .

Example:
  python scripts/run_hf_h2o_demo.py --model gpt2 --max-new-tokens 16 \\
    --heavy-ratio 0.2 --recent-ratio 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from attention_kv_h2o.h2o_scoring import H2OOracleState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H2O scoring demo with Hugging Face GPT-style LM")
    parser.add_argument("--model", type=str, default="gpt2", help="Causal LM name on the Hub")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt text for generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--heavy-ratio", type=float, default=0.2)
    parser.add_argument("--recent-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "bfloat16", "float16"))
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    past_key_values = None
    h2o = H2OOracleState(
        heavy_budget_ratio=args.heavy_ratio,
        recent_budget_ratio=args.recent_ratio,
    )

    prompt_len = int(input_ids.size(1))
    next_id: torch.Tensor | None = None

    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            past_key_values=None,
            output_attentions=True,
            use_cache=True,
            attention_mask=None,
        )
        past_key_values = prefill_out.past_key_values
        prefill_attn = prefill_out.attentions[-1].to(dtype=dtype)
        prefill_info = h2o.step(prefill_attn, assert_single_batch=True)
        pm = prefill_info["attention_masks_next"]
        print(
            f"[prefill] tokens={prompt_len} key_len={prefill_info['key_len']} "
            f"cache_budget={prefill_info['cache_budget']} "
            f"mask_ones_sum={int(pm.sum().item())}"
        )

        next_logits = prefill_out.logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        for gen_i in range(args.max_new_tokens):
            outputs = model(
                input_ids=next_id,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True,
                attention_mask=None,
            )
            past_key_values = outputs.past_key_values
            decode_attn = outputs.attentions[-1].to(dtype=dtype)
            info = h2o.step(decode_attn, assert_single_batch=True)
            mask = info["attention_masks_next"]
            print(
                f"[decode] gen={gen_i + 1} key_len={info['key_len']} "
                f"cache_budget={info['cache_budget']} "
                f"mask_ones_sum={int(mask.sum().item())}"
            )
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

    print("Done.")


if __name__ == "__main__":
    main()
