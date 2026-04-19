"""
CLI script for running DiffuSpec generation.

Example:
    python scripts/generate.py \
        --prompt "Explain the concept of speculative decoding." \
        --target-model Qwen/Qwen2.5-32B-Instruct \
        --drafter-model dream-org/dream-v0-instruct-7b \
        --max-new-tokens 256 \
        --device cuda
"""

import argparse
import sys
import time
import torch

sys.path.insert(0, ".")

from diffuspec import DiffuSpec, DiffuSpecConfig, DraftingConfig, CPSConfig, ADLConfig


def parse_args():
    p = argparse.ArgumentParser(description="DiffuSpec generation")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--drafter-model", type=str, default="dream-org/dream-v0-instruct-7b")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)

    # ADL hyperparameters (paper defaults)
    p.add_argument("--k-min", type=int, default=20)
    p.add_argument("--k-max", type=int, default=30)
    p.add_argument("--delta", type=int, default=10)
    p.add_argument("--rho", type=float, default=0.5)

    # CPS hyperparameters (paper defaults)
    p.add_argument("--M-max", type=int, default=15)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.5)

    # Refinement steps
    p.add_argument("--refinement-steps", type=int, default=1)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    config = DiffuSpecConfig(
        drafting=DraftingConfig(
            model_name=args.drafter_model,
            num_refinement_steps=args.refinement_steps,
        ),
        cps=CPSConfig(
            M_max=args.M_max,
            tau=args.tau,
            beam_size=args.beam_size,
            lam=args.lam,
        ),
        adl=ADLConfig(
            k_min=args.k_min,
            k_max=args.k_max,
            delta=args.delta,
            rho=args.rho,
        ),
        target_model_name=args.target_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"Loading DiffuSpec (target={args.target_model}, drafter={args.drafter_model})...")
    engine = DiffuSpec.from_config(config, device=args.device)

    # Tokenise prompt using the target tokenizer
    tokenizer = engine.verifier.tokenizer
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")[0]

    print(f"\nPrompt: {args.prompt}\n")
    t0 = time.perf_counter()
    generated_ids, stats = engine.generate(
        prefix_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {output_text}\n")
    print(f"--- Stats ---")
    print(f"  Steps        : {stats['n_steps']}")
    print(f"  Total tokens : {stats['total_accepted']}")
    if stats['n_steps'] > 0:
        print(f"  Mean-MAT     : {stats['mean_accepted']:.2f}")
    print(f"  Wall time    : {elapsed:.2f}s")
    tok_per_sec = stats["total_accepted"] / max(elapsed, 1e-6)
    print(f"  Throughput   : {tok_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()
