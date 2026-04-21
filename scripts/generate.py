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

    # Causal proxy (paper uses 3-gram KenLM per dataset)
    p.add_argument(
        "--kenlm-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a pre-trained KenLM model (.arpa / .arpa.gz / .bin). "
             "When omitted, CPS falls back to UniformProxy (DLM-only scoring).",
    )

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
        kenlm_model_path=args.kenlm_model,
    )

    print(f"Loading DiffuSpec (target={args.target_model}, drafter={args.drafter_model})...")
    engine = DiffuSpec.from_config(config, device=args.device)

    # Tokenise prompt using the target tokenizer, applying the instruct chat template
    # so both the target model and drafter see a properly formatted assistant turn.
    tokenizer = engine.verifier.tokenizer
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]

    print(f"\nPrompt: {args.prompt}\n")
    t0 = time.perf_counter()
    generated_ids, stats = engine.generate(
        prefix_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    import pandas as pd

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {output_text}\n")

    # ── Basic stats (always shown) ────────────────────────────────────────
    tok_per_sec = stats["total_accepted"] / max(elapsed, 1e-6)
    basic = pd.DataFrame([
        {"metric": "Steps",         "value": stats["n_steps"]},
        {"metric": "Total tokens",  "value": stats["total_accepted"]},
        {"metric": "Mean-MAT",      "value": f"{stats.get('mean_accepted', 0):.2f}"},
        {"metric": "Wall time (s)", "value": f"{elapsed:.2f}"},
        {"metric": "Throughput",    "value": f"{tok_per_sec:.1f} tok/s"},
    ])
    print("── Stats ──")
    print(basic.to_string(index=False))

    # ── Timing breakdown (verbose only) ──────────────────────────────────
    if args.verbose and stats["n_steps"] > 0:
        draft_times  = stats.get("draft_times_s", [])
        cps_times    = stats.get("cps_times_s", [])
        verify_times = stats.get("verify_times_s", [])
        if draft_times:
            def _ms(lst): return f"{sum(lst)/len(lst)*1e3:.1f} ms"
            timing = pd.DataFrame([
                {"stage": "draft",  "avg": _ms(draft_times),
                 "min": f"{min(draft_times)*1e3:.1f} ms",
                 "max": f"{max(draft_times)*1e3:.1f} ms"},
                {"stage": "CPS",    "avg": _ms(cps_times),
                 "min": f"{min(cps_times)*1e3:.1f} ms",
                 "max": f"{max(cps_times)*1e3:.1f} ms"},
                {"stage": "verify", "avg": _ms(verify_times),
                 "min": f"{min(verify_times)*1e3:.1f} ms",
                 "max": f"{max(verify_times)*1e3:.1f} ms"},
            ])
            print("\n── Timing breakdown ──")
            print(timing.to_string(index=False))


if __name__ == "__main__":
    main()
