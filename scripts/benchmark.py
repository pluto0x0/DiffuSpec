"""
Benchmark speculative decoding speedup across datasets.

Runs AR-greedy baseline and DiffuSpec on each dataset, reports per-dataset
MAT, throughput, and speedup (DiffuSpec tok/s ÷ AR tok/s).

Both methods share the same loaded target model; DiffuSpec additionally loads
the DLM drafter.  Models are loaded once and reused across all datasets.

Usage:
    python scripts/benchmark.py \\
        --target-model Qwen/Qwen2.5-32B-Instruct \\
        --drafter-model dream-org/dream-v0-instruct-7b \\
        --data-dir data \\
        --datasets gsm8k humaneval alpaca \\
        --n-samples 50 \\
        --max-new-tokens 128 \\
        --device cuda \\
        --kenlm-model models/kenlm/3-gram.pruned.arpa \\
        --output results/benchmark.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffuspec import DiffuSpec, DiffuSpecConfig, DraftingConfig, CPSConfig, ADLConfig

# ── Dataset discovery ───────────────────────────────────────────────────────

DATASET_NAMES = [
    "alpaca", "bbh", "gsm8k", "humaneval", "leetcode",
    "math", "mmlu", "mt_bench", "qa", "sum", "theoremqa",
]


def load_dataset(data_dir: Path, name: str, n_samples: Optional[int]) -> List[str]:
    """Return a list of first-turn prompts from data/{name}/question.jsonl."""
    path = data_dir / name / "question.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    prompts: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = obj.get("turns", [])
            if turns:
                prompts.append(turns[0])  # use first turn only
            if n_samples is not None and len(prompts) >= n_samples:
                break
    return prompts


# ── AR baseline ─────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_ar(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    """
    Run the target model in plain autoregressive greedy mode.

    Returns wall-clock time and number of new tokens generated.
    """
    ids = input_ids.unsqueeze(0).to(device)
    _sync()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
    )
    _sync()
    elapsed = time.perf_counter() - t0
    n_new = out.shape[1] - input_ids.shape[0]
    return {"wall_s": elapsed, "n_tokens": n_new}


# ── DiffuSpec run ────────────────────────────────────────────────────────────

def run_diffuspec(
    engine: DiffuSpec,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Run DiffuSpec speculative decoding.

    Returns wall-clock time, tokens generated, and MAT.
    """
    t0 = time.perf_counter()
    generated_ids, stats = engine.generate(
        prefix_ids=input_ids,
        max_new_tokens=max_new_tokens,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    n_tokens = len(generated_ids)
    mat = stats.get("mean_accepted", 0.0)
    return {"wall_s": elapsed, "n_tokens": n_tokens, "mat": mat}


# ── Per-dataset benchmark ────────────────────────────────────────────────────

def benchmark_dataset(
    name: str,
    prompts: List[str],
    engine: DiffuSpec,
    tokenizer,
    max_new_tokens: int,
    device: str,
    warmup_steps: int = 2,
) -> Dict[str, Any]:
    """
    Run AR and DiffuSpec on all prompts, return aggregate stats.

    Warmup runs are not counted in the final timing.
    """
    ar_model = engine.verifier.model

    def tokenize(prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(text, return_tensors="pt").input_ids[0]

    # Warmup
    print(f"  [warmup {warmup_steps} steps]", end=" ", flush=True)
    for prompt in prompts[:warmup_steps]:
        ids = tokenize(prompt)
        run_ar(ar_model, ids, max_new_tokens=32, device=device)
        run_diffuspec(engine, ids, max_new_tokens=32)
    print("done")

    # Timed runs
    ar_results: List[Dict] = []
    ds_results: List[Dict] = []

    for i, prompt in enumerate(prompts):
        print(f"\r  sample {i+1}/{len(prompts)}", end="", flush=True)
        ids = tokenize(prompt)

        ar = run_ar(ar_model, ids, max_new_tokens, device)
        ds = run_diffuspec(engine, ids, max_new_tokens)

        ar_results.append(ar)
        ds_results.append(ds)

    print()

    # Aggregate
    ar_total_s   = sum(r["wall_s"]   for r in ar_results)
    ar_total_tok = sum(r["n_tokens"] for r in ar_results)
    ds_total_s   = sum(r["wall_s"]   for r in ds_results)
    ds_total_tok = sum(r["n_tokens"] for r in ds_results)

    ar_tps  = ar_total_tok / ar_total_s if ar_total_s > 0 else 0.0
    ds_tps  = ds_total_tok / ds_total_s if ds_total_s > 0 else 0.0
    speedup = ds_tps / ar_tps if ar_tps > 0 else 0.0
    mat     = sum(r["mat"] for r in ds_results) / len(ds_results) if ds_results else 0.0

    return {
        "dataset":       name,
        "n_samples":     len(prompts),
        "ar_tok_per_s":  round(ar_tps, 2),
        "ds_tok_per_s":  round(ds_tps, 2),
        "speedup":       round(speedup, 3),
        "mat":           round(mat, 3),
        "ar_total_s":    round(ar_total_s, 2),
        "ds_total_s":    round(ds_total_s, 2),
        "ar_total_tok":  ar_total_tok,
        "ds_total_tok":  ds_total_tok,
        "per_sample":    [
            {
                "ar_wall_s": round(a["wall_s"], 4),
                "ar_tokens": a["n_tokens"],
                "ds_wall_s": round(d["wall_s"], 4),
                "ds_tokens": d["n_tokens"],
                "mat":       round(d["mat"], 3),
                "speedup":   round(
                    (d["n_tokens"] / d["wall_s"]) / (a["n_tokens"] / a["wall_s"]), 3
                ) if a["n_tokens"] > 0 and a["wall_s"] > 0 else 0.0,
            }
            for a, d in zip(ar_results, ds_results)
        ],
    }


# ── Summary table ────────────────────────────────────────────────────────────

def print_table(rows: List[Dict]) -> None:
    header = f"{'Dataset':<14} {'N':>4} {'AR tok/s':>10} {'DS tok/s':>10} {'Speedup':>9} {'MAT':>7}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['dataset']:<14} {r['n_samples']:>4} "
            f"{r['ar_tok_per_s']:>10.1f} {r['ds_tok_per_s']:>10.1f} "
            f"{r['speedup']:>8.3f}x {r['mat']:>7.3f}"
        )
    print(sep)

    # Macro-average across datasets
    mean_speedup = sum(r["speedup"] for r in rows) / len(rows)
    mean_mat     = sum(r["mat"]     for r in rows) / len(rows)
    # Micro-aggregate throughput
    total_ar_tok = sum(r["ar_total_tok"] for r in rows)
    total_ds_tok = sum(r["ds_total_tok"] for r in rows)
    total_ar_s   = sum(r["ar_total_s"]   for r in rows)
    total_ds_s   = sum(r["ds_total_s"]   for r in rows)
    micro_ar_tps = total_ar_tok / total_ar_s  if total_ar_s  > 0 else 0.0
    micro_ds_tps = total_ds_tok / total_ds_s  if total_ds_s  > 0 else 0.0
    micro_speedup = micro_ds_tps / micro_ar_tps if micro_ar_tps > 0 else 0.0
    print(
        f"{'MEAN (macro)':<14} {sum(r['n_samples'] for r in rows):>4} "
        f"{micro_ar_tps:>10.1f} {micro_ds_tps:>10.1f} "
        f"{mean_speedup:>8.3f}x {mean_mat:>7.3f}"
    )
    print(f"{'MICRO speedup':<14} {'':>4} {'':>10} {'':>10} {micro_speedup:>8.3f}x")
    print(sep + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark DiffuSpec speculative decoding speedup across datasets."
    )
    p.add_argument("--target-model",  type=str, default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--drafter-model", type=str, default="dream-org/dream-v0-instruct-7b")
    p.add_argument("--data-dir",      type=str, default="data")
    p.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="NAME",
        help=f"Datasets to run (default: all). Choices: {DATASET_NAMES}",
    )
    p.add_argument("--n-samples",      type=int, default=None,
                   help="Max samples per dataset (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--warmup-steps",   type=int, default=2,
                   help="Warmup prompts before timing (not counted in results)")

    # DiffuSpec hyperparameters
    p.add_argument("--k-min",          type=int,   default=20)
    p.add_argument("--k-max",          type=int,   default=30)
    p.add_argument("--delta",          type=int,   default=10)
    p.add_argument("--rho",            type=float, default=0.5)
    p.add_argument("--M-max",          type=int,   default=15)
    p.add_argument("--tau",            type=float, default=0.8)
    p.add_argument("--beam-size",      type=int,   default=3)
    p.add_argument("--lam",            type=float, default=0.5)
    p.add_argument("--refinement-steps", type=int, default=1)
    p.add_argument("--kenlm-model",    type=str,   default=None,
                   metavar="PATH",
                   help="KenLM model path. Omit to use UniformProxy.")

    p.add_argument("--output", type=str, default=None,
                   metavar="PATH",
                   help="Save full results to this JSON file.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    datasets = args.datasets or DATASET_NAMES
    # Filter to datasets that actually exist
    available = [d for d in datasets if (data_dir / d / "question.jsonl").exists()]
    missing   = [d for d in datasets if d not in available]
    if missing:
        print(f"[warn] datasets not found, skipping: {missing}")
    if not available:
        print("No datasets found. Check --data-dir.")
        sys.exit(1)

    # ── Load target model once ───────────────────────────────────────────────
    print(f"Loading target model: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    target_model.eval()

    # ── Build DiffuSpec engine (reuses the loaded target model) ─────────────
    print(f"Loading DLM drafter: {args.drafter_model}")
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
        temperature=0.0,
        kenlm_model_path=args.kenlm_model,
    )
    engine = DiffuSpec.from_config(
        config,
        target_model=target_model,
        target_tokenizer=tokenizer,
        device=args.device,
    )

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results: List[Dict] = []

    for name in available:
        prompts = load_dataset(data_dir, name, args.n_samples)
        print(f"\n[{name}] {len(prompts)} samples, max_new_tokens={args.max_new_tokens}")
        result = benchmark_dataset(
            name=name,
            prompts=prompts,
            engine=engine,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            warmup_steps=args.warmup_steps,
        )
        all_results.append(result)
        print(
            f"  AR {result['ar_tok_per_s']:.1f} tok/s | "
            f"DS {result['ds_tok_per_s']:.1f} tok/s | "
            f"speedup {result['speedup']:.3f}x | MAT {result['mat']:.3f}"
        )

    # ── Print summary ────────────────────────────────────────────────────────
    print_table(all_results)

    # ── Save results ─────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "target_model":     args.target_model,
                "drafter_model":    args.drafter_model,
                "max_new_tokens":   args.max_new_tokens,
                "kenlm_model":      args.kenlm_model,
                "k_min":            args.k_min,
                "k_max":            args.k_max,
                "delta":            args.delta,
                "rho":              args.rho,
                "M_max":            args.M_max,
                "tau":              args.tau,
                "beam_size":        args.beam_size,
                "lam":              args.lam,
                "refinement_steps": args.refinement_steps,
            },
            "datasets": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
