"""
Benchmark speculative decoding speedup across datasets.

Runs AR-greedy baseline and one or more speculative decoding engines on each
dataset, reporting per-dataset MAT, throughput, and speedup.

The target model is loaded once and shared across all engines.  When --mode all
is used, the DLM drafter and verifier are also shared between DiffuSpec and
NaiveSpec to avoid loading duplicate weights.

Modes:
  diffuspec  — DiffuSpec (CPS + ADL) vs AR baseline  [default]
  naive      — Naive DLM spec-dec (argmax draft, no CPS/ADL) vs AR baseline
  all        — Both engines vs AR baseline in one run

Usage:
    python scripts/benchmark.py \\
        --target-model Qwen/Qwen2.5-32B-Instruct \\
        --drafter-model dream-org/dream-v0-instruct-7b \\
        --data-dir data \\
        --datasets gsm8k humaneval alpaca \\
        --n-samples 50 \\
        --max-new-tokens 128 \\
        --device cuda \\
        --mode all \\
        --draft-len 5 \\
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

from diffuspec import (
    DiffuSpec, DiffuSpecConfig, DraftingConfig, CPSConfig, ADLConfig,
    NaiveSpec, NaiveSpecConfig,
)

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


# ── Runners ──────────────────────────────────────────────────────────────────

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
    """Run the target model in plain autoregressive greedy mode."""
    ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(ids)
    eos_id = model.config.eos_token_id
    pad_token_id = eos_id[0] if isinstance(eos_id, list) else eos_id
    _sync()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
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


def run_engine(engine, input_ids: torch.Tensor, max_new_tokens: int) -> Dict[str, Any]:
    """Run any spec-dec engine that exposes .generate(prefix_ids, max_new_tokens)."""
    t0 = time.perf_counter()
    generated_ids, stats = engine.generate(
        prefix_ids=input_ids,
        max_new_tokens=max_new_tokens,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "wall_s":   elapsed,
        "n_tokens": len(generated_ids),
        "mat":      stats.get("mean_accepted", 0.0),
    }


# ── Per-dataset benchmark ────────────────────────────────────────────────────

def benchmark_dataset(
    name: str,
    prompts: List[str],
    engines: Dict[str, Any],   # ordered {display_name: engine}
    tokenizer,
    max_new_tokens: int,
    device: str,
    warmup_steps: int = 2,
) -> Dict[str, Any]:
    """
    Run AR baseline and every engine on all prompts, return aggregate stats.

    Warmup runs are not counted in the final timing.
    """
    ar_model = next(iter(engines.values())).verifier.model

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
        for engine in engines.values():
            run_engine(engine, ids, max_new_tokens=32)
    print("done")

    # Timed runs
    ar_results: List[Dict] = []
    engine_results: Dict[str, List[Dict]] = {m: [] for m in engines}

    for i, prompt in enumerate(prompts):
        print(f"\r  sample {i+1}/{len(prompts)}", end="", flush=True)
        ids = tokenize(prompt)
        ar_results.append(run_ar(ar_model, ids, max_new_tokens, device))
        for method, engine in engines.items():
            engine_results[method].append(run_engine(engine, ids, max_new_tokens))

    print()

    # Aggregate AR
    ar_total_s   = sum(r["wall_s"]   for r in ar_results)
    ar_total_tok = sum(r["n_tokens"] for r in ar_results)
    ar_tps = ar_total_tok / ar_total_s if ar_total_s > 0 else 0.0

    # Aggregate each engine
    methods_agg: Dict[str, Dict] = {}
    for method, results in engine_results.items():
        total_s   = sum(r["wall_s"]   for r in results)
        total_tok = sum(r["n_tokens"] for r in results)
        tps = total_tok / total_s if total_s > 0 else 0.0
        mat = sum(r["mat"] for r in results) / len(results) if results else 0.0
        methods_agg[method] = {
            "tok_per_s": round(tps, 2),
            "speedup":   round(tps / ar_tps, 3) if ar_tps > 0 else 0.0,
            "mat":       round(mat, 3),
            "total_s":   round(total_s, 2),
            "total_tok": total_tok,
        }

    # Per-sample breakdown
    per_sample = []
    for i, a in enumerate(ar_results):
        row: Dict[str, Any] = {
            "ar_wall_s": round(a["wall_s"], 4),
            "ar_tokens": a["n_tokens"],
        }
        for method in engines:
            e = engine_results[method][i]
            per_sample_speedup = (
                round((e["n_tokens"] / e["wall_s"]) / (a["n_tokens"] / a["wall_s"]), 3)
                if a["n_tokens"] > 0 and a["wall_s"] > 0 and e["wall_s"] > 0
                else 0.0
            )
            row[f"{method}_wall_s"] = round(e["wall_s"], 4)
            row[f"{method}_tokens"] = e["n_tokens"]
            row[f"{method}_mat"]    = round(e["mat"], 3)
            row[f"{method}_speedup"] = per_sample_speedup
        per_sample.append(row)

    return {
        "dataset":      name,
        "n_samples":    len(prompts),
        "ar_tok_per_s": round(ar_tps, 2),
        "ar_total_s":   round(ar_total_s, 2),
        "ar_total_tok": ar_total_tok,
        "methods":      methods_agg,
        "per_sample":   per_sample,
    }


# ── Summary table ────────────────────────────────────────────────────────────

def print_table(rows: List[Dict], method_names: List[str]) -> None:
    """
    Print a summary table.  One group of (tok/s, speedup, MAT) columns per method.
    """
    # Column widths
    W_DS = 14  # "Dataset"
    W_N  =  5  # "N"
    W_AR = 10  # "AR tok/s"
    W_M  = 10  # "<method> tok/s"
    W_SP =  9  # "speedup"
    W_MT =  7  # "MAT"

    # Header
    hdr = f"{'Dataset':<{W_DS}} {'N':>{W_N}} {'AR tok/s':>{W_AR}}"
    for m in method_names:
        label = m[:W_M]  # truncate if needed
        hdr += f"  {label:>{W_M}} {'speedup':>{W_SP}} {'MAT':>{W_MT}}"
    sep = "-" * len(hdr)

    print("\n" + sep)
    print(hdr)
    print(sep)

    for r in rows:
        line = f"{r['dataset']:<{W_DS}} {r['n_samples']:>{W_N}} {r['ar_tok_per_s']:>{W_AR}.1f}"
        for m in method_names:
            mg = r["methods"].get(m, {})
            tps     = mg.get("tok_per_s", 0.0)
            speedup = mg.get("speedup",   0.0)
            mat     = mg.get("mat",       0.0)
            line += f"  {tps:>{W_M}.1f} {speedup:>{W_SP-1}.3f}x {mat:>{W_MT}.3f}"
        print(line)

    print(sep)

    # Footer: macro-average speedup + MAT; micro-aggregate throughput
    total_n       = sum(r["n_samples"]    for r in rows)
    total_ar_tok  = sum(r["ar_total_tok"] for r in rows)
    total_ar_s    = sum(r["ar_total_s"]   for r in rows)
    micro_ar_tps  = total_ar_tok / total_ar_s if total_ar_s > 0 else 0.0

    macro_line = f"{'MEAN (macro)':<{W_DS}} {total_n:>{W_N}} {micro_ar_tps:>{W_AR}.1f}"
    micro_line = f"{'MICRO speedup':<{W_DS}} {'':>{W_N}} {'':>{W_AR}}"

    for m in method_names:
        mean_speedup = sum(r["methods"][m]["speedup"] for r in rows if m in r["methods"]) / len(rows)
        mean_mat     = sum(r["methods"][m]["mat"]     for r in rows if m in r["methods"]) / len(rows)
        total_m_tok  = sum(r["methods"][m]["total_tok"] for r in rows if m in r["methods"])
        total_m_s    = sum(r["methods"][m]["total_s"]   for r in rows if m in r["methods"])
        micro_m_tps  = total_m_tok / total_m_s if total_m_s > 0 else 0.0
        micro_speedup = micro_m_tps / micro_ar_tps if micro_ar_tps > 0 else 0.0

        macro_line += f"  {micro_m_tps:>{W_M}.1f} {mean_speedup:>{W_SP-1}.3f}x {mean_mat:>{W_MT}.3f}"
        micro_line += f"  {'':>{W_M}} {micro_speedup:>{W_SP-1}.3f}x {'':>{W_MT}}"

    print(macro_line)
    print(micro_line)
    print(sep + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark speculative decoding speedup across datasets."
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

    # Mode
    p.add_argument(
        "--mode",
        type=str,
        choices=["diffuspec", "naive", "all"],
        default="diffuspec",
        help=(
            "'diffuspec' (default): DiffuSpec (CPS+ADL) vs AR. "
            "'naive': naive DLM spec-dec vs AR. "
            "'all': both engines vs AR in one run (shared drafter weights)."
        ),
    )

    # Naive mode: fixed draft length
    p.add_argument(
        "--draft-len", type=int, default=5, metavar="K",
        help="Fixed draft length per step for naive mode (default: 5).",
    )

    # DiffuSpec hyperparameters (used in modes: diffuspec, all)
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
                   help="KenLM model path for DiffuSpec CPS. Omit for UniformProxy.")

    p.add_argument("--output", type=str, default=None,
                   metavar="PATH",
                   help="Save full results to this JSON file.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    datasets = args.datasets or DATASET_NAMES
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

    drafting_cfg = DraftingConfig(
        model_name=args.drafter_model,
        num_refinement_steps=args.refinement_steps,
    )

    # ── Build engine(s) ──────────────────────────────────────────────────────
    engines: Dict[str, Any] = {}

    if args.mode in ("diffuspec", "all"):
        print(f"Loading DLM drafter for DiffuSpec: {args.drafter_model}")
        ds_config = DiffuSpecConfig(
            drafting=drafting_cfg,
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
        engines["diffuspec"] = DiffuSpec.from_config(
            ds_config,
            target_model=target_model,
            target_tokenizer=tokenizer,
            device=args.device,
        )

    if args.mode in ("naive", "all"):
        naive_config = NaiveSpecConfig(
            drafting=drafting_cfg,
            target_model_name=args.target_model,
            draft_len=args.draft_len,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        if args.mode == "all":
            # Share drafter + verifier with DiffuSpec to avoid loading duplicate weights.
            naive_engine = NaiveSpec(
                drafter=engines["diffuspec"].drafter,
                verifier=engines["diffuspec"].verifier,
                config=naive_config,
            )
        else:
            print(f"Loading DLM drafter for NaiveSpec: {args.drafter_model}")
            naive_engine = NaiveSpec.from_config(
                naive_config,
                target_model=target_model,
                target_tokenizer=tokenizer,
                device=args.device,
            )
        engines["naive"] = naive_engine

    method_names = list(engines.keys())
    print(f"Benchmarking mode: {args.mode}  |  engines: {method_names}\n")

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results: List[Dict] = []

    for name in available:
        prompts = load_dataset(data_dir, name, args.n_samples)
        print(f"[{name}] {len(prompts)} samples, max_new_tokens={args.max_new_tokens}")
        result = benchmark_dataset(
            name=name,
            prompts=prompts,
            engines=engines,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            warmup_steps=args.warmup_steps,
        )
        all_results.append(result)

        # Inline summary line
        parts = [f"AR {result['ar_tok_per_s']:.1f} tok/s"]
        for m in method_names:
            mg = result["methods"][m]
            parts.append(
                f"{m}: {mg['tok_per_s']:.1f} tok/s  {mg['speedup']:.3f}x  MAT {mg['mat']:.3f}"
            )
        print("  " + "  |  ".join(parts))

    # ── Print summary table ───────────────────────────────────────────────────
    print_table(all_results, method_names)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "mode":             args.mode,
                "target_model":     args.target_model,
                "drafter_model":    args.drafter_model,
                "max_new_tokens":   args.max_new_tokens,
                "draft_len":        args.draft_len,
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
