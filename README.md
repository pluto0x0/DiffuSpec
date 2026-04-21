# DiffuSpec

Reproduction of **DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding** (arXiv 2510.02358v1).

DiffuSpec is a training-free speculative decoding framework that uses a pretrained Diffusion LM (Dream-7B) as the drafter. Two components improve acceptance rate over a plain DLM drafter:

- **CPS** (Causal-Consistency Path Search) — selects a causally-consistent left-to-right path from the DLM token-probability lattice via beam search guided by a lightweight n-gram proxy.
- **ADL** (Adaptive Draft-Length) — adjusts the next draft length online using EMA signals of generation length and accepted length.

---

## Requirements

- Python 3.10
- CUDA GPU (A100 80GB recommended for the full Qwen2.5-32B + Dream-7B setup; smaller configs work with smaller target models)

Install the environment:

```bash
conda env create -f environment.yml
conda activate diffuspec
```

Key dependencies: `torch==2.6.0`, `transformers>=4.47,<5`, `kenlm`, `pandas`.

---

## Quick Start

### 1. Download a pre-trained KenLM proxy model

The CPS beam search uses a 3-gram KenLM as a causal fluency signal. Download a generic English model (OpenSLR Resource 11, ~30 MB compressed):

```bash
python scripts/download_kenlm.py --out-dir models/kenlm
```

This saves `models/kenlm/3-gram.pruned.arpa`. For task-specific benchmarks matching the paper, train your own model on the dataset's training split:

```bash
lmplz -o 3 < train_text.txt > models/kenlm/task.arpa
# optional: convert to fast binary format
build_binary models/kenlm/task.arpa models/kenlm/task.bin
```

### 2. Run generation

```bash
export HF_HOME=/workspace/.hf_home

python scripts/generate.py \
    --prompt "Explain the concept of speculative decoding." \
    --target-model Qwen/Qwen2.5-32B-Instruct \
    --drafter-model dream-org/dream-v0-instruct-7b \
    --kenlm-model models/kenlm/3-gram.pruned.arpa \
    --max-new-tokens 256 \
    --device cuda \
    --verbose
```

Without `--kenlm-model`, CPS falls back to a uniform proxy (equivalent to DLM-only path scoring).

---

## Hyperparameters

All defaults match the paper (Appendix A / Sec. 5):

| Flag | Default | Paper param |
|---|---|---|
| `--k-min` | 20 | ADL k_min |
| `--k-max` | 30 | ADL k_max |
| `--delta` | 10 | ADL δ |
| `--rho` | 0.5 | ADL ρ |
| `--M-max` | 15 | CPS M_max |
| `--tau` | 0.8 | CPS τ |
| `--beam-size` | 3 | CPS B |
| `--lam` | 0.5 | CPS λ |
| `--refinement-steps` | 1 | DLM S |

---

## Code Structure

```
diffuspec/
├── config.py               # DiffuSpecConfig and sub-configs (ADL / CPS / Drafting)
├── engine.py               # Algorithm 1: 4-stage pipeline (Draft → CPS → Verify → ADL)
├── drafting/
│   └── dlm_drafter.py      # Stage 1 — Dream-7B wrapper: draft() + compute_l2r_logprobs()
├── search/
│   └── cps.py              # Stage 2 — beam search over pruned DLM lattice (Eq. 7–8)
├── control/
│   └── adl.py              # Stage 4 — adaptive draft-length controller (Eq. 10–11)
├── verification/
│   └── verifier.py         # Stage 3 — AR parallel verification (greedy + stochastic)
└── proxy/
    └── ngram_proxy.py      # Causal proxies: NgramProxy / UniformProxy / KenLMProxy
scripts/
├── generate.py             # CLI generation entry point
└── download_kenlm.py       # Download pre-trained English 3-gram KenLM
tests/
└── test_components.py      # 22 unit tests (ADL, CPS, NgramProxy, KenLMProxy)
```

---

## Running Tests

```bash
python -m pytest tests/test_components.py -v
```

All 22 tests cover ADLController, CausalConsistencyPathSearch, NgramProxy, and KenLMProxy.

---

## Algorithm Overview

Each speculative step follows Algorithm 1 from the paper:

1. **Draft** — DLM produces a length-k_t block and per-position top-M candidate sets with log-scores.
2. **CPS** — Prune candidates by cumulative-mass threshold τ (Eq. 7), then run left-to-right beam search (width B) scoring paths with λ·ℓ^dlm + (1−λ)·ℓ^ng (Eq. 8). Early-stop after first EOS.
3. **Verify** — Target AR model verifies the selected path in parallel using the standard acceptance rule (Eq. 1) with q^L2R drafter log-probs.
4. **ADL** — Update EMA signals L̄^gen and L̄^acc (Eq. 10); set next draft length k_{t+1} via Eq. 11.

---

## Notes on Paper Fidelity

- **Proxy**: the paper uses a 3-gram KenLM fitted per dataset. `--kenlm-model` enables this; without it the proxy is uniform (λ is effectively 1).
- **Target model**: the paper uses Qwen2.5-32B-Instruct. Smaller targets (e.g. Qwen3-8B) work but produce different speedup numbers.
- **Vocab alignment**: Qwen3-8B (151936) and Dream-7B (152064) have a 128-token mismatch; the drafter clips OOB IDs to target vocab size automatically.

---

## Citation

```bibtex
@article{li2025diffuspec,
  title={DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding},
  author={Li, Guanghao and Fu, Zhihui and Fang, Min and Zhao, Qibin and Tang, Ming and Yuan, Chun and Wang, Jun},
  journal={arXiv preprint arXiv:2510.02358},
  year={2025}
}
```
