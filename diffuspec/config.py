"""
DiffuSpec configuration — all hyperparameters in one place.

Paper defaults (Appendix A, Table 1):
  k_min=20, k_max=30, delta=10, rho=0.5   (ADL controller)
  M_max=15, tau=0.8, beam_size=3, lam=0.5 (CPS)
  num_refinement_steps=1                   (DLM drafting, S=1)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DraftingConfig:
    """DLM drafter settings."""
    model_name: str = "dream-org/dream-v0-instruct-7b"
    # Number of denoising refinement steps (S). Paper uses S=1 to isolate drafting cost.
    num_refinement_steps: int = 1
    dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"


@dataclass
class CPSConfig:
    """Causal-Consistency Path Search (CPS) settings."""
    # Per-position candidate cap (M_max). Larger → wider lattice, more compute.
    M_max: int = 15
    # Cumulative-mass threshold for pruning. τ=0.8 retains high-confidence tokens.
    tau: float = 0.8
    # Beam width for left-to-right beam search.
    beam_size: int = 3
    # Mixing weight λ ∈ [0,1] between DLM score and causal-proxy score (Eq. 8).
    # λ=1 → DLM only; λ=0 → proxy only.
    lam: float = 0.5


@dataclass
class ADLConfig:
    """Adaptive Draft-Length (ADL) controller settings."""
    # Minimum/maximum allowed draft length.
    k_min: int = 20
    k_max: int = 30
    # Growth increment δ: k increases by δ when verifier keeps up (Eq. 11).
    delta: int = 10
    # EMA smoothing factor ρ ∈ (0, 1] (Eq. 10).
    rho: float = 0.5


@dataclass
class DiffuSpecConfig:
    """Top-level config aggregating all sub-configs."""
    drafting: DraftingConfig = field(default_factory=DraftingConfig)
    cps: CPSConfig = field(default_factory=CPSConfig)
    adl: ADLConfig = field(default_factory=ADLConfig)

    # Target LM settings
    target_model_name: str = "Qwen/Qwen2.5-32B-Instruct"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.0   # 0 → greedy (quality-locked, as in paper)
    eos_token_id: Optional[int] = None

    # Path to a pre-trained KenLM model (.arpa / .arpa.gz / .bin).
    # When set, CPS uses KenLMProxy instead of UniformProxy, matching the paper
    # which fits a 3-gram KenLM on each dataset's training split (Sec. 4.2).
    kenlm_model_path: Optional[str] = None


@dataclass
class NaiveSpecConfig:
    """
    Config for the naive speculative decoding baseline.

    The naive method uses the DLM drafter's argmax output directly (no CPS, no ADL):
      1. DLM produces a fixed-length draft block.
      2. AR verifier accepts the longest matching prefix (standard spec-dec rule).
    """
    drafting: DraftingConfig = field(default_factory=DraftingConfig)

    # Target LM settings
    target_model_name: str = "Qwen/Qwen2.5-32B-Instruct"

    # Fixed draft length per speculative step (replaces ADL's adaptive k_t).
    draft_len: int = 5

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.0
    eos_token_id: Optional[int] = None
