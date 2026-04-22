"""
Causal-Consistency Path Search (CPS) — paper section 4.2.

Given a DLM token-probability lattice, CPS extracts a left-to-right path that is
both high-confidence under the DLM and fluent under a causal proxy (n-gram LM).

Algorithm:
  1. Prune each position's candidates to a cumulative-mass prefix of size M_i
     (entropy-adaptive; Eq. 7).  Always retain EOS.
  2. Early-stop lattice expansion after the first EOS is encountered.
  3. Run left-to-right beam search (width B) scoring each partial path with
     S(π) = Σ_i [ λ·ℓ^dlm_i(πᵢ) + (1−λ)·ℓ^ng_i(prefix ∘ π_{1:i}) ]  (Eq. 8).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch


class BeamHypothesis:
    """A single beam hypothesis: token sequence + accumulated score."""

    __slots__ = ("tokens", "score", "finished")

    def __init__(self, tokens: List[int], score: float, finished: bool = False) -> None:
        self.tokens = tokens
        self.score = score
        self.finished = finished  # True once EOS is placed


class CausalConsistencyPathSearch:
    """
    Left-to-right beam search over the pruned DLM candidate lattice.

    Args:
        eos_token_id : EOS token ID (always kept in candidates; terminates beam).
        M_max        : per-position candidate cap (paper: M_max=15).
        tau          : cumulative-mass pruning threshold (paper: τ=0.8).
        beam_size    : beam width B (paper: B=3).
        lam          : DLM / proxy mixing weight λ (paper: λ=0.5).
    """

    def __init__(
        self,
        eos_token_id: int,
        M_max: int = 15,
        tau: float = 0.8,
        beam_size: int = 3,
        lam: float = 0.5,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.M_max = M_max
        self.tau = tau
        self.beam_size = beam_size
        self.lam = lam

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        prefix_ids: List[int],
        dlm_log_probs: torch.Tensor,    # [draft_len, V]  — from DLM forward pass
        causal_proxy,                    # NgramProxy / KenLMProxy / UniformProxy
    ) -> Tuple[List[int], int]:
        """
        Run CPS over the draft lattice.

        Returns:
            path      : selected token-ID sequence (may be shorter than draft_len
                        if early-stopped at first EOS).
            path_len  : length of the returned path (m_t in the paper).
        """
        draft_len = dlm_log_probs.shape[0]

        # Step 1: build pruned per-position candidate sets
        candidate_sets = self._build_candidate_sets(dlm_log_probs)  # List[List[(tok, dlm_lp)]]

        # Step 2: left-to-right beam search
        # Each beam state: (partial_path_tokens, accumulated_score)
        beams: List[BeamHypothesis] = [BeamHypothesis(tokens=[], score=0.0)]

        for pos, candidates in enumerate(candidate_sets):
            if not candidates:
                break

            new_beams: List[BeamHypothesis] = []
            cand_tok_ids = [tok_id for tok_id, _ in candidates]
            for hyp in beams:
                if hyp.finished:
                    new_beams.append(hyp)
                    continue

                # Build context once per hypothesis (constant across all candidates).
                context = prefix_ids + hyp.tokens
                # Batch proxy scoring: decodes context string only once (KenLMProxy).
                ng_lps = causal_proxy.score_tokens_batch(context, cand_tok_ids)

                for (tok_id, dlm_lp), ng_lp in zip(candidates, ng_lps):
                    # Eq. 8: combined score
                    step_score = self.lam * dlm_lp + (1.0 - self.lam) * ng_lp
                    new_hyp = BeamHypothesis(
                        tokens=hyp.tokens + [tok_id],
                        score=hyp.score + step_score,
                        finished=(tok_id == self.eos_token_id),
                    )
                    new_beams.append(new_hyp)

            # Keep top-B hypotheses (finished ones compete equally)
            new_beams.sort(key=lambda h: h.score, reverse=True)
            beams = new_beams[: self.beam_size]

            # Early-stop: if every active beam has placed EOS, no need to expand further
            if all(h.finished for h in beams):
                break

        # Best path
        best = beams[0]
        return best.tokens, len(best.tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidate_sets(
        self, dlm_log_probs: torch.Tensor  # [draft_len, V]
    ) -> List[List[Tuple[int, float]]]:
        """
        For each position build the pruned candidate set (Eq. 7).

        Retains the smallest prefix of top-probability tokens whose
        cumulative mass ≥ τ, capped at M_max.  EOS is always included.

        Returns:
            List of length draft_len, each element is a list of (token_id, log_prob).
        """
        # Ensure float32: bfloat16 has no native CPU instructions, making .exp() and
        # any sort-like op ~100x slower than float32 on CPU.
        if dlm_log_probs.dtype != torch.float32:
            dlm_log_probs = dlm_log_probs.float()

        draft_len, V = dlm_log_probs.shape
        k = min(self.M_max, V)

        # Batched tensor ops over all positions at once — one exp, one topk, one gather,
        # three tolist() calls instead of 3×draft_len separate operations.
        all_probs = dlm_log_probs.exp()                                          # [draft_len, V]
        top_probs, top_indices = torch.topk(all_probs, k, dim=1, sorted=True)   # [draft_len, k]
        lp_top = dlm_log_probs.gather(1, top_indices)                            # [draft_len, k]

        all_probs_list   = top_probs.tolist()    # [draft_len][k]  — Python floats, no more .item()
        all_indices_list = top_indices.tolist()  # [draft_len][k]
        all_lp_list      = lp_top.tolist()       # [draft_len][k]

        # Pre-fetch EOS log-probs for all positions (used only when EOS falls outside top-k).
        eos_lp_list: List[float] = dlm_log_probs[:, self.eos_token_id].tolist()

        # Per-position pruning — pure Python arithmetic, no tensor ops inside the loop.
        candidate_sets: List[List[Tuple[int, float]]] = []
        eos_encountered = False

        for pos in range(draft_len):
            if eos_encountered:
                break

            tok_ids   = all_indices_list[pos]
            probs     = all_probs_list[pos]
            lp_vals   = all_lp_list[pos]

            candidates: List[Tuple[int, float]] = []
            cumulative = 0.0
            eos_added = False

            for tok_id, p, lp_val in zip(tok_ids, probs, lp_vals):
                if tok_id == self.eos_token_id:
                    eos_added = True

                candidates.append((tok_id, lp_val))
                cumulative += p

                if len(candidates) >= self.M_max:
                    break
                if cumulative >= self.tau and eos_added:
                    break

            # Guarantee EOS is present if it fell outside the top-k window
            if not eos_added:
                candidates.append((self.eos_token_id, eos_lp_list[pos]))

            candidate_sets.append(candidates)

            # Only stop lattice expansion when EOS ranked naturally in the top-k
            # candidates (eos_added=True from inside the loop). Force-appended EOS
            # (added below M_max purely as a guarantee) should NOT cut the draft short.
            if eos_added:
                eos_encountered = True

        return candidate_sets
