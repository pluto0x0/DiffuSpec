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
            for hyp in beams:
                if hyp.finished:
                    new_beams.append(hyp)
                    continue

                for tok_id, dlm_lp in candidates:
                    # Causal proxy score for context = prefix + hyp.tokens + [tok_id]
                    context = prefix_ids + hyp.tokens
                    ng_lp = causal_proxy.score_token(context, tok_id)

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
        candidate_sets: List[List[Tuple[int, float]]] = []
        eos_encountered = False  # stop expanding after first EOS column

        # Ensure float32: bfloat16 has no native CPU instructions, making .exp() and
        # any sort-like op ~100x slower than float32 on CPU.
        if dlm_log_probs.dtype != torch.float32:
            dlm_log_probs = dlm_log_probs.float()

        for pos in range(dlm_log_probs.shape[0]):
            if eos_encountered:
                break

            lp = dlm_log_probs[pos]  # [V], float32

            # topk is O(V) vs argsort O(V log V).  We never need more than M_max
            # tokens, so there is no reason to sort the full vocabulary.
            k = min(self.M_max, lp.shape[0])
            top_probs, top_indices = torch.topk(lp.exp(), k, largest=True, sorted=True)

            # Convert to Python lists once — avoids repeated .item() overhead.
            tok_ids: List[int] = top_indices.tolist()
            probs_list: List[float] = top_probs.tolist()
            lp_vals: List[float] = lp[top_indices].tolist()

            candidates: List[Tuple[int, float]] = []
            cumulative = 0.0
            eos_added = False

            for tok_id, p, lp_val in zip(tok_ids, probs_list, lp_vals):
                if tok_id == self.eos_token_id:
                    eos_added = True

                candidates.append((tok_id, lp_val))
                cumulative += p

                if len(candidates) >= self.M_max:
                    break
                if cumulative >= self.tau and eos_added:
                    break

            # Guarantee EOS is present if it fell outside the top-M window
            if not eos_added:
                eos_lp = lp[self.eos_token_id].item()
                candidates.append((self.eos_token_id, eos_lp))

            candidate_sets.append(candidates)

            # Only stop lattice expansion when EOS ranked naturally in the top-M
            # candidates (eos_added=True from inside the loop). Force-appended EOS
            # (added below M_max purely as a guarantee) should NOT cut the draft short.
            if eos_added:
                eos_encountered = True

        return candidate_sets
