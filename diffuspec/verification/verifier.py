"""
AR Parallel Verifier — paper section 3 & 4.1.

Runs the target autoregressive LM p_θ in a single parallel forward pass over the
draft block and applies the standard speculative-decoding acceptance rule (Eq. 1).

Acceptance ratio at position i (given positions 1..i-1 were accepted):
    α_{t,i} = min( 1,  p_θ(ŷ_{j+i} | x_{1:j+i-1}) / q_φ^{L2R}(ŷ_{j+i} | x_{1:j+i-1}) )

The DLM probability q_φ^{L2R} (Eq. 6) is pre-computed by DLMDrafter.compute_l2r_logprobs.

Rejection:
  - Sample replacement from residual distribution (p - q)_+ normalised.
  - Discard all remaining draft tokens.
  - Accepted prefix length L^acc_t is returned (Eq. 2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class ARVerifier:
    """
    Parallel block verifier using an autoregressive target LM (p_θ).

    The target model must support the standard HuggingFace interface:
        output = model(input_ids=..., use_cache=False)
        logits = output.logits  # [batch, seq, vocab]
    """

    def __init__(self, model, tokenizer, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id: int = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify(
        self,
        prefix_ids: torch.Tensor,      # [prefix_len]
        draft_ids: torch.Tensor,       # [draft_len]  — the CPS-selected path
        drafter_l2r_logprobs: torch.Tensor,  # [draft_len]  — log q_φ^{L2R}(ŷ_i | ...)
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, int, bool]:
        """
        Verify the draft block against the target LM.

        Args:
            prefix_ids           : current context, shape [prefix_len].
            draft_ids            : proposed tokens, shape [draft_len].
            drafter_l2r_logprobs : log q_φ^{L2R} for each draft token, shape [draft_len].
            temperature          : 0 → greedy (paper default for quality-locked eval).

        Returns:
            accepted_ids : accepted token IDs (may include one replacement token).
            n_accepted   : L^acc_t (number of tokens accepted by the verifier).
            hit_eos      : True if EOS was accepted anywhere in the block.
        """
        prefix_ids = prefix_ids.to(self.device)
        draft_ids = draft_ids.to(self.device)
        draft_len = draft_ids.shape[0]

        # Single parallel forward pass over prefix + draft block
        full_ids = torch.cat([prefix_ids, draft_ids], dim=0).unsqueeze(0)  # [1, L]
        target_logits = self._target_logits(full_ids)  # [L, V]

        # Target LM distributes one step ahead; logits at position j+i-1 predict token j+i
        prefix_len = prefix_ids.shape[0]
        # target_logits[prefix_len - 1 + i] → predicts draft position i (0-indexed)
        draft_target_logits = target_logits[prefix_len - 1: prefix_len - 1 + draft_len]  # [draft_len, V]
        target_log_probs = F.log_softmax(draft_target_logits, dim=-1)  # [draft_len, V]

        # Per-token target log-probs at the drafted tokens
        draft_target_lp = target_log_probs[
            torch.arange(draft_len, device=self.device), draft_ids
        ]  # [draft_len]

        # Standard acceptance rule (Eq. 1): greedy → accept iff p_θ ≥ q_φ^{L2R}
        # For temperature=0 (greedy), acceptance is deterministic:
        #   accept if argmax p_θ == draft token
        # For general temperature, use the ratio test.
        if temperature == 0.0:
            accepted_ids, n_accepted, hit_eos = self._greedy_accept(
                draft_ids, draft_target_logits, prefix_len, prefix_ids, target_logits
            )
        else:
            accepted_ids, n_accepted, hit_eos = self._stochastic_accept(
                draft_ids,
                draft_target_lp,
                drafter_l2r_logprobs,
                target_log_probs,
                temperature,
            )

        return accepted_ids, n_accepted, hit_eos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run target LM and return logits [seq_len, V] (batch dim removed)."""
        out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits[0]  # [seq, V]

    def _greedy_accept(
        self,
        draft_ids: torch.Tensor,      # [draft_len]
        draft_target_logits: torch.Tensor,  # [draft_len, V]
        prefix_len: int,
        prefix_ids: torch.Tensor,
        full_logits: torch.Tensor,    # [full_len, V]
    ) -> Tuple[torch.Tensor, int, bool]:
        """
        Greedy (temperature=0) acceptance: accept draft[i] iff it equals argmax p_θ.
        """
        draft_len = draft_ids.shape[0]
        greedy_tokens = draft_target_logits.argmax(dim=-1)  # [draft_len]

        accepted: list = []
        hit_eos = False

        for i in range(draft_len):
            if greedy_tokens[i].item() == draft_ids[i].item():
                accepted.append(draft_ids[i])
                if draft_ids[i].item() == self.eos_token_id:
                    hit_eos = True
                    break
            else:
                # Rejection: take the target's greedy token as replacement
                accepted.append(greedy_tokens[i])
                if greedy_tokens[i].item() == self.eos_token_id:
                    hit_eos = True
                break

        n_accepted = len(accepted)
        # If we accepted everything, append the next target token (bonus token)
        if n_accepted == draft_len and not hit_eos:
            bonus_logits = full_logits[prefix_len - 1 + draft_len]  # next position
            bonus_tok = bonus_logits.argmax()
            accepted.append(bonus_tok)
            if bonus_tok.item() == self.eos_token_id:
                hit_eos = True

        return torch.stack(accepted), n_accepted, hit_eos

    def _stochastic_accept(
        self,
        draft_ids: torch.Tensor,
        draft_target_lp: torch.Tensor,       # [draft_len]  log p_θ(ŷ_i|...)
        drafter_l2r_logprobs: torch.Tensor,  # [draft_len]  log q_φ^L2R(ŷ_i|...)
        target_log_probs: torch.Tensor,      # [draft_len, V]
        temperature: float,
    ) -> Tuple[torch.Tensor, int, bool]:
        """
        Stochastic acceptance rule (Eq. 1):
            α_{t,i} = min(1, p_θ / q_φ^{L2R})
        On rejection sample from residual (p - q)_+.
        """
        draft_len = draft_ids.shape[0]
        # Convert log-probs to probs for the ratio test
        p_target = draft_target_lp.exp()    # p_θ(ŷ_i | ...)
        q_drafter = drafter_l2r_logprobs.exp()   # q_φ^{L2R}(ŷ_i | ...)

        accepted: list = []
        hit_eos = False

        for i in range(draft_len):
            alpha = min(1.0, (p_target[i] / (q_drafter[i] + 1e-10)).item())
            u = torch.rand(1).item()

            if u <= alpha:
                accepted.append(draft_ids[i])
                if draft_ids[i].item() == self.eos_token_id:
                    hit_eos = True
                    break
            else:
                # Sample replacement from residual distribution (p - q)_+
                p_dist = target_log_probs[i].exp()
                q_dist = torch.zeros_like(p_dist)
                q_dist[draft_ids[i]] = q_drafter[i]

                residual = (p_dist - q_dist).clamp(min=0.0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual = residual / residual_sum
                    replacement = torch.multinomial(residual, 1)[0]
                else:
                    replacement = p_dist.argmax()
                accepted.append(replacement)
                if replacement.item() == self.eos_token_id:
                    hit_eos = True
                break

        return torch.stack(accepted), len(accepted), hit_eos
