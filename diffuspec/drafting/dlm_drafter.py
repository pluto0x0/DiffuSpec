"""
DLM Drafter — wraps a pretrained discrete diffusion language model (Dream-7B).

Responsibilities:
  1. draft()            : one forward pass → draft tokens + per-position log-probs
  2. compute_l2r_logprobs(): batched L2R conditional log-probs used in acceptance ratio (Eq. 6)

Paper section 4.1:
  - Inference uses S=1 refinement step by default (ablated in Sec. 5).
  - q_φ^{L2R}(v | x_{1:j+i-1}) is computed by masking all future in-block positions (Eq. 6).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


def _patch_rope_init_functions() -> None:
    """
    Inject 'default' into transformers ROPE_INIT_FUNCTIONS if missing.

    Dream-7B's modeling code uses rope_type='default' (standard RoPE, no scaling),
    but transformers ≥5.x removed that key.  We add it back as the identity init:
        inv_freq = 1 / base^(2i/dim),  attention_scaling = 1.0
    """
    try:
        from transformers import modeling_rope_utils
        if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
                if config is not None:
                    base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
                    head_dim = getattr(config, "head_dim", None)
                    if head_dim is None:
                        head_dim = config.hidden_size // config.num_attention_heads
                    partial = getattr(config, "partial_rotary_factor", 1.0)
                    dim = int(head_dim * partial)
                else:
                    base = kwargs.get("base", 10000.0)
                    dim = kwargs.get("dim", 64)

                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
                )
                return inv_freq, 1.0

            modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass  # If patching fails, the model load will surface the real error


_patch_rope_init_functions()


class DLMDrafter:
    """
    Wraps a masked-diffusion LM (e.g., Dream-7B) for speculative drafting.

    The DLM denoises a fully-masked draft block in S refinement steps.
    For S=1 (paper default), a single forward pass yields all draft tokens.
    """

    def __init__(
        self,
        model_name: str = "dream-org/dream-v0-instruct-7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_refinement_steps: int = 1,
        target_vocab_size: Optional[int] = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_refinement_steps = num_refinement_steps
        # Tokens with IDs >= target_vocab_size are Dream-specific and unknown to the
        # target AR model; mask them out so they are never drafted.
        self.target_vocab_size = target_vocab_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Dream-7B uses a masked-LM interface; fall back to AutoModel if needed.
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        self.model.eval()

        # Mask token used to initialise the draft block
        self.mask_token_id: int = (
            self.tokenizer.mask_token_id
            if self.tokenizer.mask_token_id is not None
            else self.tokenizer.unk_token_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def draft(
        self,
        prefix_ids: torch.Tensor,   # [prefix_len]  long tensor on any device
        draft_len: int,
        top_m: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce a draft block via iterative DLM refinement (S steps).

        Args:
            prefix_ids : prefix token IDs, shape [prefix_len].
            draft_len  : number of tokens to draft (k_t).
            top_m      : per-position candidate cap M_max (used externally by CPS).

        Returns:
            draft_ids  : [draft_len]         – argmax denoised token IDs.
            log_probs  : [draft_len, V]      – log-softmax over vocab at each position.
        """
        prefix_ids = prefix_ids.to(self.device)
        mask_block = torch.full(
            (draft_len,), self.mask_token_id, dtype=torch.long, device=self.device
        )
        input_ids = torch.cat([prefix_ids, mask_block], dim=0).unsqueeze(0)  # [1, L]

        prefix_len = prefix_ids.shape[0]

        # Iterative refinement: S steps (paper uses S=1)
        current_ids = input_ids.clone()
        log_probs: Optional[torch.Tensor] = None

        for step in range(self.num_refinement_steps):
            logits = self._forward(current_ids)               # [1, L, V]
            draft_logits = logits[0, prefix_len:, :]          # [draft_len, V]
            if self.target_vocab_size is not None and draft_logits.shape[-1] > self.target_vocab_size:
                draft_logits = draft_logits.clone()
                draft_logits[:, self.target_vocab_size:] = float("-inf")
            log_probs = F.log_softmax(draft_logits, dim=-1)   # [draft_len, V]
            draft_ids = log_probs.argmax(dim=-1)              # [draft_len]

            # For multi-step refinement: update all masked positions
            # (top-K by confidence; for S=1 this branch never executes)
            if step < self.num_refinement_steps - 1:
                current_ids = current_ids.clone()
                current_ids[0, prefix_len:] = draft_ids

        assert log_probs is not None
        return draft_ids, log_probs

    @torch.no_grad()
    def compute_l2r_logprobs(
        self,
        prefix_ids: torch.Tensor,   # [prefix_len]
        draft_ids: torch.Tensor,    # [draft_len]
    ) -> torch.Tensor:
        """
        Compute per-position left-to-right conditional log-probs (Eq. 6).

        For position i in [0, draft_len):
            input = prefix || draft[0:i] || [MASK]^{draft_len - i}
            logprob = log q_φ(draft[i] | input)[draft[i]]

        This is batched: all k sequences are run in parallel.

        Returns:
            l2r_logprobs : [draft_len]  – scalar log-prob per draft position.
        """
        prefix_ids = prefix_ids.to(self.device)
        draft_ids = draft_ids.to(self.device)
        draft_len = draft_ids.shape[0]
        prefix_len = prefix_ids.shape[0]
        total_len = prefix_len + draft_len

        # Build batch: row i → prefix || draft[0:i] || [MASK] * (draft_len - i)
        batch = torch.full(
            (draft_len, total_len), self.mask_token_id, dtype=torch.long, device=self.device
        )
        batch[:, :prefix_len] = prefix_ids.unsqueeze(0)  # broadcast prefix

        for i in range(draft_len):
            if i > 0:
                batch[i, prefix_len: prefix_len + i] = draft_ids[:i]
            # position prefix_len + i stays [MASK] → the position being scored

        logits = self._forward(batch)  # [draft_len, total_len, V]

        # Extract log-prob of the actual draft token at position prefix_len + i
        scores = torch.zeros(draft_len, device=self.device)
        for i in range(draft_len):
            pos_logits = logits[i, prefix_len + i, :]           # [V]
            lp = F.log_softmax(pos_logits, dim=-1)
            scores[i] = lp[draft_ids[i]]

        return scores  # [draft_len]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single model forward pass.  Returns logits [batch, seq, V].
        Handles different output formats (MaskedLMOutput vs plain tensor).
        """
        output = self.model(input_ids=input_ids)
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, torch.Tensor):
            return output
        # Some models return a tuple; first element is logits
        return output[0]
