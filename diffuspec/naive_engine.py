"""
NaiveSpec — naive speculative decoding baseline using a Diffusion LM drafter.

Pipeline per step (no CPS, no ADL):
  (1) Draft  : DLM argmax → fixed-length draft block
  (2) Verify : AR target LM accepts longest matching prefix

This serves as the ablation baseline for DiffuSpec: same drafter and verifier,
but without CPS path reranking or ADL draft-length control.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import NaiveSpecConfig, DraftingConfig
from .drafting.dlm_drafter import DLMDrafter
from .verification.verifier import ARVerifier


class NaiveSpec:
    """
    Naive speculative decoding with a DLM drafter.

    Differences from DiffuSpec:
      - No CPS: draft tokens are the DLM argmax, not a beam-searched causal path.
      - No ADL: draft length is fixed to config.draft_len every step.
      - No proxy/KenLM: these are only used inside CPS.
    """

    def __init__(
        self,
        drafter: DLMDrafter,
        verifier: ARVerifier,
        config: NaiveSpecConfig,
    ) -> None:
        self.drafter = drafter
        self.verifier = verifier
        self.config = config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: NaiveSpecConfig,
        target_model=None,
        target_tokenizer=None,
        device: str = "cuda",
    ) -> "NaiveSpec":
        """
        Build a NaiveSpec instance from a NaiveSpecConfig.

        If target_model / target_tokenizer are not supplied they are loaded
        from config.target_model_name.
        """
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(config.drafting.dtype, torch.bfloat16)

        if target_model is None or target_tokenizer is None:
            target_tokenizer = AutoTokenizer.from_pretrained(
                config.target_model_name, trust_remote_code=True
            )
            target_model = AutoModelForCausalLM.from_pretrained(
                config.target_model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            target_model.eval()

        drafter = DLMDrafter(
            model_name=config.drafting.model_name,
            device=device,
            dtype=dtype,
            num_refinement_steps=config.drafting.num_refinement_steps,
            target_vocab_size=target_tokenizer.vocab_size,
        )

        verifier = ARVerifier(
            model=target_model,
            tokenizer=target_tokenizer,
            device=device,
        )

        return cls(drafter=drafter, verifier=verifier, config=config)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate tokens via the naive DLM speculative decoding loop.

        Args:
            prefix_ids     : 1-D LongTensor of prompt token IDs.
            max_new_tokens : generation budget (overrides config if given).
            verbose        : print per-step diagnostics.

        Returns:
            generated_ids : all newly generated token IDs (excluding prompt).
            stats         : dict with timing and acceptance statistics.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        eos_id = (
            self.config.eos_token_id
            if self.config.eos_token_id is not None
            else self.verifier.eos_token_id
        )
        stop_ids = self.verifier._stop_ids or ({eos_id} if eos_id is not None else set())

        context = prefix_ids.clone()
        generated: List[int] = []
        _step_rows: List[dict] = []

        stats = {
            "n_steps": 0,
            "total_accepted": 0,
            "accepted_lengths": [],
            "draft_lengths": [],
            "wall_time_s": 0.0,
        }

        t_start = time.perf_counter()

        _sync = (
            torch.cuda.synchronize
            if torch.cuda.is_available()
            else (lambda: None)
        )

        k = self.config.draft_len

        while len(generated) < max_new_tokens:
            # ── Stage 1: Draft ───────────────────────────────────────────
            _sync(); t_draft = time.perf_counter()
            draft_ids, _log_probs = self.drafter.draft(
                prefix_ids=context,
                draft_len=k,
                top_m=k,  # top_m unused without CPS; pass k as a no-op
            )
            _sync(); t_draft = time.perf_counter() - t_draft

            # ── Stage 2: Verify ──────────────────────────────────────────
            # For greedy (temperature=0) the acceptance check is argmax equality,
            # so drafter_l2r_logprobs are not used — pass zeros to skip the O(k²)
            # compute_l2r_logprobs batch call.
            # For stochastic acceptance the ratio test requires L2R log-probs.
            _sync(); t_verify = time.perf_counter()
            if self.config.temperature > 0.0:
                l2r_logprobs = self.drafter.compute_l2r_logprobs(context, draft_ids)
            else:
                l2r_logprobs = torch.zeros(k, device=context.device)

            accepted_ids, n_acc, hit_eos = self.verifier.verify(
                prefix_ids=context,
                draft_ids=draft_ids,
                drafter_l2r_logprobs=l2r_logprobs,
                temperature=self.config.temperature,
            )
            _sync(); t_verify = time.perf_counter() - t_verify

            accepted_list = accepted_ids.tolist()
            generated.extend(accepted_list)
            context = torch.cat([context, accepted_ids.to(context.device)])

            stats["n_steps"] += 1
            stats["total_accepted"] += n_acc
            stats["accepted_lengths"].append(n_acc)
            stats["draft_lengths"].append(k)
            stats.setdefault("draft_times_s", []).append(t_draft)
            stats.setdefault("verify_times_s", []).append(t_verify)

            if verbose:
                tok = self.verifier.tokenizer
                context_before = generated[: len(generated) - len(accepted_list)]
                context_text = tok.decode(context_before, skip_special_tokens=False)
                if len(context_text) > 40:
                    context_text = "…" + context_text[-40:]
                draft_text = tok.decode(draft_ids.tolist(), skip_special_tokens=False)
                accepted_text = tok.decode(accepted_list, skip_special_tokens=False)
                _step_rows.append({
                    "step": stats["n_steps"],
                    "k": k,
                    "acc": f"{n_acc}/{k}",
                    "context": context_text,
                    "draft": draft_text,
                    "accepted": accepted_text,
                    "t_draft_ms": round(t_draft * 1e3),
                    "t_verify_ms": round(t_verify * 1e3),
                })

            if hit_eos or bool(stop_ids & set(accepted_list)):
                break

            if len(generated) >= max_new_tokens:
                break

        stats["wall_time_s"] = time.perf_counter() - t_start
        if stats["n_steps"] > 0:
            stats["mean_accepted"] = stats["total_accepted"] / stats["n_steps"]

        if verbose and _step_rows:
            import pandas as pd
            df = pd.DataFrame(_step_rows)
            print("\n── Per-step trace (NaiveSpec) ──")
            print(df.to_string(index=False))
            print()

        stats["step_rows"] = _step_rows
        return torch.tensor(generated, dtype=torch.long), stats
