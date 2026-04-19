"""
DiffuSpec Engine — paper Algorithm 1.

Four-stage pipeline per speculative step:
  (1) Draft    : DLM produces length-k_t block + per-position candidates
  (2) CPS      : beam search over pruned lattice → left-to-right path ŷ
  (3) Verify   : target LM accepts prefix; advance context
  (4) ADL      : update EMAs, set k_{t+1}

Usage:
    engine = DiffuSpec.from_config(config)
    output_ids = engine.generate(prompt_ids)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DiffuSpecConfig, DraftingConfig
from .drafting.dlm_drafter import DLMDrafter
from .search.cps import CausalConsistencyPathSearch
from .control.adl import ADLController
from .verification.verifier import ARVerifier
from .proxy.ngram_proxy import NgramProxy, UniformProxy


class DiffuSpec:
    """
    Training-free speculative decoding with a diffusion LM drafter.

    Components:
        drafter  : DLMDrafter        — masked-diffusion forward pass
        cps      : CausalConsistencyPathSearch — lattice path selection
        adl      : ADLController     — adaptive draft-length
        verifier : ARVerifier        — AR target-model parallel verification
        proxy    : NgramProxy / UniformProxy — causal n-gram scoring signal
    """

    def __init__(
        self,
        drafter: DLMDrafter,
        cps: CausalConsistencyPathSearch,
        adl: ADLController,
        verifier: ARVerifier,
        proxy,
        config: DiffuSpecConfig,
    ) -> None:
        self.drafter = drafter
        self.cps = cps
        self.adl = adl
        self.verifier = verifier
        self.proxy = proxy
        self.config = config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: DiffuSpecConfig,
        target_model=None,
        target_tokenizer=None,
        proxy=None,
        device: str = "cuda",
    ) -> "DiffuSpec":
        """
        Build a DiffuSpec instance from a DiffuSpecConfig.

        If target_model / target_tokenizer are not supplied they are loaded
        from config.target_model_name (requires enough GPU VRAM).
        """
        import torch

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(config.drafting.dtype, torch.bfloat16)

        # Target LM
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

        # Drafter — restrict to target vocab so OOB token IDs never reach the AR model
        drafter = DLMDrafter(
            model_name=config.drafting.model_name,
            device=device,
            dtype=dtype,
            num_refinement_steps=config.drafting.num_refinement_steps,
            target_vocab_size=target_tokenizer.vocab_size,
        )

        eos_token_id = (
            config.eos_token_id
            if config.eos_token_id is not None
            else target_tokenizer.eos_token_id
        )

        # CPS
        cps_module = CausalConsistencyPathSearch(
            eos_token_id=eos_token_id,
            M_max=config.cps.M_max,
            tau=config.cps.tau,
            beam_size=config.cps.beam_size,
            lam=config.cps.lam,
        )

        # ADL controller
        adl_ctrl = ADLController(
            k_min=config.adl.k_min,
            k_max=config.adl.k_max,
            delta=config.adl.delta,
            rho=config.adl.rho,
        )

        # Verifier
        verifier = ARVerifier(
            model=target_model,
            tokenizer=target_tokenizer,
            device=device,
        )

        # Causal proxy: use UniformProxy if none provided
        if proxy is None:
            vocab_size = target_tokenizer.vocab_size
            proxy = UniformProxy(vocab_size)

        return cls(
            drafter=drafter,
            cps=cps_module,
            adl=adl_ctrl,
            verifier=verifier,
            proxy=proxy,
            config=config,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prefix_ids: torch.Tensor,        # [prefix_len]  — tokenised prompt
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate tokens via the DiffuSpec 4-stage loop (Algorithm 1).

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
        # Use the verifier's full stop-token set (includes <|im_end|> etc.)
        stop_ids = self.verifier._stop_ids or ({eos_id} if eos_id is not None else set())

        self.adl.reset()
        context = prefix_ids.clone()
        generated: List[int] = []

        stats = {
            "n_steps": 0,
            "total_accepted": 0,
            "accepted_lengths": [],
            "draft_lengths": [],
            "wall_time_s": 0.0,
        }

        t_start = time.perf_counter()

        while len(generated) < max_new_tokens:
            k_t = self.adl.next_k

            # ── Stage 1: Draft ──────────────────────────────────────────
            draft_ids, draft_log_probs = self.drafter.draft(
                prefix_ids=context,
                draft_len=k_t,
                top_m=self.config.cps.M_max,
            )

            # ── Stage 2: CPS ─────────────────────────────────────────────
            # Convert to list for proxy scoring (which operates on token IDs)
            prefix_list = context.tolist()
            path_tokens, path_len = self.cps.search(
                prefix_ids=prefix_list,
                dlm_log_probs=draft_log_probs,
                causal_proxy=self.proxy,
            )

            if path_len == 0:
                # Degenerate: nothing drafted; run target LM greedily for one step
                next_id = self._greedy_step(context)
                generated.append(next_id)
                context = torch.cat([context, torch.tensor([next_id], device=context.device)])
                if next_id in stop_ids:
                    break
                self.adl.update(l_gen=1, l_acc=1)
                continue

            path_tensor = torch.tensor(path_tokens, dtype=torch.long, device=context.device)

            # ── Stage 3: L2R log-probs + Parallel verification ──────────
            # L2R log-probs are only used for stochastic acceptance (temperature > 0).
            # Skipping the O(k²) Dream batch call in greedy mode saves significant time.
            if self.config.temperature > 0.0:
                l2r_logprobs = self.drafter.compute_l2r_logprobs(context, path_tensor)
            else:
                l2r_logprobs = torch.zeros(path_len, device=context.device)

            accepted_ids, n_acc, hit_eos = self.verifier.verify(
                prefix_ids=context,
                draft_ids=path_tensor,
                drafter_l2r_logprobs=l2r_logprobs,
                temperature=self.config.temperature,
            )

            # Advance context
            accepted_list = accepted_ids.tolist()
            generated.extend(accepted_list)
            context = torch.cat([context, accepted_ids.to(context.device)])

            # ── Stage 4: ADL ─────────────────────────────────────────────
            # L^gen: EOS-aware generation length from raw DLM draft (Eq. 9)
            l_gen = self._compute_l_gen(draft_ids, stop_ids, k_t)
            self.adl.update(l_gen=l_gen, l_acc=n_acc)

            stats["n_steps"] += 1
            stats["total_accepted"] += n_acc
            stats["accepted_lengths"].append(n_acc)
            stats["draft_lengths"].append(k_t)

            if verbose:
                print(
                    f"step {stats['n_steps']:4d} | k={k_t:3d} | "
                    f"path_len={path_len} | acc={n_acc} | "
                    f"ema_gen={self.adl.ema_gen:.1f} | ema_acc={self.adl.ema_acc:.1f}"
                )

            if hit_eos or bool(stop_ids & set(accepted_list)):
                break

            if len(generated) >= max_new_tokens:
                break

        stats["wall_time_s"] = time.perf_counter() - t_start
        if stats["n_steps"] > 0:
            stats["mean_accepted"] = stats["total_accepted"] / stats["n_steps"]

        return torch.tensor(generated, dtype=torch.long), stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_l_gen(draft_ids: torch.Tensor, stop_ids: set, k_t: int) -> int:
        """
        EOS-aware generation length L^gen_t = min(s_t − 1, k_t) (Eq. 9).
        s_t is the index (1-based) of the first stop token in the raw draft.
        If no stop token: L^gen = k_t.
        """
        for i, tok in enumerate(draft_ids.tolist()):
            if tok in stop_ids:
                return min(i, k_t)  # s_t-1 where s_t = i+1
        return k_t

    @torch.no_grad()
    def _greedy_step(self, context: torch.Tensor) -> int:
        """Single greedy step from the target LM (fallback)."""
        out = self.verifier.model(
            input_ids=context.unsqueeze(0).to(self.verifier.device),
            use_cache=False,
        )
        logits = out.logits[0, -1, :]
        return logits.argmax().item()
