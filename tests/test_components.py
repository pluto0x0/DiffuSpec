"""
Unit tests for DiffuSpec components (no GPU / model weights required).
All tests use tiny synthetic tensors to verify logic, not outputs.
"""

import math
import pytest
import torch

from diffuspec.control.adl import ADLController
from diffuspec.search.cps import CausalConsistencyPathSearch
from diffuspec.proxy.ngram_proxy import NgramProxy, UniformProxy


# ─── ADL Controller ───────────────────────────────────────────────────────────

class TestADLController:

    def test_initial_k_is_k_max(self):
        adl = ADLController(k_min=20, k_max=30, delta=10, rho=0.5)
        assert adl.next_k == 30

    def test_reset_restores_initial_state(self):
        adl = ADLController(k_min=20, k_max=30, delta=10, rho=0.5)
        adl.update(l_gen=25, l_acc=25)
        adl.reset()
        assert adl.next_k == 30
        assert adl.ema_gen == 0.0
        assert adl.ema_acc == 0.0

    def test_ema_update_formula(self):
        """Eq. 10: L̄^gen = (1-ρ)*L̄^gen_prev + ρ*L^gen"""
        adl = ADLController(k_min=20, k_max=30, delta=10, rho=0.5)
        adl.update(l_gen=20, l_acc=20)
        assert abs(adl.ema_gen - 10.0) < 1e-6   # 0.5 * 0 + 0.5 * 20 = 10
        assert abs(adl.ema_acc - 10.0) < 1e-6

    def test_k_clips_to_bounds(self):
        adl = ADLController(k_min=20, k_max=30, delta=10, rho=0.5)
        # Very large l_gen should produce k = k_max
        for _ in range(100):
            adl.update(l_gen=200, l_acc=200)
        assert adl.next_k == 30

        adl.reset()
        # Very small l_gen should produce k = k_min
        for _ in range(100):
            adl.update(l_gen=1, l_acc=0)
        assert adl.next_k == 20

    def test_k_grows_only_when_acceptance_keeps_up(self):
        """k grows by delta iff ema_acc >= ema_gen."""
        adl = ADLController(k_min=5, k_max=100, delta=10, rho=1.0)  # rho=1 → no smoothing
        adl.update(l_gen=15, l_acc=15)  # acc == gen → should grow
        k_after_good = adl.next_k

        adl.reset()
        adl.update(l_gen=15, l_acc=5)   # acc < gen → no growth
        k_after_bad = adl.next_k

        assert k_after_good > k_after_bad


# ─── CPS ──────────────────────────────────────────────────────────────────────

EOS_ID = 2


def make_uniform_logprobs(vocab_size: int, draft_len: int) -> torch.Tensor:
    """Uniform log-probs across vocab."""
    lp = torch.full((draft_len, vocab_size), -math.log(vocab_size))
    return lp


def make_peaked_logprobs(vocab_size: int, draft_len: int, peak_token: int) -> torch.Tensor:
    """Log-probs strongly peaked at peak_token."""
    lp = torch.full((draft_len, vocab_size), -100.0)
    lp[:, peak_token] = 0.0  # log(1) = 0
    return lp


class TestCPS:

    def setup_method(self):
        self.cps = CausalConsistencyPathSearch(
            eos_token_id=EOS_ID,
            M_max=5,
            tau=0.8,
            beam_size=3,
            lam=1.0,  # use DLM score only (ignore proxy)
        )
        self.proxy = UniformProxy(vocab_size=100)

    def test_peaked_draft_returns_top_token(self):
        """When log-probs are strongly peaked, path should be the argmax token."""
        peak_token = 42
        lp = make_peaked_logprobs(vocab_size=100, draft_len=5, peak_token=peak_token)
        path, path_len = self.cps.search(prefix_ids=[1, 2, 3], dlm_log_probs=lp, causal_proxy=self.proxy)
        assert path[0] == peak_token

    def test_eos_terminates_path_early(self):
        """When EOS is the top token at position 2, path should stop there."""
        vocab_size = 100
        lp = make_uniform_logprobs(vocab_size, draft_len=10)
        # Force EOS to position 2 (0-indexed)
        lp[2, :] = -100.0
        lp[2, EOS_ID] = 0.0

        path, path_len = self.cps.search(prefix_ids=[1], dlm_log_probs=lp, causal_proxy=self.proxy)
        # Path must stop at or before the EOS position
        assert path_len <= 3   # at most up to and including the EOS position
        if EOS_ID in path:
            eos_idx = path.index(EOS_ID)
            assert eos_idx == len(path) - 1  # EOS must be last

    def test_path_length_within_draft_len(self):
        lp = make_uniform_logprobs(vocab_size=50, draft_len=8)
        # Remove EOS from all positions to ensure full path
        lp[:, EOS_ID] = -100.0
        path, path_len = self.cps.search(prefix_ids=[], dlm_log_probs=lp, causal_proxy=self.proxy)
        assert 1 <= path_len <= 8

    def test_eos_always_in_candidates(self):
        """Even when EOS has very low probability it must appear in candidates."""
        vocab_size = 100
        lp = torch.full((5, vocab_size), -1.0)
        lp[:, EOS_ID] = -50.0  # very low EOS probability

        # Build candidate set manually
        candidate_sets = self.cps._build_candidate_sets(lp)
        for cs in candidate_sets:
            tokens = [t for t, _ in cs]
            assert EOS_ID in tokens, "EOS must always be in the candidate set"


# ─── N-gram Proxy ─────────────────────────────────────────────────────────────

class TestNgramProxy:

    def test_fit_and_score(self):
        proxy = NgramProxy(n=2, k=0.0)  # no smoothing for deterministic test
        corpus = [[1, 2, 3, 1, 2, 3]]
        proxy.fit(corpus, vocab_size=10)
        # p(3 | 2) should be high (always followed by 3 in corpus)
        lp = proxy.score_token([2], 3)
        assert lp > proxy.score_token([2], 5)   # 5 never follows 2

    def test_score_sequence_sum(self):
        proxy = NgramProxy(n=2, k=0.1)
        corpus = [[1, 2, 3, 4, 5]]
        proxy.fit(corpus, vocab_size=10)
        seq = [1, 2, 3]
        total = proxy.score_sequence(seq)
        # Should equal sum of individual conditional log-probs
        manual = (
            proxy.score_token([], 1) +
            proxy.score_token([1], 2) +
            proxy.score_token([2], 3)
        )
        assert abs(total - manual) < 1e-9

    def test_uniform_proxy_constant(self):
        proxy = UniformProxy(vocab_size=1000)
        lp = proxy.score_token([1, 2, 3], 42)
        assert abs(lp - (-math.log(1000))) < 1e-9
        lp2 = proxy.score_token([5], 999)
        assert abs(lp - lp2) < 1e-9  # uniform → all same
