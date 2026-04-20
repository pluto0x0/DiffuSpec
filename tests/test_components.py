"""
Unit tests for DiffuSpec components (no GPU / model weights required).
All tests use tiny synthetic tensors to verify logic, not outputs.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from diffuspec.control.adl import ADLController
from diffuspec.drafting.dlm_drafter import DLMDrafter


# ─── DLM Drafter mock ─────────────────────────────────────────────────────────

VOCAB_SIZE = 100
PEAK_TOKEN = 42   # token the mock model always peaks at
MASK_TOKEN = 0


class _FixedLogitModel:
    """Returns logits strongly peaked at PEAK_TOKEN for every position."""
    def __call__(self, input_ids):
        B, L = input_ids.shape
        logits = torch.full((B, L, VOCAB_SIZE), -10.0)
        logits[:, :, PEAK_TOKEN] = 10.0
        return logits


def _make_drafter(num_refinement_steps=1, target_vocab_size=None) -> DLMDrafter:
    drafter = object.__new__(DLMDrafter)
    drafter.device = "cpu"
    drafter.dtype = torch.float32
    drafter.num_refinement_steps = num_refinement_steps
    drafter.target_vocab_size = target_vocab_size
    drafter.mask_token_id = MASK_TOKEN
    drafter.model = _FixedLogitModel()
    return drafter


class TestDLMDrafter:

    def setup_method(self):
        self.prefix = torch.tensor([1, 2, 3])
        self.draft_len = 5
        self.drafter = _make_drafter()

    def test_draft_output_shapes(self):
        """draft() returns (draft_len,) ids and (draft_len, V) log-probs."""
        draft_ids, log_probs = self.drafter.draft(self.prefix, self.draft_len)
        assert draft_ids.shape == (self.draft_len,)
        assert log_probs.shape == (self.draft_len, VOCAB_SIZE)

    def test_draft_ids_are_argmax_of_log_probs(self):
        """Eq. 5 (S=1): draft_ids[i] = argmax qϕ(yi | context)."""
        draft_ids, log_probs = self.drafter.draft(self.prefix, self.draft_len)
        assert torch.all(draft_ids == log_probs.argmax(dim=-1))

    def test_log_probs_are_valid_log_softmax(self):
        """log_probs must be a proper log-probability distribution at each position."""
        _, log_probs = self.drafter.draft(self.prefix, self.draft_len)
        prob_sums = log_probs.exp().sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(self.draft_len), atol=1e-5)

    def test_draft_ids_peaked_at_mock_token(self):
        """With a fixed-logit mock, every position should draft PEAK_TOKEN."""
        draft_ids, _ = self.drafter.draft(self.prefix, self.draft_len)
        assert torch.all(draft_ids == PEAK_TOKEN)

    def test_vocab_masking_blocks_out_of_range_tokens(self):
        """Tokens with id >= target_vocab_size must never be drafted (vocab mismatch fix)."""
        # PEAK_TOKEN=42; set target_vocab_size=30 so it's masked out
        drafter = _make_drafter(target_vocab_size=30)
        draft_ids, _ = drafter.draft(self.prefix, self.draft_len)
        assert torch.all(draft_ids < 30)

    def test_draft_multistep_same_result(self):
        """With a fixed-logit mock, S=2 refinement must agree with S=1."""
        drafter_s2 = _make_drafter(num_refinement_steps=2)
        ids1, lp1 = self.drafter.draft(self.prefix, self.draft_len)
        ids2, lp2 = drafter_s2.draft(self.prefix, self.draft_len)
        assert torch.all(ids1 == ids2)

    # -- compute_l2r_logprobs (Eq. 6) -----------------------------------------

    def test_l2r_logprobs_shape(self):
        """Eq. 6: returns a scalar log-prob per draft position — shape (draft_len,)."""
        draft_ids, _ = self.drafter.draft(self.prefix, self.draft_len)
        l2r = self.drafter.compute_l2r_logprobs(self.prefix, draft_ids)
        assert l2r.shape == (self.draft_len,)

    def test_l2r_logprobs_are_non_positive(self):
        """Log-probabilities must be ≤ 0."""
        draft_ids, _ = self.drafter.draft(self.prefix, self.draft_len)
        l2r = self.drafter.compute_l2r_logprobs(self.prefix, draft_ids)
        assert torch.all(l2r <= 1e-6)

    def test_l2r_logprobs_score_drafted_token(self):
        """
        Eq. 6: l2r[i] = log qϕ(draft_ids[i] | prefix ∘ draft[0:i] ∘ [MASK]^{k-i}).
        With the fixed-logit mock, draft_ids[i] == PEAK_TOKEN and the mock always
        assigns high probability to it, so each l2r score should be close to 0.
        """
        draft_ids, _ = self.drafter.draft(self.prefix, self.draft_len)
        l2r = self.drafter.compute_l2r_logprobs(self.prefix, draft_ids)
        # log-prob of the clearly dominant token ≈ log(softmax(10 vs -10)) ≈ 0
        assert torch.all(l2r > -1.0)

    def test_l2r_lower_for_non_peak_token(self):
        """l2r of a non-peak token should be lower than that of the peak token."""
        draft_ids_peak = torch.full((self.draft_len,), PEAK_TOKEN, dtype=torch.long)
        draft_ids_off = torch.full((self.draft_len,), PEAK_TOKEN + 1, dtype=torch.long)

        l2r_peak = self.drafter.compute_l2r_logprobs(self.prefix, draft_ids_peak)
        l2r_off = self.drafter.compute_l2r_logprobs(self.prefix, draft_ids_off)
        assert torch.all(l2r_peak > l2r_off)
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
