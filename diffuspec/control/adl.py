"""
Adaptive Draft-Length (ADL) controller — paper section 4.3.

Two EMA signals track the drafting quality in real time:
  L̄^gen : EOS-aware generation length (how much the DLM is willing to produce)
  L̄^acc : accepted prefix length     (how much the verifier actually accepts)

The next draft length k_{t+1} grows by δ only when both signals agree
(acceptance keeps up with generation), and clips to [k_min, k_max] (Eq. 11).
"""

from __future__ import annotations


class ADLController:
    """
    Online adaptive controller for the draft-length k_t.

    Paper hyperparameters (Appendix A):
        k_min = 20,  k_max = 30,  delta = 10,  rho = 0.5

    Signals (Eq. 10):
        L̄^gen_t = (1−ρ) L̄^gen_{t-1} + ρ L^gen_t
        L̄^acc_t = (1−ρ) L̄^acc_{t-1} + ρ L^acc_t

    Update rule (Eq. 11):
        k_{t+1} = clip( ⌊L̄^gen_t⌋ + δ · 1{L̄^acc_t ≥ L̄^gen_t},  k_min, k_max )
    """

    def __init__(
        self,
        k_min: int = 20,
        k_max: int = 30,
        delta: int = 10,
        rho: float = 0.5,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.delta = delta
        self.rho = rho

        # EMA state (initialised to 0 per Algorithm 1)
        self._ema_gen: float = 0.0
        self._ema_acc: float = 0.0

        # Current draft length; paper initialises k_1 = k_max
        self.k: int = k_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, l_gen: int, l_acc: int) -> None:
        """
        Update EMA signals after one speculative step and advance k.

        Args:
            l_gen : L^gen_t — EOS-aware generation length from the raw DLM proposal.
                    Computed as min(first_eos_index − 1, k_t) before CPS.
            l_acc : L^acc_t — number of tokens accepted by the verifier.
        """
        # Eq. 10: exponential moving averages
        self._ema_gen = (1.0 - self.rho) * self._ema_gen + self.rho * l_gen
        self._ema_acc = (1.0 - self.rho) * self._ema_acc + self.rho * l_acc

        # Eq. 11: grow only when acceptance keeps pace with generation
        growth = self.delta if self._ema_acc >= self._ema_gen else 0
        raw_next = int(self._ema_gen) + growth
        self.k = max(self.k_min, min(self.k_max, raw_next))

    @property
    def next_k(self) -> int:
        """Draft length to use at the next speculative step."""
        return self.k

    @property
    def ema_gen(self) -> float:
        return self._ema_gen

    @property
    def ema_acc(self) -> float:
        return self._ema_acc

    def reset(self) -> None:
        """Reset controller state (call between independent generation requests)."""
        self._ema_gen = 0.0
        self._ema_acc = 0.0
        self.k = self.k_max
