"""
N-gram causal language-model proxy used inside CPS scoring (Eq. 8).

The paper fits a 3-gram KenLM on the training split of each dataset.
This module provides:
  - NgramProxy: a pure-Python n-gram LM with add-k smoothing (default, no extra deps)
  - KenLMProxy: thin wrapper around kenlm (optional, requires `pip install kenlm`)

Both expose the same interface:
    proxy.score_sequence(token_ids: List[int]) -> float   (total log-prob, base e)
    proxy.score_token(context: List[int], token_id: int)  -> float  (conditional log-prob)
"""

import math
from collections import defaultdict
from typing import List, Dict, Optional, Sequence


class NgramProxy:
    """
    Simple add-k smoothed n-gram language model trained on a token-id corpus.

    Usage:
        proxy = NgramProxy(n=3)
        proxy.fit(corpus_token_ids)   # List[List[int]]
        lp = proxy.score_token([tok1, tok2], tok3)  # log p(tok3 | tok1, tok2)
    """

    def __init__(self, n: int = 3, k: float = 0.1):
        self.n = n
        self.k = k  # Laplace-style smoothing count
        # ngram_counts[(ctx_tuple)][token] = count
        self._counts: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._ctx_totals: Dict[tuple, int] = defaultdict(int)
        self._vocab_size: int = 0
        self._fitted: bool = False

    def fit(self, corpus: List[List[int]], vocab_size: Optional[int] = None) -> None:
        """Train on a tokenised corpus (list of token-id sequences)."""
        for seq in corpus:
            for i in range(len(seq)):
                for order in range(1, self.n + 1):
                    if i - order + 1 < 0:
                        break
                    ctx = tuple(seq[i - order + 1: i])
                    tok = seq[i]
                    self._counts[ctx][tok] += 1
                    self._ctx_totals[ctx] += 1

        if vocab_size is not None:
            self._vocab_size = vocab_size
        else:
            # Infer vocab size from seen tokens
            seen: set = set()
            for ctx_dict in self._counts.values():
                seen.update(ctx_dict.keys())
            self._vocab_size = max(seen) + 1 if seen else 1

        self._fitted = True

    def score_token(self, context: List[int], token_id: int) -> float:
        """
        Log p(token_id | context[-n+1:]).  Returns log-prob in nats.
        Falls back to lower-order n-grams (stupid back-off style).
        """
        for order in range(min(self.n, len(context) + 1), 0, -1):
            ctx = tuple(context[-order + 1:]) if order > 1 else ()
            total = self._ctx_totals.get(ctx, 0)
            count = self._counts[ctx].get(token_id, 0)
            if total > 0 or order == 1:
                # Add-k smoothed estimate (k=0 with unseen token falls back to lower order)
                smoothed_count = count + self.k
                if smoothed_count <= 0:
                    continue  # unseen with k=0 → try lower order
                smoothed_total = total + self.k * self._vocab_size
                return math.log(smoothed_count / smoothed_total)
        # Uniform fallback
        return -math.log(max(self._vocab_size, 1))

    def score_sequence(self, token_ids: List[int]) -> float:
        """Sum of conditional log-probs for the full sequence."""
        if not token_ids:
            return 0.0
        total = 0.0
        for i, tok in enumerate(token_ids):
            ctx = token_ids[max(0, i - self.n + 1): i]
            total += self.score_token(ctx, tok)
        return total


class UniformProxy:
    """
    Trivial uniform proxy (λ=0 effectively).  Use when no training data is available.
    """

    def __init__(self, vocab_size: int):
        self._lp = -math.log(max(vocab_size, 1))

    def score_token(self, context: List[int], token_id: int) -> float:
        return self._lp

    def score_sequence(self, token_ids: List[int]) -> float:
        return self._lp * len(token_ids)


try:
    import kenlm as _kenlm

    class KenLMProxy:
        """
        Wrapper around a pre-built KenLM model for high-quality n-gram scoring.

        Usage:
            proxy = KenLMProxy("path/to/lm.arpa")
        """

        def __init__(self, model_path: str):
            self._model = _kenlm.Model(model_path)

        def score_token(self, context: List[int], token_id: int) -> float:
            # KenLM operates on string tokens; here we convert int ids to strings
            words = [str(t) for t in context] + [str(token_id)]
            # log10 → natural log
            return self._model.score(" ".join(words)) * math.log(10)

        def score_sequence(self, token_ids: List[int]) -> float:
            words = " ".join(str(t) for t in token_ids)
            return self._model.score(words) * math.log(10)

except ImportError:
    KenLMProxy = None  # type: ignore
