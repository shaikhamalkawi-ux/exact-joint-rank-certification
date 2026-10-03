"""Sampling detectability beyond an EJRC first-tie boundary.

The functions here are conditional on fixed scores, fixed reference weights, and
independent uniform multiplicative weight perturbations q_j ~ U(1-r, 1+r).
They do not define physical failure probabilities or fuzzy-membership
probabilities.

The one-draw reversal probability is evaluated by the classical weighted
hypercube-slice inclusion-exclusion formula, specialized to the EJRC
pairwise transformed margin.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import mpmath as mp
import numpy as np


@dataclass(frozen=True)
class DetectabilityResult:
    probability: float
    r_star: float
    active_dimension: int
    expected_in_n: float | None = None
    miss_probability_n: float | None = None
    n_for_target_detection: int | None = None


def _normalized_weights(weights: Sequence[float]) -> list[mp.mpf]:
    w = [mp.mpf(str(x)) for x in weights]
    if any(x < 0 for x in w):
        raise ValueError("weights must be nonnegative")
    s = mp.fsum(w)
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return [x / s for x in w]


def pairwise_sampling_coefficients(
    matrix: np.ndarray,
    weights: Sequence[float],
    leader: int,
    competitor: int,
    *,
    dps: int = 80,
) -> tuple[list[mp.mpf], mp.mpf, mp.mpf, mp.mpf]:
    """Return active a_j coefficients, M, B, and r* for fixed-score weight sampling."""
    mp.mp.dps = dps
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if np.any(X < 0) or np.any(X >= 1):
        raise ValueError("scores must lie in [0, 1)")
    w = _normalized_weights(weights)
    row_i = [mp.mpf(str(x)) for x in X[leader]]
    row_k = [mp.mpf(str(x)) for x in X[competitor]]
    a = [
        v * (mp.atanh(xi) - mp.atanh(xk))
        for v, xi, xk in zip(w, row_i, row_k)
    ]
    M = mp.fsum(a)
    active = [x for x in a if x != 0]
    B = mp.fsum(abs(x) for x in active)
    if M <= 0 or B <= 0:
        raise ValueError("leader must have a positive nominal transformed margin")
    return active, M, B, M / B


def weighted_cube_slice_cdf(widths: Sequence[mp.mpf], threshold: mp.mpf) -> mp.mpf:
    """CDF of sum U(0,width_j), using inclusion-exclusion."""
    widths = [mp.mpf(x) for x in widths]
    if not widths or any(x <= 0 for x in widths):
        raise ValueError("widths must be nonempty and strictly positive")
    threshold = mp.mpf(threshold)
    total = mp.fsum(widths)
    if threshold <= 0:
        return mp.mpf("0")
    if threshold >= total:
        return mp.mpf("1")
    if threshold > total / 2:
        return 1 - weighted_cube_slice_cdf(widths, total - threshold)
    d = len(widths)
    terms = []
    for bits in itertools.product((0, 1), repeat=d):
        shifted = threshold - mp.fsum(
            width for bit, width in zip(bits, widths) if bit
        )
        if shifted > 0:
            terms.append(((-1) ** sum(bits)) * shifted**d)
    return mp.fsum(terms) / (mp.factorial(d) * mp.fprod(widths))


def reversal_probability_uniform(
    matrix: np.ndarray,
    weights: Sequence[float],
    leader: int,
    competitor: int,
    r: float,
    *,
    dps: int = 80,
) -> DetectabilityResult:
    """Exact one-draw reversal probability for q_j iid U(1-r,1+r)."""
    if not 0 < r < 1:
        raise ValueError("r must lie in (0,1)")
    mp.mp.dps = dps
    active, M, B, r_star = pairwise_sampling_coefficients(
        matrix, weights, leader, competitor, dps=dps
    )
    rmp = mp.mpf(str(r))
    if rmp <= r_star:
        p = mp.mpf("0")
    else:
        h = rmp * B - M
        widths = [2 * rmp * abs(x) for x in active]
        p = weighted_cube_slice_cdf(widths, h)
    return DetectabilityResult(
        probability=float(p),
        r_star=float(r_star),
        active_dimension=len(active),
    )


def detection_sample_size(probability: float, beta: float = 0.95) -> int:
    """Minimum iid draws for at least beta probability of >=1 reversal."""
    p = float(probability)
    if not 0 < beta < 1:
        raise ValueError("beta must lie in (0,1)")
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1
    return math.ceil(math.log1p(-beta) / math.log1p(-p))


def sampling_detectability(
    matrix: np.ndarray,
    weights: Sequence[float],
    leader: int,
    competitor: int,
    r: float,
    *,
    n_draws: int | None = None,
    beta: float = 0.95,
    dps: int = 80,
) -> DetectabilityResult:
    """Return probability plus optional N-draw and target-detection summaries."""
    base = reversal_probability_uniform(
        matrix, weights, leader, competitor, r, dps=dps
    )
    p = base.probability
    expected = miss = None
    if n_draws is not None:
        if n_draws < 0:
            raise ValueError("n_draws must be nonnegative")
        expected = n_draws * p
        miss = math.exp(n_draws * math.log1p(-p)) if p < 1 else 0.0
    n_target = detection_sample_size(p, beta=beta)
    return DetectabilityResult(
        probability=p,
        r_star=base.r_star,
        active_dimension=base.active_dimension,
        expected_in_n=expected,
        miss_probability_n=miss,
        n_for_target_detection=n_target,
    )
