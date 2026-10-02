"""Exact Joint Rank Certification (EJRC).

Reusable analytical certificates for positive Einstein aggregation under
bounded uncertainty in criterion weights and normalized scores.

The implementation follows the manuscript's notation but is intentionally
small: it exposes pairwise, winner, shortlist, symmetric-box, and budgeted
weight certificates without depending on the paper's plotting or reporting
code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class PairwiseCertificate:
    """Result of an exact pairwise box certificate."""

    value: float
    certified: bool
    transformed_worst_case: np.ndarray
    adverse_multipliers: np.ndarray


@dataclass(frozen=True)
class SetCertificate:
    """Winner/shortlist certificate over all set-boundary comparisons."""

    value: float
    certified: bool
    controlling_pair: tuple[int, int]


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return a


def _normalize_reference_weights(weights: ArrayLike) -> np.ndarray:
    w = _as_1d(weights, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have a positive finite sum")
    return w / total


def _validate_scores(scores: np.ndarray, name: str = "scores") -> None:
    if np.any(~np.isfinite(scores)):
        raise ValueError(f"{name} must be finite")
    if np.any(scores < 0) or np.any(scores >= 1):
        raise ValueError(f"{name} must lie in [0, 1)")


def einstein_scores(matrix: np.ndarray, weights: ArrayLike) -> np.ndarray:
    """Positive Einstein weighted scores for an alternatives-by-criteria matrix."""
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _validate_scores(X, "matrix")
    w = _normalize_reference_weights(weights)
    if X.shape[1] != len(w):
        raise ValueError("matrix column count must equal number of weights")
    return np.tanh(np.arctanh(X) @ w)


def renormalize_weights(weights: ArrayLike, multipliers: ArrayLike) -> np.ndarray:
    """Apply positive multiplicative perturbations and renormalize to unit sum."""
    w = _normalize_reference_weights(weights)
    q = _as_1d(multipliers, "multipliers")
    if len(q) != len(w):
        raise ValueError("weights and multipliers must have the same length")
    if np.any(q <= 0):
        raise ValueError("multipliers must be strictly positive")
    z = w * q
    return z / z.sum()


def pairwise_box_certificate(
    reference_weights: ArrayLike,
    q_lower: ArrayLike,
    q_upper: ArrayLike,
    leader_lower_scores: ArrayLike,
    competitor_upper_scores: ArrayLike,
) -> PairwiseCertificate:
    """Exact necessary-and-sufficient pairwise certificate on a rectangular box."""
    v = _normalize_reference_weights(reference_weights)
    ql = _as_1d(q_lower, "q_lower")
    qu = _as_1d(q_upper, "q_upper")
    Li = _as_1d(leader_lower_scores, "leader_lower_scores")
    Uk = _as_1d(competitor_upper_scores, "competitor_upper_scores")
    n = len(v)
    if not (len(ql) == len(qu) == len(Li) == len(Uk) == n):
        raise ValueError("all criterion vectors must have the same length")
    if np.any(ql <= 0) or np.any(qu < ql):
        raise ValueError("multiplier bounds require 0 < q_lower <= q_upper")
    _validate_scores(Li, "leader_lower_scores")
    _validate_scores(Uk, "competitor_upper_scores")
    d_wc = np.arctanh(Li) - np.arctanh(Uk)
    q_adv = np.where(d_wc >= 0.0, ql, qu)
    value = float(np.sum(v * q_adv * d_wc))
    return PairwiseCertificate(value, value > 0.0, d_wc, q_adv)


def symmetric_joint_certificate(
    matrix: np.ndarray,
    reference_weights: ArrayLike,
    leader: int,
    competitor: int,
    r: float,
    delta: float,
) -> PairwiseCertificate:
    """Special-case EJRC certificate with common r and common delta."""
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _validate_scores(X, "matrix")
    if not (0 <= r < 1):
        raise ValueError("r must satisfy 0 <= r < 1")
    if delta < 0:
        raise ValueError("delta must be nonnegative")
    Li = X[leader] - delta
    Uk = X[competitor] + delta
    if np.any(Li < 0) or np.any(Uk >= 1):
        raise ValueError("adverse score endpoints leave the admissible [0,1) domain")
    n = X.shape[1]
    return pairwise_box_certificate(
        reference_weights,
        np.full(n, 1.0 - r),
        np.full(n, 1.0 + r),
        Li,
        Uk,
    )


def weight_only_radius(
    matrix: np.ndarray,
    reference_weights: ArrayLike,
    leader: int,
    competitor: int,
) -> float:
    """First tie boundary for symmetric multiplicative weight uncertainty."""
    X = np.asarray(matrix, dtype=float)
    _validate_scores(X, "matrix")
    v = _normalize_reference_weights(reference_weights)
    d = np.arctanh(X[leader]) - np.arctanh(X[competitor])
    c = v * d
    P = float(c[c >= 0].sum())
    N = float(c[c < 0].sum())
    if P + N <= 0:
        raise ValueError("leader must have a positive nominal transformed margin")
    if abs(N) < 1e-16:
        return 1.0
    return float((P + N) / (P - N))


def winner_certificate(
    reference_weights: ArrayLike,
    q_lower: ArrayLike,
    q_upper: ArrayLike,
    score_lower: np.ndarray,
    score_upper: np.ndarray,
    winner: int,
) -> SetCertificate:
    """Exact certificate that one alternative remains above every competitor."""
    m = np.asarray(score_lower).shape[0]
    return shortlist_certificate(
        reference_weights, q_lower, q_upper, score_lower, score_upper,
        shortlist=[winner], universe=range(m),
    )


def shortlist_certificate(
    reference_weights: ArrayLike,
    q_lower: ArrayLike,
    q_upper: ArrayLike,
    score_lower: np.ndarray,
    score_upper: np.ndarray,
    shortlist: Iterable[int],
    universe: Iterable[int] | None = None,
) -> SetCertificate:
    """Exact certificate preserving shortlist membership, not internal order."""
    L = np.asarray(score_lower, dtype=float)
    U = np.asarray(score_upper, dtype=float)
    if L.shape != U.shape or L.ndim != 2:
        raise ValueError("score_lower and score_upper must have the same 2D shape")
    _validate_scores(L, "score_lower")
    _validate_scores(U, "score_upper")
    if np.any(L > U):
        raise ValueError("score_lower cannot exceed score_upper")
    S = tuple(dict.fromkeys(int(i) for i in shortlist))
    if not S:
        raise ValueError("shortlist must not be empty")
    all_idx = tuple(range(L.shape[0])) if universe is None else tuple(int(i) for i in universe)
    out = tuple(i for i in all_idx if i not in S)
    if not out:
        raise ValueError("shortlist must not contain the entire universe")
    best_value = float("inf")
    best_pair = (-1, -1)
    for i in S:
        for k in out:
            c = pairwise_box_certificate(
                reference_weights, q_lower, q_upper, L[i], U[k]
            ).value
            if c < best_value:
                best_value = c
                best_pair = (i, k)
    return SetCertificate(best_value, best_value > 0.0, best_pair)


def budgeted_weight_certificate(
    reference_weights: ArrayLike,
    transformed_worst_case: ArrayLike,
    r: float,
    gamma: float,
) -> float:
    """Exact Bertsimas-Sim-style budgeted EJRC weight certificate."""
    if r < 0 or r >= 1:
        raise ValueError("r must satisfy 0 <= r < 1")
    v = _normalize_reference_weights(reference_weights)
    d = _as_1d(transformed_worst_case, "transformed_worst_case")
    if len(d) != len(v):
        raise ValueError("weights and transformed_worst_case must have same length")
    n = len(v)
    if gamma < 0 or gamma > n:
        raise ValueError("gamma must satisfy 0 <= gamma <= number of criteria")
    a = v * d
    b = np.sort(np.abs(a))[::-1]
    g = int(np.floor(gamma))
    frac = gamma - g
    penalty = float(np.sum(b[:g]))
    if g < n:
        penalty += frac * float(b[g])
    return float(np.sum(a) - r * penalty)


def budgeted_pair_certificate(
    reference_weights: ArrayLike,
    leader_lower_scores: ArrayLike,
    competitor_upper_scores: ArrayLike,
    r: float,
    gamma: float,
) -> float:
    """Convenience wrapper for a budgeted pairwise score/weight certificate."""
    Li = _as_1d(leader_lower_scores, "leader_lower_scores")
    Uk = _as_1d(competitor_upper_scores, "competitor_upper_scores")
    _validate_scores(Li, "leader_lower_scores")
    _validate_scores(Uk, "competitor_upper_scores")
    if len(Li) != len(Uk):
        raise ValueError("score vectors must have the same length")
    d = np.arctanh(Li) - np.arctanh(Uk)
    return budgeted_weight_certificate(reference_weights, d, r, gamma)


def gamma_star(
    reference_weights: ArrayLike,
    transformed_worst_case: ArrayLike,
    r: float,
) -> float:
    """First budget level at which the pairwise certificate reaches a tie."""
    if r <= 0 or r >= 1:
        raise ValueError("r must satisfy 0 < r < 1")
    v = _normalize_reference_weights(reference_weights)
    d = _as_1d(transformed_worst_case, "transformed_worst_case")
    if len(d) != len(v):
        raise ValueError("weights and transformed_worst_case must have same length")
    a = v * d
    nominal = float(np.sum(a))
    if nominal <= 0:
        return 0.0
    b = np.sort(np.abs(a))[::-1]
    target = nominal / r
    cumulative = 0.0
    for h, val in enumerate(b):
        if target <= cumulative + val + 1e-15:
            if val == 0:
                return float(h)
            return float(h + (target - cumulative) / val)
        cumulative += float(val)
    return float("inf")
