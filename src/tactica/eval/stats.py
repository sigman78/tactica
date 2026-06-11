"""Statistical helpers: confidence intervals and the SPRT log-likelihood ratio."""
from __future__ import annotations

import math


def mean_ci95(samples: list[float]) -> tuple[float, float]:
    """(mean, 95% half-width) via the normal approximation on the sample."""
    n = len(samples)
    if n == 0:
        return 0.5, float("inf")
    mean = sum(samples) / n
    if n == 1:
        return mean, float("inf")
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_llr(wins: int, draws: int, losses: int,
             elo0: float, elo1: float) -> float:
    """Trinomial GSPRT log-likelihood ratio (the standard fishtest-style
    approximation) for H1: elo=elo1 vs H0: elo=elo0."""
    n = wins + draws + losses
    if n == 0 or wins == n or losses == n:
        # Degenerate sample: regularize by adding half a draw.
        wins, draws, losses = wins, draws + 1, losses
        n += 1
    w, d = wins / n, draws / n
    score = w + d / 2.0
    m2 = w + d / 4.0
    var = m2 - score**2
    if var <= 0:
        return 0.0
    var_s = var / n
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * var_s)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper) LLR bounds: cross upper -> accept H1, lower -> accept H0."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)
