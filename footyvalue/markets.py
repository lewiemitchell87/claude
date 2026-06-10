"""Derive market probabilities from a scoreline matrix.

Every function here takes a ``P[home, away]`` matrix (see :mod:`footyvalue.scoreline`)
and returns a dict of ``{selection: probability}``. Because all markets are read
off the same matrix they are guaranteed mutually consistent (e.g. Over + Under +
Push == 1, Home + Draw + Away == 1).
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def match_odds(matrix: np.ndarray) -> Dict[str, float]:
    """1X2 (Match Odds) probabilities: home win, draw, away win."""
    home = float(np.tril(matrix, -1).sum())  # home goals > away goals
    away = float(np.triu(matrix, 1).sum())   # away goals > home goals
    draw = float(np.trace(matrix))
    return {"home": home, "draw": draw, "away": away}


def total_goals_distribution(matrix: np.ndarray) -> np.ndarray:
    """Distribution over total goals (home + away)."""
    n = matrix.shape[0]
    totals = np.zeros(2 * (n - 1) + 1)
    i_idx, j_idx = np.indices(matrix.shape)
    np.add.at(totals, (i_idx + j_idx).ravel(), matrix.ravel())
    return totals


def over_under(matrix: np.ndarray, line: float) -> Dict[str, float]:
    """Over/Under total goals for ``line`` (e.g. 2.5, or an integer push line).

    For half-lines (2.5) ``push`` is 0. For whole lines (2.0) the probability of
    landing exactly on the line is returned as ``push``.
    """
    totals = total_goals_distribution(matrix)
    goals = np.arange(len(totals))
    over = float(totals[goals > line].sum())
    under = float(totals[goals < line].sum())
    push = float(1.0 - over - under)
    # Numerical hygiene: clamp a tiny negative push to zero.
    if -1e-12 < push < 0.0:
        push = 0.0
    return {"over": over, "under": under, "push": push}


def btts(matrix: np.ndarray) -> Dict[str, float]:
    """Both Teams To Score (Yes/No)."""
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": float(1.0 - yes)}


def correct_score(matrix: np.ndarray, max_each: int = 5) -> Dict[str, float]:
    """Correct-score probabilities up to ``max_each`` goals per team.

    Scores beyond the grid are bucketed under ``"other"``.
    """
    out: Dict[str, float] = {}
    covered = 0.0
    n = matrix.shape[0]
    for h in range(min(max_each, n - 1) + 1):
        for a in range(min(max_each, n - 1) + 1):
            p = float(matrix[h, a])
            out[f"{h}-{a}"] = p
            covered += p
    out["other"] = float(max(0.0, 1.0 - covered))
    return out


def all_markets(matrix: np.ndarray, ou_lines: Iterable[float] = (0.5, 1.5, 2.5, 3.5)) -> Dict[str, Dict[str, float]]:
    """Convenience: every supported market in one structure.

    Returns a dict keyed by market name. Over/Under lines are keyed as
    ``"over_under_2.5"`` etc.
    """
    out: Dict[str, Dict[str, float]] = {
        "match_odds": match_odds(matrix),
        "btts": btts(matrix),
    }
    for line in ou_lines:
        out[f"over_under_{line}"] = over_under(matrix, line)
    return out
