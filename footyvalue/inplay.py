"""In-play market probabilities.

Given the *pre-match* expected goals for each side, the current match minute and
the current score, we compute live probabilities for every market.

Model
-----
The final score is ``(current_home + H, current_away + A)`` where ``H`` and ``A``
are the *additional* goals scored in the remaining time. We assume the remaining
goals are Poisson with means scaled from the pre-match expectation by the fraction
of the match still to play (optionally modulated by a scoring-intensity profile,
since goals are slightly more frequent late in matches).

This deliberately keeps the same engine as the pre-match model: we build a
scoreline matrix for the *additional* goals and then offset every cell by the
current score before reading the markets.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .scoreline import score_matrix

DEFAULT_TOTAL_MINUTES = 90.0


def remaining_fraction(
    minute: float,
    total_minutes: float = DEFAULT_TOTAL_MINUTES,
    intensity: Optional[Callable[[float], float]] = None,
) -> float:
    """Fraction of the match's goal-expectation still to be played.

    With ``intensity=None`` this is simply the linear time remaining. An
    ``intensity`` callable ``g(t)`` (relative scoring rate at minute ``t``) lets
    you encode that scoring rates rise through a match; the remaining fraction is
    then the integral of ``g`` over the remaining minutes divided by its integral
    over the whole match.
    """
    minute = max(0.0, min(float(minute), total_minutes))
    if intensity is None:
        return (total_minutes - minute) / total_minutes

    grid = np.linspace(0.0, total_minutes, int(total_minutes) + 1)
    weights = np.array([max(0.0, float(intensity(t))) for t in grid])
    full = np.trapz(weights, grid)
    if full <= 0.0:
        return (total_minutes - minute) / total_minutes
    mask = grid >= minute
    remaining = np.trapz(weights[mask], grid[mask])
    return float(remaining / full)


def inplay_markets(
    lam_home: float,
    lam_away: float,
    minute: float,
    home_goals: int,
    away_goals: int,
    *,
    total_minutes: float = DEFAULT_TOTAL_MINUTES,
    rho: float = 0.0,
    max_goals: int = 10,
    ou_lines=(0.5, 1.5, 2.5, 3.5),
    intensity: Optional[Callable[[float], float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Live probabilities for all markets given the current game state.

    Parameters
    ----------
    lam_home, lam_away:
        Pre-match expected goals for each side (full 90 minutes).
    minute:
        Current match minute (0..total_minutes).
    home_goals, away_goals:
        Goals already scored.
    """
    if home_goals < 0 or away_goals < 0:
        raise ValueError("Scores must be non-negative")

    frac = remaining_fraction(minute, total_minutes, intensity)
    add = score_matrix(lam_home * frac, lam_away * frac, max_goals=max_goals, rho=rho)

    n = add.shape[0]
    i_idx, j_idx = np.indices((n, n))
    p = add  # additional-goal probabilities

    # Final-score quantities, offset by the goals already on the board.
    final_diff = (home_goals - away_goals) + (i_idx - j_idx)
    final_total = (home_goals + away_goals) + (i_idx + j_idx)
    final_home = home_goals + i_idx
    final_away = away_goals + j_idx

    home = float(p[final_diff > 0].sum())
    draw = float(p[final_diff == 0].sum())
    away = float(p[final_diff < 0].sum())

    yes = float(p[(final_home >= 1) & (final_away >= 1)].sum())

    markets: Dict[str, Dict[str, float]] = {
        "match_odds": {"home": home, "draw": draw, "away": away},
        "btts": {"yes": yes, "no": float(1.0 - yes)},
    }

    for line in ou_lines:
        over = float(p[final_total > line].sum())
        under = float(p[final_total < line].sum())
        push = float(1.0 - over - under)
        if -1e-12 < push < 0.0:
            push = 0.0
        markets[f"over_under_{line}"] = {"over": over, "under": under, "push": push}

    return markets
