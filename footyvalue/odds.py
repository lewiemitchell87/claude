"""Odds <-> probability utilities and overround (vig) removal.

The market's quoted decimal odds imply probabilities ``1/O`` that sum to more
than 1 — the excess is the bookmaker's margin ("overround"/"vig"). Removing it
gives the market's *fair* probability estimate, which is a useful sanity-check
baseline to compare your own model against.
"""

from __future__ import annotations

from typing import Dict, List


def decimal_to_prob(odds: float) -> float:
    """Implied (gross) probability of decimal odds."""
    if odds <= 0:
        raise ValueError("Decimal odds must be positive")
    return 1.0 / odds


def prob_to_decimal(prob: float) -> float:
    """Fair decimal odds for a probability."""
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def overround(odds: List[float]) -> float:
    """Total booked percentage minus 1 (the margin). 0 == a fair book."""
    return sum(1.0 / o for o in odds) - 1.0


def remove_overround_proportional(odds: List[float]) -> List[float]:
    """Fair probabilities via proportional normalisation of ``1/O``.

    Simple and robust; assumes the margin is applied proportionally across
    outcomes.
    """
    raw = [1.0 / o for o in odds]
    total = sum(raw)
    if total <= 0:
        raise ValueError("Invalid odds")
    return [r / total for r in raw]


def remove_overround_shin(odds: List[float], max_iter: int = 100, tol: float = 1e-12) -> List[float]:
    """Fair probabilities via Shin's (1992) model.

    Shin's method attributes the margin to a proportion ``z`` of insider trading
    and tends to be more accurate than proportional scaling for longshots. Falls
    back to proportional normalisation if it fails to converge.
    """
    raw = [1.0 / o for o in odds]
    booksum = sum(raw)
    n = len(raw)
    if booksum <= 0:
        raise ValueError("Invalid odds")

    z = 0.0
    for _ in range(max_iter):
        denom = 0.0
        probs = []
        for pi in raw:
            val = ((z * z + 4.0 * (1.0 - z) * pi * pi / booksum) ** 0.5 - z) / (2.0 * (1.0 - z))
            probs.append(val)
            denom += val
        # Update z from the implied probabilities.
        new_z = (sum((p * p) for p in probs) * booksum - 1.0) / (n - 1.0) if n > 1 else 0.0
        new_z = min(max(new_z, 0.0), 0.5)
        if abs(new_z - z) < tol:
            z = new_z
            break
        z = new_z

    denom = 0.0
    probs = []
    for pi in raw:
        val = ((z * z + 4.0 * (1.0 - z) * pi * pi / booksum) ** 0.5 - z) / (2.0 * (1.0 - z))
        probs.append(val)
        denom += val
    if denom <= 0:
        return remove_overround_proportional(odds)
    return [p / denom for p in probs]


def fair_probabilities(odds: Dict[str, float], method: str = "proportional") -> Dict[str, float]:
    """Return ``{selection: fair probability}`` with the margin removed."""
    selections = list(odds.keys())
    values = [odds[s] for s in selections]
    if method == "shin":
        fair = remove_overround_shin(values)
    elif method == "proportional":
        fair = remove_overround_proportional(values)
    else:
        raise ValueError(f"Unknown method: {method}")
    return dict(zip(selections, fair))
