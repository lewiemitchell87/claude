"""Value detection: compare model probabilities against market prices.

A *back* bet on a selection at decimal odds ``O`` with true win probability ``p``
has expected profit per unit staked of ``p * (O - 1) - (1 - p) = p*O - 1``. When
this is positive the bet has positive expected value ("value").

We also report:
* ``edge``      — ``p - 1/O``, the probability advantage over the price.
* ``kelly``     — the full Kelly stake fraction maximising long-run growth.

Staking uses *fractional* Kelly (a cap on the Kelly fraction) because full Kelly
is notoriously aggressive and unforgiving of model error.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


def expected_value(model_prob: float, odds: float) -> float:
    """Expected profit per unit stake for a back bet (``p*O - 1``)."""
    return model_prob * odds - 1.0


def edge(model_prob: float, odds: float) -> float:
    """Probability edge over the offered price (``p - 1/O``)."""
    if odds <= 0:
        return 0.0
    return model_prob - 1.0 / odds


def kelly_fraction(model_prob: float, odds: float) -> float:
    """Full Kelly stake fraction for a back bet. Never negative."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * odds - 1.0) / b
    return max(0.0, f)


def fair_odds(model_prob: float) -> float:
    """Break-even decimal odds implied by the model probability."""
    if model_prob <= 0:
        return float("inf")
    return 1.0 / model_prob


@dataclass
class ValueOpportunity:
    """A single positive-EV selection."""

    market: str
    selection: str
    model_prob: float
    odds: float
    fair_odds: float
    implied_prob: float
    edge: float
    ev: float
    kelly: float
    stake: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def find_value(
    market: str,
    model_probs: Dict[str, float],
    market_odds: Dict[str, float],
    *,
    min_edge: float = 0.0,
    min_ev: float = 0.02,
    kelly_cap: float = 0.25,
    bankroll: float = 0.0,
) -> List[ValueOpportunity]:
    """Find value selections within a single market.

    Parameters
    ----------
    market:
        Market name (used only for labelling the results).
    model_probs:
        ``{selection: model probability}``.
    market_odds:
        ``{selection: decimal odds}`` offered by the book/exchange.
    min_edge:
        Minimum probability edge required to flag a bet.
    min_ev:
        Minimum expected value per unit stake required (e.g. ``0.02`` = +2%).
        A small positive threshold guards against flagging noise as value.
    kelly_cap:
        Upper bound on the Kelly fraction actually staked (fractional Kelly).
    bankroll:
        If > 0, ``stake`` is filled in as ``bankroll * min(kelly, kelly_cap)``.

    Returns
    -------
    list of :class:`ValueOpportunity`, sorted by expected value descending.
    """
    out: List[ValueOpportunity] = []
    for selection, odds in market_odds.items():
        if selection not in model_probs:
            continue
        if odds is None or odds <= 1.0:
            continue
        p = float(model_probs[selection])
        ev = expected_value(p, odds)
        eg = edge(p, odds)
        if ev >= min_ev and eg >= min_edge:
            kf = kelly_fraction(p, odds)
            staked_fraction = min(kf, kelly_cap)
            out.append(
                ValueOpportunity(
                    market=market,
                    selection=selection,
                    model_prob=p,
                    odds=float(odds),
                    fair_odds=fair_odds(p),
                    implied_prob=1.0 / odds,
                    edge=eg,
                    ev=ev,
                    kelly=kf,
                    stake=bankroll * staked_fraction if bankroll > 0 else 0.0,
                )
            )
    out.sort(key=lambda o: o.ev, reverse=True)
    return out


def scan_markets(
    model_markets: Dict[str, Dict[str, float]],
    odds_markets: Dict[str, Dict[str, float]],
    *,
    min_edge: float = 0.0,
    min_ev: float = 0.02,
    kelly_cap: float = 0.25,
    bankroll: float = 0.0,
) -> List[ValueOpportunity]:
    """Run :func:`find_value` across every market present in both inputs."""
    results: List[ValueOpportunity] = []
    for market, probs in model_markets.items():
        if market not in odds_markets:
            continue
        results.extend(
            find_value(
                market,
                probs,
                odds_markets[market],
                min_edge=min_edge,
                min_ev=min_ev,
                kelly_cap=kelly_cap,
                bankroll=bankroll,
            )
        )
    results.sort(key=lambda o: o.ev, reverse=True)
    return results
