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
    """A single positive-EV selection.

    ``stake`` is the amount **put at risk** (for a back bet that is the stake; for
    a lay bet it is the liability). ``ev`` is expressed per unit of that amount, so
    back and lay opportunities are directly comparable. ``backer_stake`` is the
    matched stake of a lay bet (``liability / (lay_odds - 1)``); it is 0 for backs.
    """

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
    side: str = "back"          # "back" or "lay"
    commission: float = 0.0
    backer_stake: float = 0.0   # for lay bets: the matched stake (stake == liability)

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


# --------------------------------------------------------------------------- #
# Exchange support: back AND lay, with commission on net winnings.
#
# A lay bet at odds L is mathematically a back bet on the opposite outcome at
# decimal odds L/(L-1), where the amount staked is the liability (L-1) per unit
# of matched backer stake. We unify both sides through a single payoff helper so
# the EV/Kelly maths is identical and provably consistent.
# --------------------------------------------------------------------------- #
def _payoff_metrics(win_prob: float, payoff: float):
    """Given win probability and net profit per unit *staked-at-risk* if the bet
    wins, return ``(ev, kelly, effective_odds)``.

    ``ev`` is per unit staked-at-risk; full Kelly fraction is ``ev / payoff``.
    """
    if payoff <= 0:
        return 0.0, 0.0, float("inf")
    ev = win_prob * payoff - (1.0 - win_prob)
    kelly = max(0.0, ev / payoff)
    effective_odds = payoff + 1.0
    return ev, kelly, effective_odds


def back_payoff(odds: float, commission: float = 0.0) -> float:
    """Net profit per unit stake if a back bet wins (after commission)."""
    return (odds - 1.0) * (1.0 - commission)


def lay_payoff(lay_odds: float, commission: float = 0.0) -> float:
    """Net profit per unit *liability* if a lay bet wins (selection loses)."""
    if lay_odds <= 1.0:
        return 0.0
    return (1.0 - commission) / (lay_odds - 1.0)


def evaluate_selection(
    market: str,
    selection: str,
    model_prob: float,
    *,
    back_odds: Optional[float] = None,
    lay_odds: Optional[float] = None,
    commission: float = 0.0,
    min_edge: float = 0.0,
    min_ev: float = 0.02,
    kelly_cap: float = 0.25,
    bankroll: float = 0.0,
) -> Optional[ValueOpportunity]:
    """Return the better of the available back/lay bets on a selection, if it
    clears the thresholds; otherwise ``None``.

    For a lay bet the model "win" probability is ``1 - model_prob`` (the
    selection must *lose*), the price implies the selection wins with probability
    ``1/lay_odds``, so lay value exists when ``model_prob < 1/lay_odds``.
    """
    candidates: List[ValueOpportunity] = []

    if back_odds is not None and back_odds > 1.0:
        payoff = back_payoff(back_odds, commission)
        ev, kelly, _ = _payoff_metrics(model_prob, payoff)
        eg = model_prob - 1.0 / back_odds
        if ev >= min_ev and eg >= min_edge:
            staked = bankroll * min(kelly, kelly_cap) if bankroll > 0 else 0.0
            candidates.append(
                ValueOpportunity(
                    market=market, selection=selection, model_prob=model_prob,
                    odds=float(back_odds), fair_odds=fair_odds(model_prob),
                    implied_prob=1.0 / back_odds, edge=eg, ev=ev, kelly=kelly,
                    stake=staked, side="back", commission=commission,
                )
            )

    if lay_odds is not None and lay_odds > 1.0:
        win_prob = 1.0 - model_prob          # selection loses
        payoff = lay_payoff(lay_odds, commission)
        ev, kelly, _ = _payoff_metrics(win_prob, payoff)
        eg = (1.0 / lay_odds) - model_prob   # we think it's less likely than the price
        if ev >= min_ev and eg >= min_edge:
            liability = bankroll * min(kelly, kelly_cap) if bankroll > 0 else 0.0
            backer_stake = liability / (lay_odds - 1.0) if liability else 0.0
            candidates.append(
                ValueOpportunity(
                    market=market, selection=selection, model_prob=model_prob,
                    odds=float(lay_odds), fair_odds=fair_odds(win_prob),
                    implied_prob=1.0 / lay_odds, edge=eg, ev=ev, kelly=kelly,
                    stake=liability, side="lay", commission=commission,
                    backer_stake=backer_stake,
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda o: o.ev)


def find_value_exchange(
    market: str,
    model_probs: Dict[str, float],
    back_odds: Optional[Dict[str, float]] = None,
    lay_odds: Optional[Dict[str, float]] = None,
    *,
    commission: float = 0.0,
    min_edge: float = 0.0,
    min_ev: float = 0.02,
    kelly_cap: float = 0.25,
    bankroll: float = 0.0,
) -> List[ValueOpportunity]:
    """Value selections within one market considering both back and lay prices."""
    back_odds = back_odds or {}
    lay_odds = lay_odds or {}
    out: List[ValueOpportunity] = []
    for selection, p in model_probs.items():
        opp = evaluate_selection(
            market, selection, float(p),
            back_odds=back_odds.get(selection),
            lay_odds=lay_odds.get(selection),
            commission=commission, min_edge=min_edge, min_ev=min_ev,
            kelly_cap=kelly_cap, bankroll=bankroll,
        )
        if opp is not None:
            out.append(opp)
    out.sort(key=lambda o: o.ev, reverse=True)
    return out


def scan_markets_exchange(
    model_markets: Dict[str, Dict[str, float]],
    back_markets: Optional[Dict[str, Dict[str, float]]] = None,
    lay_markets: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    commission: float = 0.0,
    min_edge: float = 0.0,
    min_ev: float = 0.02,
    kelly_cap: float = 0.25,
    bankroll: float = 0.0,
) -> List[ValueOpportunity]:
    """Run :func:`find_value_exchange` across every modelled market."""
    back_markets = back_markets or {}
    lay_markets = lay_markets or {}
    results: List[ValueOpportunity] = []
    markets = set(back_markets) | set(lay_markets)
    for market in markets:
        if market not in model_markets:
            continue
        results.extend(
            find_value_exchange(
                market, model_markets[market],
                back_markets.get(market), lay_markets.get(market),
                commission=commission, min_edge=min_edge, min_ev=min_ev,
                kelly_cap=kelly_cap, bankroll=bankroll,
            )
        )
    results.sort(key=lambda o: o.ev, reverse=True)
    return results
