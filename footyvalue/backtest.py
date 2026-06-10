"""Walk-forward backtester with CLV and ROI metrics.

The backtester replays historical matches in chronological order. For each match
it fits ratings on **only the matches that came before** (no lookahead), prices
every market, places the value bets the engine would have flagged, settles them
against the real result, and records the profit and the closing-line value (CLV).

Metrics reported:

* **ROI / yield** — profit divided by total amount staked.
* **win rate** — fraction of settled bets that won.
* **CLV** — did we beat the closing price? Positive average CLV is the single best
  predictor that an edge is real, independent of short-run variance.
* **max drawdown**, an equity curve, and a per-market / per-side breakdown.

Input format
------------
``matches`` is a list of dicts, each::

    {
      "date": date(...), "home": "A", "away": "B",
      "home_goals": 2, "away_goals": 1,
      "odds":    {market: {selection: decimal_odds}},      # the price you'd take
      "closing": {market: {selection: decimal_odds}},      # optional, for CLV
    }

Only the ``odds`` markets are bet; ``closing`` is used solely to measure CLV.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from .ratings import Match, fit_dixon_coles, TeamRatings
from .scoreline import score_matrix
from .markets import all_markets
from .value import evaluate_selection, ValueOpportunity


# --------------------------------------------------------------------------- #
# Settlement
# --------------------------------------------------------------------------- #
def settle(market: str, selection: str, home_goals: int, away_goals: int) -> Optional[bool]:
    """Return True if the selection won, False if it lost, None if it pushed/void."""
    if market == "match_odds":
        if home_goals > away_goals:
            result = "home"
        elif home_goals < away_goals:
            result = "away"
        else:
            result = "draw"
        return selection == result

    if market == "btts":
        both = home_goals >= 1 and away_goals >= 1
        return both if selection == "yes" else (not both)

    if market.startswith("over_under_"):
        line = float(market.rsplit("_", 1)[1])
        total = home_goals + away_goals
        if total == line:          # whole-line push
            return None
        over = total > line
        return over if selection == "over" else (not over)

    raise ValueError(f"Cannot settle unknown market: {market}")


def bet_profit(opp: ValueOpportunity, won: Optional[bool]) -> float:
    """Profit for a settled bet, in the same units as ``opp.stake`` (amount at risk).

    Back: win => stake * (odds-1) * (1-comm); lose => -stake.
    Lay:  win (selection loses) => backer_stake * (1-comm); lose => -liability.
    Push/void => 0.
    """
    if won is None:
        return 0.0
    if opp.side == "back":
        if won:
            return opp.stake * (opp.odds - 1.0) * (1.0 - opp.commission)
        return -opp.stake
    else:  # lay: "won" here means the *selection* won => the lay loses
        selection_won = won
        if selection_won:
            return -opp.stake  # lose the liability
        return opp.backer_stake * (1.0 - opp.commission)


def closing_value(opp: ValueOpportunity, closing_odds: Optional[float]) -> Optional[float]:
    """CLV as a fraction. Positive means we got a better price than the close.

    Back: ``taken / closing - 1`` (higher back odds are better).
    Lay:  ``closing / taken - 1`` (lower lay odds are better).
    """
    if closing_odds is None or closing_odds <= 1.0:
        return None
    if opp.side == "back":
        return opp.odds / closing_odds - 1.0
    return closing_odds / opp.odds - 1.0


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class SettledBet:
    date: Optional[str]
    home: str
    away: str
    market: str
    selection: str
    side: str
    odds: float
    model_prob: float
    ev: float
    stake: float
    won: Optional[bool]
    profit: float
    clv: Optional[float]
    bankroll_after: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    bets: List[SettledBet] = field(default_factory=list)
    starting_bankroll: float = 0.0
    final_bankroll: float = 0.0

    # summary metrics (filled by ``summarise``)
    n_bets: int = 0
    n_won: int = 0
    n_push: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    avg_ev: float = 0.0
    avg_clv: Optional[float] = None
    pct_positive_clv: Optional[float] = None
    max_drawdown: float = 0.0
    by_market: Dict[str, dict] = field(default_factory=dict)
    by_side: Dict[str, dict] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)

    def summary_text(self) -> str:
        clv = "n/a" if self.avg_clv is None else f"{self.avg_clv*100:+.2f}%"
        pos = "n/a" if self.pct_positive_clv is None else f"{self.pct_positive_clv*100:.0f}%"
        lines = [
            f"Bets:          {self.n_bets}  (won {self.n_won}, push {self.n_push})",
            f"Win rate:      {self.win_rate*100:.1f}%",
            f"Total staked:  {self.total_staked:,.2f}",
            f"Profit:        {self.total_profit:+,.2f}",
            f"ROI / yield:   {self.roi*100:+.2f}%",
            f"Avg EV (model):{self.avg_ev*100:+.2f}%",
            f"Avg CLV:       {clv}   (positive: {pos})",
            f"Max drawdown:  {self.max_drawdown:,.2f}",
            f"Bankroll:      {self.starting_bankroll:,.2f} -> {self.final_bankroll:,.2f}",
        ]
        if self.by_market:
            lines.append("By market:")
            for m, s in sorted(self.by_market.items()):
                lines.append(
                    f"  {m:<16} bets {s['n']:>4}  staked {s['staked']:>10,.1f}  "
                    f"profit {s['profit']:>+10,.1f}  ROI {s['roi']*100:>+6.1f}%"
                )
        return "\n".join(lines)


def _summarise(result: BacktestResult) -> BacktestResult:
    bets = result.bets
    result.n_bets = len(bets)
    settled = [b for b in bets if b.won is not None]
    result.n_won = sum(1 for b in bets if b.won is True)
    result.n_push = sum(1 for b in bets if b.won is None)
    result.total_staked = sum(b.stake for b in bets)
    result.total_profit = sum(b.profit for b in bets)
    result.roi = result.total_profit / result.total_staked if result.total_staked else 0.0
    result.win_rate = (result.n_won / len(settled)) if settled else 0.0
    result.avg_ev = (sum(b.ev for b in bets) / len(bets)) if bets else 0.0

    clvs = [b.clv for b in bets if b.clv is not None]
    if clvs:
        result.avg_clv = sum(clvs) / len(clvs)
        result.pct_positive_clv = sum(1 for c in clvs if c > 0) / len(clvs)

    # equity curve + max drawdown
    equity = [result.starting_bankroll]
    peak = result.starting_bankroll
    max_dd = 0.0
    for b in bets:
        equity.append(b.bankroll_after)
        peak = max(peak, b.bankroll_after)
        max_dd = max(max_dd, peak - b.bankroll_after)
    result.equity_curve = equity
    result.max_drawdown = max_dd
    result.final_bankroll = equity[-1]

    # breakdowns
    def breakdown(key_fn):
        out: Dict[str, dict] = {}
        for b in bets:
            k = key_fn(b)
            s = out.setdefault(k, {"n": 0, "staked": 0.0, "profit": 0.0, "won": 0})
            s["n"] += 1
            s["staked"] += b.stake
            s["profit"] += b.profit
            s["won"] += 1 if b.won else 0
        for s in out.values():
            s["roi"] = s["profit"] / s["staked"] if s["staked"] else 0.0
        return out

    result.by_market = breakdown(lambda b: b.market)
    result.by_side = breakdown(lambda b: b.side)
    return result


# --------------------------------------------------------------------------- #
# Backtest driver
# --------------------------------------------------------------------------- #
def run_backtest(
    matches: Sequence[dict],
    *,
    min_train_matches: int = 60,
    refit_every: int = 10,
    xi: float = 0.0,
    fit_rho: bool = True,
    min_ev: float = 0.02,
    min_edge: float = 0.0,
    commission: float = 0.0,
    allow_lay: bool = False,
    staking: str = "flat",          # "flat" or "kelly"
    stake_unit: float = 1.0,        # flat stake (level stakes)
    kelly_cap: float = 0.25,
    starting_bankroll: float = 1000.0,
    max_goals: int = 10,
    ou_lines: Sequence[float] = (0.5, 1.5, 2.5, 3.5),
) -> BacktestResult:
    """Replay ``matches`` chronologically, betting flagged value and settling it.

    ``staking="flat"`` stakes ``stake_unit`` per bet (level stakes — the cleanest
    way to read ROI). ``staking="kelly"`` stakes ``bankroll * min(kelly, cap)`` and
    compounds the bankroll.
    """
    ordered = sorted(
        matches,
        key=lambda m: (m.get("date") is None, m.get("date")),
    )

    result = BacktestResult(starting_bankroll=starting_bankroll)
    bankroll = starting_bankroll
    training: List[Match] = []
    ratings: Optional[TeamRatings] = None
    matches_since_fit = 0

    for m in ordered:
        home, away = m["home"], m["away"]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])

        # Fit on prior matches only. Refit periodically for speed.
        if len(training) >= min_train_matches and (
            ratings is None or matches_since_fit >= refit_every
        ):
            ratings = fit_dixon_coles(training, xi=xi, fit_rho=fit_rho)
            matches_since_fit = 0

        if ratings is not None and "odds" in m:
            lam_h, lam_a = ratings.expected_goals(home, away)
            matrix = score_matrix(lam_h, lam_a, max_goals=max_goals, rho=ratings.rho)
            model = all_markets(matrix, ou_lines=ou_lines)

            for market, sel_odds in m["odds"].items():
                if market not in model:
                    continue
                for selection, price in sel_odds.items():
                    if selection not in model[market]:
                        continue
                    p = model[market][selection]
                    stake_bankroll = bankroll if staking == "kelly" else 0.0
                    opp = evaluate_selection(
                        market, selection, p,
                        back_odds=price,
                        lay_odds=price if allow_lay else None,
                        commission=commission,
                        min_ev=min_ev, min_edge=min_edge,
                        kelly_cap=kelly_cap, bankroll=stake_bankroll,
                    )
                    if opp is None:
                        continue

                    # Determine actual stake-at-risk.
                    if staking == "kelly":
                        stake = opp.stake
                        if opp.side == "lay":
                            backer = opp.backer_stake
                        else:
                            backer = 0.0
                    else:  # flat / level stakes
                        stake = stake_unit
                        backer = stake_unit / (opp.odds - 1.0) if opp.side == "lay" else 0.0
                    if stake <= 0:
                        continue
                    opp.stake = stake
                    opp.backer_stake = backer

                    won = settle(market, selection, hg, ag)
                    profit = bet_profit(opp, won)
                    bankroll += profit

                    closing = (m.get("closing") or {}).get(market, {}).get(selection)
                    clv = closing_value(opp, closing)

                    result.bets.append(SettledBet(
                        date=str(m["date"]) if m.get("date") else None,
                        home=home, away=away, market=market, selection=selection,
                        side=opp.side, odds=opp.odds, model_prob=p, ev=opp.ev,
                        stake=stake, won=won, profit=profit, clv=clv,
                        bankroll_after=bankroll,
                    ))

        training.append(Match(home, away, hg, ag, m.get("date")))
        if ratings is not None:
            matches_since_fit += 1

    return _summarise(result)
