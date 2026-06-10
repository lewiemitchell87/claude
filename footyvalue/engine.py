"""Orchestration: turn ratings + market odds into ranked value opportunities.

This is the layer the CLI (and any future service) calls. It is intentionally
side-effect free: feed it a :class:`~footyvalue.ratings.TeamRatings`, a fixture
and the market odds, and it returns the value selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ratings import TeamRatings
from .markets import all_markets
from .scoreline import score_matrix
from .inplay import inplay_markets
from .value import ValueOpportunity, scan_markets


@dataclass
class ScanConfig:
    """Thresholds and staking parameters for a value scan."""

    min_ev: float = 0.02
    min_edge: float = 0.0
    kelly_cap: float = 0.25
    bankroll: float = 0.0
    max_goals: int = 10
    ou_lines: tuple = (0.5, 1.5, 2.5, 3.5)


@dataclass
class FixtureResult:
    """The model probabilities and value bets for one fixture."""

    home: str
    away: str
    lam_home: float
    lam_away: float
    minute: Optional[int]
    score: Optional[tuple]
    model_markets: Dict[str, Dict[str, float]]
    opportunities: List[ValueOpportunity] = field(default_factory=list)


def model_markets_for_fixture(
    ratings: TeamRatings,
    home: str,
    away: str,
    config: ScanConfig,
    *,
    minute: Optional[int] = None,
    home_goals: int = 0,
    away_goals: int = 0,
):
    """Return model market probabilities and the expected goals for a fixture.

    If ``minute`` is provided, in-play probabilities (conditioned on the current
    score) are produced; otherwise pre-match probabilities are returned.
    """
    lam_home, lam_away = ratings.expected_goals(home, away)
    if minute is None:
        matrix = score_matrix(lam_home, lam_away, max_goals=config.max_goals, rho=ratings.rho)
        markets = all_markets(matrix, ou_lines=config.ou_lines)
    else:
        markets = inplay_markets(
            lam_home,
            lam_away,
            minute=minute,
            home_goals=home_goals,
            away_goals=away_goals,
            rho=ratings.rho,
            max_goals=config.max_goals,
            ou_lines=config.ou_lines,
        )
    return markets, lam_home, lam_away


def evaluate_fixture(
    ratings: TeamRatings,
    home: str,
    away: str,
    odds_markets: Dict[str, Dict[str, float]],
    config: Optional[ScanConfig] = None,
    *,
    minute: Optional[int] = None,
    home_goals: int = 0,
    away_goals: int = 0,
    sources: Optional[Dict[str, Dict[str, str]]] = None,
) -> FixtureResult:
    """Price a fixture against its market odds and return ranked value bets.

    ``sources`` optionally maps ``{market: {selection: bookmaker}}`` so each
    flagged opportunity records which book offers the price.
    """
    config = config or ScanConfig()
    model_markets, lam_home, lam_away = model_markets_for_fixture(
        ratings, home, away, config,
        minute=minute, home_goals=home_goals, away_goals=away_goals,
    )

    opportunities = scan_markets(
        model_markets,
        odds_markets,
        sources=sources,
        min_edge=config.min_edge,
        min_ev=config.min_ev,
        kelly_cap=config.kelly_cap,
        bankroll=config.bankroll,
    )

    return FixtureResult(
        home=home,
        away=away,
        lam_home=lam_home,
        lam_away=lam_away,
        minute=minute,
        score=(home_goals, away_goals) if minute is not None else None,
        model_markets=model_markets,
        opportunities=opportunities,
    )


def scan_fixtures(
    ratings: TeamRatings,
    fixtures: List[dict],
    config: Optional[ScanConfig] = None,
) -> List[FixtureResult]:
    """Evaluate many fixtures.

    Each fixture dict needs ``home``, ``away`` and ``markets`` (the odds), and may
    optionally include ``minute``, ``home_goals`` and ``away_goals`` for in-play.
    Results are returned sorted by the best EV available in each fixture.
    """
    config = config or ScanConfig()
    results: List[FixtureResult] = []
    for fx in fixtures:
        result = evaluate_fixture(
            ratings,
            fx["home"],
            fx["away"],
            fx.get("markets", {}),
            config,
            minute=fx.get("minute"),
            home_goals=fx.get("home_goals", 0),
            away_goals=fx.get("away_goals", 0),
            sources=fx.get("sources"),
        )
        results.append(result)

    results.sort(
        key=lambda r: (r.opportunities[0].ev if r.opportunities else float("-inf")),
        reverse=True,
    )
    return results
