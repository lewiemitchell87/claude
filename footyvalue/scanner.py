"""Continuous pre-game value scanner.

Polls an odds source on an interval, evaluates every **not-yet-started** fixture
against the model, and emits newly-appearing value bets (deduplicated, so you are
only alerted once per price). Supports both back and lay (exchange) staking with
commission, so it works with bookmaker feeds (back only) or an exchange feed that
also carries lay prices.

The odds source is a zero-argument callable returning a list of
:class:`FixtureOdds`. The bundled live source wraps The Odds API
(:mod:`footyvalue.data.odds_api`), but you can plug in any feed — including a
Betfair exchange adapter that fills in lay prices — by producing ``FixtureOdds``.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .ratings import TeamRatings
from .value import ValueOpportunity, scan_markets_exchange


@dataclass
class FixtureOdds:
    """Normalised pre-game fixture with back (and optional lay) prices."""

    event_id: str
    home: str
    away: str
    commence_time: str  # ISO-8601, e.g. "2026-06-10T14:00:00Z"
    back: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lay: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ScannerHit:
    """A freshly-detected value opportunity, with its fixture context."""

    event_id: str
    home: str
    away: str
    commence_time: str
    opportunity: ValueOpportunity


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


class PreGameScanner:
    """Stateful scanner that emits new value opportunities as prices appear."""

    def __init__(
        self,
        ratings: TeamRatings,
        odds_source: Callable[[], List[FixtureOdds]],
        *,
        min_ev: float = 0.02,
        min_edge: float = 0.0,
        kelly_cap: float = 0.25,
        bankroll: float = 0.0,
        commission: float = 0.0,
        min_seconds_to_start: float = 0.0,
        team_aliases: Optional[Dict[str, str]] = None,
        skip_unknown_teams: bool = True,
        improve_tick: float = 0.0,
    ):
        self.ratings = ratings
        self.odds_source = odds_source
        self.min_ev = min_ev
        self.min_edge = min_edge
        self.kelly_cap = kelly_cap
        self.bankroll = bankroll
        self.commission = commission
        self.min_seconds_to_start = min_seconds_to_start
        self.team_aliases = team_aliases or {}
        self.skip_unknown_teams = skip_unknown_teams
        self.improve_tick = improve_tick
        # key -> best price already emitted, so we only alert on new/improved value
        self._seen: Dict[tuple, float] = {}

    # -- helpers ------------------------------------------------------------ #
    def _resolve(self, team: str) -> str:
        return self.team_aliases.get(team, team)

    def _known(self, team: str) -> bool:
        return team in self.ratings.attack

    def _is_pregame(self, fixture: FixtureOdds, now: datetime) -> bool:
        start = _parse_iso(fixture.commence_time)
        if start is None:
            return True  # no timestamp: assume eligible
        return (start - now).total_seconds() >= self.min_seconds_to_start

    def _is_new(self, hit_key: tuple, price: float, side: str) -> bool:
        prev = self._seen.get(hit_key)
        if prev is None:
            return True
        # Re-alert only if the price improved beyond the tick (better = higher
        # back odds, or lower lay odds).
        if side == "back":
            return price >= prev + self.improve_tick and price > prev
        return price <= prev - self.improve_tick and price < prev

    # -- core --------------------------------------------------------------- #
    def evaluate_fixture(self, fixture: FixtureOdds) -> List[ValueOpportunity]:
        home = self._resolve(fixture.home)
        away = self._resolve(fixture.away)
        if self.skip_unknown_teams and not (self._known(home) and self._known(away)):
            return []
        model = self.ratings.market_probabilities(home, away)
        return scan_markets_exchange(
            model, fixture.back, fixture.lay,
            commission=self.commission, min_edge=self.min_edge,
            min_ev=self.min_ev, kelly_cap=self.kelly_cap, bankroll=self.bankroll,
        )

    def poll_once(self, now: Optional[datetime] = None) -> List[ScannerHit]:
        """One scan cycle: return only newly-appearing/improved opportunities."""
        now = now or datetime.now(timezone.utc)
        hits: List[ScannerHit] = []
        for fixture in self.odds_source():
            if not self._is_pregame(fixture, now):
                continue
            for opp in self.evaluate_fixture(fixture):
                key = (fixture.event_id, opp.market, opp.selection, opp.side)
                if self._is_new(key, opp.odds, opp.side):
                    self._seen[key] = opp.odds
                    hits.append(ScannerHit(
                        event_id=fixture.event_id, home=fixture.home,
                        away=fixture.away, commence_time=fixture.commence_time,
                        opportunity=opp,
                    ))
        hits.sort(key=lambda h: h.opportunity.ev, reverse=True)
        return hits

    def run(
        self,
        *,
        poll_interval: float = 60.0,
        max_polls: Optional[int] = None,
        on_hit: Optional[Callable[[ScannerHit], None]] = None,
        sleep_fn: Callable[[float], None] = _time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        """Run the scan loop. ``max_polls=None`` runs until interrupted.

        ``sleep_fn`` / ``now_fn`` are injectable for testing.
        """
        polls = 0
        while max_polls is None or polls < max_polls:
            for hit in self.poll_once(now=now_fn()):
                if on_hit:
                    on_hit(hit)
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            sleep_fn(poll_interval)


def odds_api_source(
    sport: str = "soccer_epl",
    *,
    api_key: Optional[str] = None,
    regions: str = "uk,eu",
) -> Callable[[], List[FixtureOdds]]:
    """Build an odds source backed by The Odds API (bookmaker back prices)."""
    from .data.odds_api import OddsApiClient

    client = OddsApiClient(api_key=api_key)

    def source() -> List[FixtureOdds]:
        fixtures = client.fetch_odds(sport=sport, regions=regions, markets="h2h,totals,btts")
        return [
            FixtureOdds(
                event_id=f.event_id, home=f.home, away=f.away,
                commence_time=f.commence_time, back=f.markets,
            )
            for f in fixtures
        ]

    return source
