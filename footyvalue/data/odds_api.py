"""Live odds via The Odds API (the-odds-api.com).

The Odds API offers a free tier covering soccer with Match Odds (``h2h``),
Over/Under (``totals``) and Both Teams To Score (``btts``) markets across many
bookmakers. This adapter fetches odds and normalises them into the engine's
shape::

    {
      "match_odds":      {"home": 2.10, "draw": 3.40, "away": 3.60},
      "btts":            {"yes": 1.90, "no": 1.85},
      "over_under_2.5":  {"over": 1.95, "under": 1.90},
      ...
    }

Bookmaker odds are aggregated by taking the *best* (highest) price available for
each selection, since that maximises any value edge — though you can override the
aggregation.

Set the API key via the ``ODDS_API_KEY`` environment variable or pass it in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

BASE_URL = "https://api.the-odds-api.com/v4"

# Map The Odds API market keys to our internal market names.
# 'totals' carries a point (line) we expand into over_under_<line>.
_H2H = "h2h"
_TOTALS = "totals"
_BTTS = "btts"


@dataclass
class Fixture:
    """A normalised fixture with aggregated best-price odds.

    ``markets`` holds ``{market: {selection: best decimal odds}}`` and ``sources``
    holds the matching ``{market: {selection: bookmaker title}}`` so you know
    *where* each price is available.
    """

    event_id: str
    home: str
    away: str
    commence_time: str
    markets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)


def _book_allowed(book: dict, allowed: Optional[set]) -> bool:
    if not allowed:
        return True
    return (book.get("key", "").lower() in allowed
            or (book.get("title") or "").lower() in allowed)


def _consider(prices, sources, market_key, selection, price, book_name):
    """Keep the best price for a selection and record which book offered it."""
    if price is None:
        return
    cur = prices.get(market_key, {}).get(selection)
    if cur is None or price > cur:
        prices.setdefault(market_key, {})[selection] = price
        sources.setdefault(market_key, {})[selection] = book_name


def aggregate_event(event: dict, home: str, away: str, allowed_books: Optional[set] = None):
    """Aggregate an Odds API event into ``(prices, sources)``.

    ``prices``  -> ``{market: {selection: best decimal odds}}``
    ``sources`` -> ``{market: {selection: bookmaker title}}``

    ``allowed_books`` (a set of lowercased keys or titles) restricts which
    bookmakers are considered.
    """
    prices: Dict[str, Dict[str, float]] = {}
    sources: Dict[str, Dict[str, str]] = {}

    for book in event.get("bookmakers", []):
        if not _book_allowed(book, allowed_books):
            continue
        book_name = book.get("title") or book.get("key") or "?"
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == _H2H:
                for o in outcomes:
                    name, price = o.get("name"), o.get("price")
                    if name == home:
                        _consider(prices, sources, "match_odds", "home", price, book_name)
                    elif name == away:
                        _consider(prices, sources, "match_odds", "away", price, book_name)
                    elif name and name.lower() == "draw":
                        _consider(prices, sources, "match_odds", "draw", price, book_name)

            elif key == _TOTALS:
                for o in outcomes:
                    point = o.get("point")
                    name = (o.get("name") or "").lower()
                    if point is None or name not in ("over", "under"):
                        continue
                    _consider(prices, sources, f"over_under_{point}", name, o.get("price"), book_name)

            elif key == _BTTS:
                for o in outcomes:
                    name = (o.get("name") or "").lower()
                    if name in ("yes", "no"):
                        _consider(prices, sources, "btts", name, o.get("price"), book_name)

    return prices, sources


def normalise_event(event: dict, home: str, away: str) -> Dict[str, Dict[str, float]]:
    """Best-price ``{market: {selection: odds}}`` for one event (prices only)."""
    prices, _ = aggregate_event(event, home, away)
    return prices


class OddsApiClient:
    """Thin client for The Odds API soccer odds."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = BASE_URL):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError(
                "No Odds API key. Set ODDS_API_KEY or pass api_key=... "
                "(get a free key at https://the-odds-api.com)."
            )

    def fetch_odds(
        self,
        sport: str = "soccer_epl",
        regions: str = "uk,eu",
        markets: str = "h2h,totals,btts",
        odds_format: str = "decimal",
        bookmakers: Optional[Sequence[str]] = None,
        timeout: float = 30.0,
    ) -> List[Fixture]:
        """Fetch and normalise upcoming/in-play odds for a competition.

        ``sport`` is a The Odds API sport key, e.g. ``soccer_epl``,
        ``soccer_uefa_champs_league``, ``soccer_spain_la_liga``.

        ``bookmakers`` optionally restricts which books are considered (match by
        Odds API key, e.g. ``"bet365"``, or by title, e.g. ``"Bet365"``) — useful
        for limiting to books you actually hold accounts with.
        """
        import requests  # lazy

        allowed = {b.strip().lower() for b in bookmakers} if bookmakers else None

        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        events = resp.json()

        fixtures: List[Fixture] = []
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            prices, sources = aggregate_event(event, home, away, allowed)
            fixtures.append(
                Fixture(
                    event_id=event.get("id", ""),
                    home=home,
                    away=away,
                    commence_time=event.get("commence_time", ""),
                    markets=prices,
                    sources=sources,
                )
            )
        return fixtures
