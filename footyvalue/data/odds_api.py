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
from typing import Dict, List, Optional

BASE_URL = "https://api.the-odds-api.com/v4"

# Map The Odds API market keys to our internal market names.
# 'totals' carries a point (line) we expand into over_under_<line>.
_H2H = "h2h"
_TOTALS = "totals"
_BTTS = "btts"


@dataclass
class Fixture:
    """A normalised fixture with aggregated best-price odds."""

    event_id: str
    home: str
    away: str
    commence_time: str
    markets: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _best(existing: Optional[float], candidate: float) -> float:
    return candidate if existing is None else max(existing, candidate)


def normalise_event(event: dict, home: str, away: str) -> Dict[str, Dict[str, float]]:
    """Turn one Odds API event payload into our ``{market: {selection: odds}}``."""
    markets: Dict[str, Dict[str, float]] = {}

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == _H2H:
                dest = markets.setdefault("match_odds", {})
                for o in outcomes:
                    name, price = o.get("name"), o.get("price")
                    if price is None:
                        continue
                    if name == home:
                        dest["home"] = _best(dest.get("home"), price)
                    elif name == away:
                        dest["away"] = _best(dest.get("away"), price)
                    elif name and name.lower() == "draw":
                        dest["draw"] = _best(dest.get("draw"), price)

            elif key == _TOTALS:
                for o in outcomes:
                    point = o.get("point")
                    name = (o.get("name") or "").lower()
                    price = o.get("price")
                    if point is None or price is None or name not in ("over", "under"):
                        continue
                    dest = markets.setdefault(f"over_under_{point}", {})
                    dest[name] = _best(dest.get(name), price)

            elif key == _BTTS:
                dest = markets.setdefault("btts", {})
                for o in outcomes:
                    name = (o.get("name") or "").lower()
                    price = o.get("price")
                    if price is None:
                        continue
                    if name in ("yes", "no"):
                        dest[name] = _best(dest.get(name), price)

    return markets


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
        timeout: float = 30.0,
    ) -> List[Fixture]:
        """Fetch and normalise upcoming/in-play odds for a competition.

        ``sport`` is a The Odds API sport key, e.g. ``soccer_epl``,
        ``soccer_uefa_champs_league``, ``soccer_spain_la_liga``.
        """
        import requests  # lazy

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
            fixtures.append(
                Fixture(
                    event_id=event.get("id", ""),
                    home=home,
                    away=away,
                    commence_time=event.get("commence_time", ""),
                    markets=normalise_event(event, home, away),
                )
            )
        return fixtures
