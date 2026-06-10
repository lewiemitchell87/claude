"""Load historical results from CSV (football-data.co.uk format or generic).

football-data.co.uk publishes free CSVs for most major leagues with columns
including ``Date``, ``HomeTeam``, ``AwayTeam``, ``FTHG`` (full-time home goals)
and ``FTAG`` (full-time away goals). This loader reads those, and also accepts a
generic schema via column overrides.
"""

from __future__ import annotations

import csv
import io
from typing import List, Optional

from ..ratings import Match

# Default column names (football-data.co.uk).
DEFAULT_COLUMNS = {
    "date": "Date",
    "home": "HomeTeam",
    "away": "AwayTeam",
    "home_goals": "FTHG",
    "away_goals": "FTAG",
}


def _rows_to_matches(rows, columns) -> List[Match]:
    matches: List[Match] = []
    for row in rows:
        try:
            hg = row.get(columns["home_goals"], "")
            ag = row.get(columns["away_goals"], "")
            home = row.get(columns["home"], "")
            away = row.get(columns["away"], "")
            if home == "" or away == "" or hg == "" or ag == "":
                continue
            matches.append(
                Match(
                    home=home.strip(),
                    away=away.strip(),
                    home_goals=int(float(hg)),
                    away_goals=int(float(ag)),
                    match_date=Match.parse_date(row.get(columns["date"])),
                )
            )
        except (ValueError, AttributeError):
            # Skip malformed / incomplete rows (e.g. trailing blank lines).
            continue
    return matches


def load_matches_from_string(text: str, columns: Optional[dict] = None) -> List[Match]:
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    reader = csv.DictReader(io.StringIO(text))
    return _rows_to_matches(reader, columns)


def load_matches_csv(path: str, columns: Optional[dict] = None) -> List[Match]:
    """Load matches from a local CSV file."""
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return _rows_to_matches(reader, columns)


def load_matches_url(url: str, columns: Optional[dict] = None, timeout: float = 30.0) -> List[Match]:
    """Load matches from a CSV URL (e.g. a football-data.co.uk season file).

    ``requests`` is imported lazily so the core library has no hard HTTP
    dependency.
    """
    import requests  # lazy

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return load_matches_from_string(resp.text, columns)
