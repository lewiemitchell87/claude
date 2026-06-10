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


# --------------------------------------------------------------------------- #
# Odds extraction for backtesting
#
# football-data.co.uk publishes per-bookmaker odds plus market averages/maxima.
# Column conventions: <BOOK>H/D/A for match odds (e.g. B365H), <BOOK>>2.5 / <2.5
# for Over/Under 2.5, and a "C" infix marks closing odds (e.g. B365CH, AvgC>2.5).
# We look through a prioritised list of candidate columns so the loader tolerates
# the schema differences between seasons.
# --------------------------------------------------------------------------- #
# Bet (taken) price priority, then closing price priority.
_MO_BET_BOOKS = ["B365", "PS", "BW", "Avg", "Max"]
_MO_CLOSE_BOOKS = ["AvgC", "B365C", "PSC", "MaxC"]
_OU_BET_BOOKS = ["B365", "P", "Avg", "Max"]
_OU_CLOSE_BOOKS = ["AvgC", "B365C", "PC", "MaxC"]


def _first_float(row: dict, keys) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                f = float(v)
                if f > 1.0:
                    return f
            except ValueError:
                continue
    return None


def _extract_odds(row: dict, books_mo, books_ou) -> dict:
    """Pull a ``{market: {selection: odds}}`` block from one CSV row."""
    out: dict = {}

    mo = {
        "home": _first_float(row, [f"{b}H" for b in books_mo]),
        "draw": _first_float(row, [f"{b}D" for b in books_mo]),
        "away": _first_float(row, [f"{b}A" for b in books_mo]),
    }
    mo = {k: v for k, v in mo.items() if v is not None}
    if len(mo) == 3:
        out["match_odds"] = mo

    over = _first_float(row, [f"{b}>2.5" for b in books_ou])
    under = _first_float(row, [f"{b}<2.5" for b in books_ou])
    if over is not None and under is not None:
        out["over_under_2.5"] = {"over": over, "under": under}

    return out


def load_backtest_matches(path: str, columns: Optional[dict] = None) -> List[dict]:
    """Load results **with odds** for the backtester.

    Returns a list of dicts shaped for :func:`footyvalue.backtest.run_backtest`:
    ``{date, home, away, home_goals, away_goals, odds, closing}``. Rows without
    usable result fields are skipped; markets without odds are simply omitted.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    out: List[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                hg = row.get(cols["home_goals"], "")
                ag = row.get(cols["away_goals"], "")
                home = (row.get(cols["home"]) or "").strip()
                away = (row.get(cols["away"]) or "").strip()
                if not home or not away or hg == "" or ag == "":
                    continue
                record = {
                    "date": Match.parse_date(row.get(cols["date"])),
                    "home": home,
                    "away": away,
                    "home_goals": int(float(hg)),
                    "away_goals": int(float(ag)),
                    "odds": _extract_odds(row, _MO_BET_BOOKS, _OU_BET_BOOKS),
                    "closing": _extract_odds(row, _MO_CLOSE_BOOKS, _OU_CLOSE_BOOKS),
                }
                out.append(record)
            except (ValueError, AttributeError):
                continue
    return out
