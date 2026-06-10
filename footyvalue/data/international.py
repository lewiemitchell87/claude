"""Loader for international football results (martj42/international_results).

That open dataset lists every international since 1872 and is updated continually,
with columns::

    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral

It carries a ``neutral`` flag (vital for tournaments played on neutral ground)
and includes *future* fixtures with empty scores, which we skip when fitting.

Use this to build live-ready ratings for international matches (e.g. the World
Cup), since domestic-league history won't contain national teams.
"""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import List, Optional, Sequence

from ..ratings import Match

DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _rows_to_matches(
    rows,
    *,
    from_date: Optional[date],
    exclude_friendlies: bool,
    tournaments: Optional[set],
) -> List[Match]:
    matches: List[Match] = []
    for row in rows:
        hs = row.get("home_score", "")
        as_ = row.get("away_score", "")
        if hs in ("", "NA") or as_ in ("", "NA"):
            continue  # future / unplayed fixture
        d = Match.parse_date(row.get("date"))
        if from_date is not None and (d is None or d < from_date):
            continue
        tournament = (row.get("tournament") or "").strip()
        if exclude_friendlies and tournament.lower() == "friendly":
            continue
        if tournaments is not None and tournament not in tournaments:
            continue
        try:
            matches.append(
                Match(
                    home=(row.get("home_team") or "").strip(),
                    away=(row.get("away_team") or "").strip(),
                    home_goals=int(float(hs)),
                    away_goals=int(float(as_)),
                    match_date=d,
                    neutral=_parse_bool(row.get("neutral", "")),
                )
            )
        except (ValueError, AttributeError):
            continue
    return matches


def load_international_from_string(
    text: str,
    *,
    from_date: Optional[date] = None,
    exclude_friendlies: bool = False,
    tournaments: Optional[Sequence[str]] = None,
) -> List[Match]:
    reader = csv.DictReader(io.StringIO(text))
    return _rows_to_matches(
        reader,
        from_date=from_date,
        exclude_friendlies=exclude_friendlies,
        tournaments=set(tournaments) if tournaments else None,
    )


def load_international_csv(path: str, **kwargs) -> List[Match]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return load_international_from_string(fh.read(), **kwargs)


def load_international_url(url: str = DATASET_URL, timeout: float = 60.0, **kwargs) -> List[Match]:
    """Download and parse the international results dataset.

    ``requests`` is imported lazily so the core library has no hard HTTP
    dependency.
    """
    import requests  # lazy

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return load_international_from_string(resp.text, **kwargs)
