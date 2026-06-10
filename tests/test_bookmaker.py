import pytest

from footyvalue.data.odds_api import aggregate_event, normalise_event
from footyvalue.value import find_value, evaluate_selection, find_value_exchange
from footyvalue.ratings import TeamRatings
from footyvalue.scanner import PreGameScanner, FixtureOdds
from datetime import datetime, timedelta, timezone


def _event():
    return {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "bookmakers": [
            {"key": "bet365", "title": "Bet365", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Arsenal", "price": 1.90},
                    {"name": "Chelsea", "price": 4.20},
                    {"name": "Draw", "price": 3.60},
                ]},
            ]},
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Arsenal", "price": 2.05},   # best Arsenal price
                    {"name": "Chelsea", "price": 4.00},
                    {"name": "Draw", "price": 3.50},
                ]},
            ]},
        ],
    }


def test_aggregate_records_best_price_and_book():
    prices, sources = aggregate_event(_event(), "Arsenal", "Chelsea")
    assert prices["match_odds"]["home"] == 2.05
    assert sources["match_odds"]["home"] == "Pinnacle"   # who offered the best
    assert sources["match_odds"]["away"] == "Bet365"     # Bet365's 4.20 beat 4.00


def test_normalise_event_still_returns_prices_only():
    prices = normalise_event(_event(), "Arsenal", "Chelsea")
    assert prices["match_odds"]["home"] == 2.05
    assert isinstance(prices["match_odds"]["home"], float)


def test_bookmaker_filter_restricts_books():
    prices, sources = aggregate_event(_event(), "Arsenal", "Chelsea", allowed_books={"bet365"})
    assert prices["match_odds"]["home"] == 1.90         # Pinnacle excluded
    assert sources["match_odds"]["home"] == "Bet365"


def test_filter_matches_on_title_too():
    prices, sources = aggregate_event(_event(), "Arsenal", "Chelsea", allowed_books={"pinnacle"})
    assert sources["match_odds"]["home"] == "Pinnacle"
    assert prices["match_odds"]["home"] == 2.05


def test_find_value_attaches_bookmaker():
    res = find_value("match_odds", {"home": 0.55},
                     {"home": 2.10}, sources={"home": "Bet365"}, min_ev=0.0)
    assert res[0].bookmaker == "Bet365"


def test_evaluate_selection_back_bookmaker():
    opp = evaluate_selection("match_odds", "home", 0.55,
                             back_odds=2.10, back_bookmaker="Pinnacle", min_ev=0.0)
    assert opp.side == "back"
    assert opp.bookmaker == "Pinnacle"


def test_find_value_exchange_carries_back_source():
    res = find_value_exchange("match_odds", {"home": 0.55},
                              back_odds={"home": 2.10},
                              back_sources={"home": "William Hill"}, min_ev=0.0)
    assert res[0].bookmaker == "William Hill"


def test_scanner_emits_bookmaker():
    ratings = TeamRatings(attack={"A": 0.3, "B": -0.3}, defence={"A": 0.15, "B": -0.15},
                          home_advantage=0.25, rho=0.0, teams=["A", "B"])
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    fx = FixtureOdds("e1", "A", "B", future,
                     back={"match_odds": {"home": 3.0}},
                     back_sources={"match_odds": {"home": "Bet365"}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02)
    hits = scanner.poll_once()
    assert hits and hits[0].opportunity.bookmaker == "Bet365"
