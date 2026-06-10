import os

import pytest

from footyvalue.ratings import fit_dixon_coles
from footyvalue.data.football_data import load_matches_csv
from footyvalue.data.odds_api import normalise_event
from footyvalue.engine import ScanConfig, evaluate_fixture, scan_fixtures

HISTORY = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_history.csv")


@pytest.fixture(scope="module")
def ratings():
    return fit_dixon_coles(load_matches_csv(HISTORY), xi=0.0, fit_rho=True)


def test_evaluate_fixture_flags_overpriced_selection(ratings):
    # Foxes (away Eagles) model away prob ~0.53; price it well above fair.
    odds = {"match_odds": {"home": 5.0, "draw": 3.6, "away": 2.10}}
    result = evaluate_fixture(ratings, "Foxes", "Eagles", odds, ScanConfig(min_ev=0.02))
    sels = {o.selection for o in result.opportunities}
    assert "away" in sels
    assert all(o.ev >= 0.02 for o in result.opportunities)


def test_evaluate_fixture_no_value_on_fair_book(ratings):
    # Price exactly at the model's fair odds => no value should be flagged.
    mk = ratings.market_probabilities("Wolves", "Bears")["match_odds"]
    fair_odds = {"match_odds": {k: 1.0 / v for k, v in mk.items()}}
    result = evaluate_fixture(ratings, "Wolves", "Bears", fair_odds, ScanConfig(min_ev=0.01))
    assert result.opportunities == []


def test_inplay_evaluation_uses_score(ratings):
    odds = {"match_odds": {"home": 1.15, "draw": 12.0, "away": 60.0}}
    result = evaluate_fixture(
        ratings, "Eagles", "Foxes", odds, ScanConfig(min_ev=0.02),
        minute=70, home_goals=1, away_goals=0,
    )
    assert result.minute == 70
    assert result.score == (1, 0)
    # Home leading late and strong => high model home prob.
    assert result.model_markets["match_odds"]["home"] > 0.8


def test_scan_fixtures_sorts_by_best_ev(ratings):
    fixtures = [
        {"home": "Foxes", "away": "Eagles",
         "markets": {"match_odds": {"home": 5.0, "draw": 3.6, "away": 2.10}}},
        {"home": "Wolves", "away": "Bears",
         "markets": {"match_odds": {"home": 4.4, "draw": 3.5, "away": 1.95}}},
    ]
    results = scan_fixtures(ratings, fixtures, ScanConfig(min_ev=0.0))
    best = [r.opportunities[0].ev if r.opportunities else float("-inf") for r in results]
    assert best == sorted(best, reverse=True)


def test_odds_api_normalisation():
    event = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {"markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Arsenal", "price": 1.90},
                    {"name": "Chelsea", "price": 4.20},
                    {"name": "Draw", "price": 3.60},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.95, "point": 2.5},
                    {"name": "Under", "price": 1.85, "point": 2.5},
                ]},
                {"key": "btts", "outcomes": [
                    {"name": "Yes", "price": 1.80},
                    {"name": "No", "price": 1.95},
                ]},
            ]},
            {"markets": [  # second book offers a better Arsenal price
                {"key": "h2h", "outcomes": [
                    {"name": "Arsenal", "price": 2.00},
                    {"name": "Chelsea", "price": 4.00},
                    {"name": "Draw", "price": 3.50},
                ]},
            ]},
        ],
    }
    markets = normalise_event(event, "Arsenal", "Chelsea")
    assert markets["match_odds"]["home"] == 2.00  # best price taken
    assert markets["match_odds"]["draw"] == 3.60
    assert markets["over_under_2.5"] == {"over": 1.95, "under": 1.85}
    assert markets["btts"] == {"yes": 1.80, "no": 1.95}
