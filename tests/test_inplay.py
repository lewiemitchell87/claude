import pytest

from footyvalue.scoreline import score_matrix
from footyvalue.markets import all_markets, match_odds, over_under, btts
from footyvalue.inplay import inplay_markets, remaining_fraction


def test_remaining_fraction_linear():
    assert remaining_fraction(0) == pytest.approx(1.0)
    assert remaining_fraction(45) == pytest.approx(0.5)
    assert remaining_fraction(90) == pytest.approx(0.0)
    assert remaining_fraction(120, total_minutes=90) == pytest.approx(0.0)


def test_inplay_at_kickoff_matches_pregame():
    lam_h, lam_a = 1.7, 1.1
    live = inplay_markets(lam_h, lam_a, minute=0, home_goals=0, away_goals=0)
    pre = all_markets(score_matrix(lam_h, lam_a))
    for sel in ("home", "draw", "away"):
        assert live["match_odds"][sel] == pytest.approx(pre["match_odds"][sel], abs=1e-9)
    assert live["btts"]["yes"] == pytest.approx(pre["btts"]["yes"], abs=1e-9)
    assert live["over_under_2.5"]["over"] == pytest.approx(pre["over_under_2.5"]["over"], abs=1e-9)


def test_leading_team_win_prob_rises_late():
    lam_h, lam_a = 1.5, 1.3
    early = inplay_markets(lam_h, lam_a, minute=20, home_goals=1, away_goals=0)
    late = inplay_markets(lam_h, lam_a, minute=80, home_goals=1, away_goals=0)
    assert late["match_odds"]["home"] > early["match_odds"]["home"]


def test_full_time_resolved_markets_are_certain():
    # 2-1 at minute 90: home win certain, BTTS yes certain, Over 2.5 certain.
    live = inplay_markets(1.5, 1.2, minute=90, home_goals=2, away_goals=1)
    assert live["match_odds"]["home"] == pytest.approx(1.0, abs=1e-9)
    assert live["match_odds"]["draw"] == pytest.approx(0.0, abs=1e-9)
    assert live["btts"]["yes"] == pytest.approx(1.0, abs=1e-9)
    assert live["over_under_2.5"]["over"] == pytest.approx(1.0, abs=1e-9)


def test_btts_yes_certain_once_both_scored():
    live = inplay_markets(1.4, 1.2, minute=55, home_goals=1, away_goals=1)
    assert live["btts"]["yes"] == pytest.approx(1.0, abs=1e-9)


def test_over_already_decided_when_three_scored():
    live = inplay_markets(1.4, 1.2, minute=70, home_goals=2, away_goals=1)
    assert live["over_under_2.5"]["over"] == pytest.approx(1.0, abs=1e-9)


def test_inplay_probabilities_normalised():
    live = inplay_markets(1.6, 1.0, minute=37, home_goals=0, away_goals=1)
    mo = live["match_odds"]
    assert mo["home"] + mo["draw"] + mo["away"] == pytest.approx(1.0, abs=1e-9)
    ou = live["over_under_1.5"]
    assert ou["over"] + ou["under"] + ou["push"] == pytest.approx(1.0, abs=1e-9)
