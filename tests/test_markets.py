import numpy as np
import pytest

from footyvalue.scoreline import score_matrix
from footyvalue.markets import (
    match_odds,
    over_under,
    btts,
    total_goals_distribution,
    correct_score,
    all_markets,
)


def test_match_odds_sum_to_one():
    m = score_matrix(1.5, 1.2)
    mo = match_odds(m)
    assert mo["home"] + mo["draw"] + mo["away"] == pytest.approx(1.0, abs=1e-12)


def test_home_advantage_makes_home_favourite():
    # Strictly higher home expectation => home win prob exceeds away win prob.
    m = score_matrix(1.8, 1.0)
    mo = match_odds(m)
    assert mo["home"] > mo["away"]


def test_symmetric_teams_have_equal_win_probs():
    m = score_matrix(1.3, 1.3)
    mo = match_odds(m)
    assert mo["home"] == pytest.approx(mo["away"], abs=1e-12)


def test_over_under_half_line_has_no_push():
    m = score_matrix(1.4, 1.1)
    ou = over_under(m, 2.5)
    assert ou["push"] == pytest.approx(0.0, abs=1e-12)
    assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=1e-12)


def test_over_under_whole_line_push_equals_exact_total():
    m = score_matrix(1.4, 1.1)
    totals = total_goals_distribution(m)
    ou = over_under(m, 2.0)
    assert ou["push"] == pytest.approx(totals[2], abs=1e-12)
    assert ou["over"] + ou["under"] + ou["push"] == pytest.approx(1.0, abs=1e-12)


def test_over_monotonic_in_expected_goals():
    low = over_under(score_matrix(0.8, 0.7), 2.5)["over"]
    high = over_under(score_matrix(2.5, 2.2), 2.5)["over"]
    assert high > low


def test_btts_complementary_and_reasonable():
    m = score_matrix(1.5, 1.3)
    b = btts(m)
    assert b["yes"] + b["no"] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < b["yes"] < 1.0


def test_btts_low_scoring_match_favours_no():
    m = score_matrix(0.4, 0.4)
    assert btts(m)["no"] > btts(m)["yes"]


def test_correct_score_covers_full_probability():
    m = score_matrix(1.6, 1.2)
    cs = correct_score(m, max_each=5)
    assert sum(cs.values()) == pytest.approx(1.0, abs=1e-9)


def test_all_markets_contains_expected_keys():
    m = score_matrix(1.5, 1.2)
    am = all_markets(m)
    assert "match_odds" in am and "btts" in am
    assert "over_under_2.5" in am
