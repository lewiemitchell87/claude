import pytest

from footyvalue.value import (
    expected_value,
    edge,
    kelly_fraction,
    fair_odds,
    find_value,
    scan_markets,
)
from footyvalue import odds as oddsmod


def test_expected_value_zero_at_fair_odds():
    p = 0.5
    assert expected_value(p, 1.0 / p) == pytest.approx(0.0)


def test_expected_value_positive_when_overpriced():
    # True prob 0.5 but offered 2.20 => +EV.
    assert expected_value(0.5, 2.20) == pytest.approx(0.10)


def test_edge_definition():
    assert edge(0.55, 2.0) == pytest.approx(0.05)


def test_kelly_fraction_known_value():
    # p=0.6, odds=2.0 => f = (0.6*2 - 1)/(2-1) = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_never_negative_for_bad_bet():
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_find_value_flags_only_positive_ev():
    model = {"home": 0.55, "draw": 0.25, "away": 0.20}
    market = {"home": 2.10, "draw": 3.40, "away": 4.50}
    # home: 0.55*2.10-1 = 0.155 (+EV); draw: 0.25*3.40-1=-0.15; away:0.20*4.5-1=-0.10
    res = find_value("match_odds", model, market, min_ev=0.02)
    assert len(res) == 1
    assert res[0].selection == "home"
    assert res[0].ev == pytest.approx(0.155)


def test_find_value_respects_min_ev_threshold():
    model = {"yes": 0.51}
    market = {"yes": 2.0}  # EV = 0.02 exactly
    assert len(find_value("btts", model, market, min_ev=0.02)) == 1
    assert len(find_value("btts", model, market, min_ev=0.05)) == 0


def test_find_value_stake_uses_fractional_kelly():
    model = {"home": 0.6}
    market = {"home": 2.0}  # full kelly = 0.2
    res = find_value("match_odds", model, market, kelly_cap=0.1, bankroll=1000)
    assert res[0].kelly == pytest.approx(0.2)
    assert res[0].stake == pytest.approx(100.0)  # capped at 0.1 * 1000


def test_find_value_ignores_invalid_odds():
    model = {"home": 0.6}
    assert find_value("match_odds", model, {"home": 1.0}) == []
    assert find_value("match_odds", model, {"home": 0.0}) == []


def test_scan_markets_aggregates_and_sorts():
    model = {
        "match_odds": {"home": 0.55, "draw": 0.25, "away": 0.20},
        "btts": {"yes": 0.6, "no": 0.4},
    }
    odds = {
        "match_odds": {"home": 2.10, "draw": 3.40, "away": 4.50},
        "btts": {"yes": 2.0, "no": 1.7},
    }
    res = scan_markets(model, odds, min_ev=0.02)
    evs = [r.ev for r in res]
    assert evs == sorted(evs, reverse=True)
    assert {r.selection for r in res} == {"home", "yes"}


def test_overround_removal_proportional_sums_to_one():
    fair = oddsmod.fair_probabilities({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert sum(fair.values()) == pytest.approx(1.0)


def test_overround_shin_sums_to_one():
    fair = oddsmod.fair_probabilities({"home": 2.0, "draw": 3.5, "away": 4.0}, method="shin")
    assert sum(fair.values()) == pytest.approx(1.0, abs=1e-9)


def test_fair_odds_inverse():
    assert fair_odds(0.25) == pytest.approx(4.0)
