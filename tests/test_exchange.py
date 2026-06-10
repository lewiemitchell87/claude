import pytest

from footyvalue.value import (
    back_payoff, lay_payoff, evaluate_selection,
    find_value_exchange, scan_markets_exchange,
)


def test_back_payoff_with_commission():
    # odds 3.0, 2% commission: net win = 2.0 * 0.98 = 1.96
    assert back_payoff(3.0, 0.02) == pytest.approx(1.96)


def test_lay_payoff_basics():
    # lay 2.0, no commission: win 1 unit per 1 unit liability
    assert lay_payoff(2.0, 0.0) == pytest.approx(1.0)
    # lay 5.0: liability 4 per 1 backer stake -> payoff per liability 0.25
    assert lay_payoff(5.0, 0.0) == pytest.approx(0.25)
    assert lay_payoff(3.0, 0.1) == pytest.approx(0.9 / 2.0)


def test_back_value_matches_legacy_formula():
    opp = evaluate_selection("match_odds", "home", 0.5, back_odds=2.2, min_ev=0.0)
    assert opp.side == "back"
    assert opp.ev == pytest.approx(0.10)        # 0.5*2.2 - 1
    assert opp.kelly == pytest.approx(0.10 / 1.2)


def test_lay_value_hand_computed():
    # Model prob the selection wins = 0.30; lay available at 2.0.
    # Lay wins when selection loses (prob 0.70); payoff per liability = 1.0.
    # EV = 0.70*1 - 0.30 = 0.40 ; edge = 1/2.0 - 0.30 = 0.20 ; kelly = 0.40.
    opp = evaluate_selection("match_odds", "home", 0.30, lay_odds=2.0, min_ev=0.0)
    assert opp.side == "lay"
    assert opp.ev == pytest.approx(0.40)
    assert opp.edge == pytest.approx(0.20)
    assert opp.kelly == pytest.approx(0.40)


def test_no_lay_value_when_price_too_short():
    # Lay at 1.9 implies the selection wins 1/1.9 = 52.6%, but the model says 55%.
    # The selection is *more* likely than the price implies, so laying is -EV.
    opp = evaluate_selection("match_odds", "home", 0.55, lay_odds=1.9, min_ev=0.0)
    assert opp is None


def test_picks_better_of_back_and_lay():
    # Strong back value (odds 5.0, model 0.30 -> EV 0.5) should beat any lay here.
    opp = evaluate_selection("x", "y", 0.30, back_odds=5.0, lay_odds=5.0, min_ev=0.0)
    assert opp.side == "back"
    assert opp.ev == pytest.approx(0.5)


def test_lay_liability_and_backer_stake_sizing():
    # bankroll 1000, kelly 0.40 capped at 0.25 -> risk 250 liability at lay 2.0.
    opp = evaluate_selection("m", "s", 0.30, lay_odds=2.0,
                             min_ev=0.0, kelly_cap=0.25, bankroll=1000)
    assert opp.stake == pytest.approx(250.0)             # liability
    assert opp.backer_stake == pytest.approx(250.0)      # /(2.0-1)


def test_find_value_exchange_combines_sides():
    model = {"home": 0.30, "draw": 0.30, "away": 0.40}
    back = {"away": 2.2}            # away back value: 0.40*2.2-1 = -0.12 (none)
    lay = {"home": 2.0}            # home lay value (as above)
    res = find_value_exchange("match_odds", model, back, lay, min_ev=0.0)
    sides = {(o.selection, o.side) for o in res}
    assert ("home", "lay") in sides


def test_scan_markets_exchange_sorts_by_ev():
    model = {"match_odds": {"home": 0.30, "draw": 0.30, "away": 0.40}}
    back = {"match_odds": {"away": 3.0}}     # 0.4*3-1 = 0.2
    lay = {"match_odds": {"home": 2.0}}      # ev 0.4
    res = scan_markets_exchange(model, back, lay, min_ev=0.0)
    evs = [o.ev for o in res]
    assert evs == sorted(evs, reverse=True)
