import os
from datetime import date, timedelta

import pytest

from footyvalue.backtest import (
    settle, bet_profit, closing_value, run_backtest,
)
from footyvalue.value import ValueOpportunity
from footyvalue.data.football_data import load_backtest_matches

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_backtest.csv")


# -- settlement ------------------------------------------------------------- #
def test_settle_match_odds():
    assert settle("match_odds", "home", 2, 1) is True
    assert settle("match_odds", "draw", 2, 1) is False
    assert settle("match_odds", "away", 1, 2) is True
    assert settle("match_odds", "draw", 1, 1) is True


def test_settle_btts():
    assert settle("btts", "yes", 1, 1) is True
    assert settle("btts", "no", 1, 0) is True
    assert settle("btts", "yes", 0, 3) is False


def test_settle_over_under_half_line():
    assert settle("over_under_2.5", "over", 2, 1) is True   # total 3
    assert settle("over_under_2.5", "under", 1, 1) is True  # total 2


def test_settle_whole_line_push():
    assert settle("over_under_2.0", "over", 1, 1) is None   # exactly 2 -> push


# -- profit / CLV ----------------------------------------------------------- #
def _back(stake=10.0, odds=2.5, comm=0.0):
    return ValueOpportunity("m", "s", 0.5, odds, 2.0, 0.4, 0.1, 0.25, 0.0,
                            stake=stake, side="back", commission=comm)


def _lay(stake=10.0, odds=2.0, comm=0.0):
    backer = stake / (odds - 1.0)
    return ValueOpportunity("m", "s", 0.3, odds, 0.0, 0.0, 0.0, 0.0, 0.0,
                            stake=stake, side="lay", commission=comm,
                            backer_stake=backer)


def test_back_profit():
    assert bet_profit(_back(10, 2.5), True) == pytest.approx(15.0)   # 10*1.5
    assert bet_profit(_back(10, 2.5), False) == pytest.approx(-10.0)
    assert bet_profit(_back(10, 2.5), None) == 0.0


def test_back_profit_with_commission():
    assert bet_profit(_back(10, 3.0, comm=0.05), True) == pytest.approx(10 * 2.0 * 0.95)


def test_lay_profit():
    opp = _lay(stake=10.0, odds=2.0)  # liability 10, backer 10
    assert bet_profit(opp, False) == pytest.approx(10.0)   # selection lost -> we win backer stake
    assert bet_profit(opp, True) == pytest.approx(-10.0)   # selection won -> lose liability


def test_clv_directionality():
    back = _back(odds=2.10)
    assert closing_value(back, 2.0) == pytest.approx(0.05)   # got better back price
    lay = _lay(odds=2.0)
    assert closing_value(lay, 2.2) == pytest.approx(0.10)    # layed lower than close = good
    assert closing_value(back, None) is None


# -- driver ----------------------------------------------------------------- #
def test_no_bets_before_min_train():
    matches = load_backtest_matches(DATA)
    res = run_backtest(matches, min_train_matches=10_000)
    assert res.n_bets == 0
    assert res.final_bankroll == res.starting_bankroll


def test_backtest_runs_and_is_internally_consistent():
    matches = load_backtest_matches(DATA)
    res = run_backtest(matches, min_train_matches=56, refit_every=20,
                       min_ev=0.03, staking="flat", stake_unit=1.0)
    assert res.n_bets > 0
    # Profit equals sum of individual bet profits.
    assert res.total_profit == pytest.approx(sum(b.profit for b in res.bets))
    # Final bankroll = start + total profit.
    assert res.final_bankroll == pytest.approx(res.starting_bankroll + res.total_profit)
    # Equity curve length = bets + 1.
    assert len(res.equity_curve) == res.n_bets + 1
    # CLV is populated (the dataset carries closing prices).
    assert res.avg_clv is not None


def test_backtest_walk_forward_has_no_lookahead_signal():
    # If we never refit (min_train huge), there can be no bets -> guards the
    # ordering/training logic from accidentally using the current match.
    matches = load_backtest_matches(DATA)
    res = run_backtest(matches, min_train_matches=len(matches) + 1)
    assert res.n_bets == 0


def test_loader_extracts_odds_and_closing():
    matches = load_backtest_matches(DATA)
    assert matches
    m = matches[0]
    assert "match_odds" in m["odds"]
    assert set(m["odds"]["match_odds"]) == {"home", "draw", "away"}
    assert "match_odds" in m["closing"]
