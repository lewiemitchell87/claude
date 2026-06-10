from datetime import datetime, timedelta, timezone

import pytest

from footyvalue.ratings import TeamRatings
from footyvalue.scanner import PreGameScanner, FixtureOdds, _parse_iso


@pytest.fixture
def ratings():
    return TeamRatings(
        attack={"A": 0.30, "B": -0.30},
        defence={"A": 0.15, "B": -0.15},
        home_advantage=0.25,
        rho=0.0,
        teams=["A", "B"],
    )


def _future(seconds=86400):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _past(seconds=3600):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_parse_iso_handles_z_suffix():
    dt = _parse_iso("2026-06-10T14:00:00Z")
    assert dt.tzinfo is not None


def test_emits_value_for_generous_back_price(ratings):
    # A strong home team priced generously should be flagged.
    fx = FixtureOdds("e1", "A", "B", _future(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02)
    hits = scanner.poll_once()
    assert any(h.opportunity.selection == "home" for h in hits)


def test_dedup_suppresses_repeat(ratings):
    fx = FixtureOdds("e1", "A", "B", _future(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02)
    first = scanner.poll_once()
    second = scanner.poll_once()
    assert first and second == []


def test_reemits_on_improved_price(ratings):
    state = {"odds": 3.0}
    fx_source = lambda: [FixtureOdds("e1", "A", "B", _future(),
                                     back={"match_odds": {"home": state["odds"]}})]
    scanner = PreGameScanner(ratings, fx_source, min_ev=0.02)
    assert scanner.poll_once()           # initial hit
    assert scanner.poll_once() == []     # same price, deduped
    state["odds"] = 3.5                  # better back price
    assert scanner.poll_once()           # re-alerted


def test_skips_started_fixtures(ratings):
    fx = FixtureOdds("e1", "A", "B", _past(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02)
    assert scanner.poll_once() == []


def test_min_seconds_to_start_filter(ratings):
    fx = FixtureOdds("e1", "A", "B", _future(60), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02,
                             min_seconds_to_start=300)
    assert scanner.poll_once() == []     # starts in 60s, threshold 300s


def test_skips_unknown_teams(ratings):
    fx = FixtureOdds("e1", "Unknown", "B", _future(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02, skip_unknown_teams=True)
    assert scanner.poll_once() == []


def test_team_aliases_resolve(ratings):
    fx = FixtureOdds("e1", "Team A FC", "B", _future(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02,
                             team_aliases={"Team A FC": "A"})
    assert scanner.poll_once()


def test_run_loop_invokes_callback_and_sleep(ratings):
    fx = FixtureOdds("e1", "A", "B", _future(), back={"match_odds": {"home": 3.0}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.02)
    seen = []
    sleeps = []
    scanner.run(poll_interval=5, max_polls=2,
                on_hit=lambda h: seen.append(h),
                sleep_fn=lambda s: sleeps.append(s))
    assert len(seen) == 1            # only first poll yields a new hit
    assert sleeps == [5]            # slept once between the two polls


def test_lay_value_via_exchange_feed(ratings):
    # Provide a lay price that makes laying the (weak) away side attractive.
    model = ratings.market_probabilities("A", "B")["match_odds"]
    # Pick a lay price just above the model's implied fair for 'away'.
    lay_price = round(1.0 / model["away"] * 0.85, 2)  # generous (low) lay = value
    fx = FixtureOdds("e1", "A", "B", _future(),
                     lay={"match_odds": {"away": lay_price}})
    scanner = PreGameScanner(ratings, lambda: [fx], min_ev=0.0, commission=0.02)
    hits = scanner.poll_once()
    assert any(h.opportunity.side == "lay" and h.opportunity.selection == "away"
               for h in hits)
