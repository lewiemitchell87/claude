import os

import pytest

from footyvalue.ratings import Match, TeamRatings, fit_dixon_coles
from footyvalue.data.football_data import load_matches_csv, load_matches_from_string

HISTORY = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_history.csv")


@pytest.fixture(scope="module")
def ratings():
    matches = load_matches_csv(HISTORY)
    return fit_dixon_coles(matches, xi=0.0, fit_rho=True)


def test_fit_recovers_team_ordering(ratings):
    # In the synthetic data the Lions are strongest and the Moles weakest.
    strengths = {t: ratings.attack[t] - 0 for t in ratings.teams}
    assert ratings.attack["Lions"] > ratings.attack["Moles"]
    assert ratings.defence["Lions"] > ratings.defence["Moles"]
    # Best attacking xG order: Lions should have the highest expected goals at home.
    lam_lions, _ = ratings.expected_goals("Lions", "Moles")
    lam_moles, _ = ratings.expected_goals("Moles", "Lions")
    assert lam_lions > lam_moles


def test_fit_recovers_home_advantage(ratings):
    # True home advantage in the generator is +0.28 (log-goals).
    assert 0.15 < ratings.home_advantage < 0.45


def test_expected_goals_unknown_team_is_average(ratings):
    lam_h, lam_a = ratings.expected_goals("NobodyFC", "AlsoNobody")
    # Two average teams: home expectation exceeds away purely via home advantage.
    assert lam_h > lam_a


def test_market_probabilities_consistent(ratings):
    mk = ratings.market_probabilities("Lions", "Moles")
    mo = mk["match_odds"]
    assert mo["home"] + mo["draw"] + mo["away"] == pytest.approx(1.0, abs=1e-9)
    # Heavy favourite at home should be priced as such.
    assert mo["home"] > 0.7


def test_save_and_load_roundtrip(ratings, tmp_path):
    path = tmp_path / "r.json"
    ratings.save(str(path))
    loaded = TeamRatings.load(str(path))
    assert loaded.attack == ratings.attack
    assert loaded.home_advantage == pytest.approx(ratings.home_advantage)
    assert loaded.rho == pytest.approx(ratings.rho)


def test_time_decay_weights_recent_matches():
    # Two teams; older results favour A, recent results favour B. With strong
    # decay the ratings should lean towards the recent (B-favouring) outcomes.
    from datetime import date, timedelta

    base = date(2024, 1, 1)
    matches = []
    for i in range(10):  # old: A thrashes B
        matches.append(Match("A", "B", 3, 0, base + timedelta(days=i)))
    for i in range(10):  # recent: B thrashes A
        matches.append(Match("A", "B", 0, 3, base + timedelta(days=200 + i)))

    decayed = fit_dixon_coles(matches, xi=0.05, fit_rho=False)
    lam_a, lam_b = decayed.expected_goals("A", "B")
    # Recent form (B strong) should dominate: B expected to outscore A.
    assert lam_b > lam_a


def test_csv_parser_skips_blank_rows():
    text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/01/2024,A,B,2,1\n,,,,\n02/01/2024,B,A,0,0\n"
    matches = load_matches_from_string(text)
    assert len(matches) == 2
