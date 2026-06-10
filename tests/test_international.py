import os
from datetime import date

import pytest

from footyvalue.data.international import (
    load_international_csv, load_international_from_string,
)
from footyvalue.ratings import TeamRatings, fit_dixon_coles, Match

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_international.csv")

_CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,FALSE\n"
    "2023-03-23,England,Brazil,1,1,Friendly,London,England,FALSE\n"
    "2024-06-15,Spain,Italy,2,0,UEFA Euro,Berlin,Germany,TRUE\n"
    "2026-06-27,Croatia,Ghana,NA,NA,FIFA World Cup,Philadelphia,United States,TRUE\n"
)


def test_skips_future_na_rows():
    matches = load_international_from_string(_CSV)
    assert len(matches) == 3                      # the NA row is dropped
    assert all(isinstance(m.home_goals, int) for m in matches)


def test_neutral_flag_parsed():
    matches = load_international_from_string(_CSV)
    by_teams = {(m.home, m.away): m for m in matches}
    assert by_teams[("Spain", "Italy")].neutral is True
    assert by_teams[("Qatar", "Ecuador")].neutral is False


def test_from_date_filter():
    matches = load_international_from_string(_CSV, from_date=date(2024, 1, 1))
    assert len(matches) == 1
    assert matches[0].home == "Spain"


def test_exclude_friendlies():
    matches = load_international_from_string(_CSV, exclude_friendlies=True)
    assert all(True for _ in matches)             # no exception
    assert ("England", "Brazil") not in {(m.home, m.away) for m in matches}


def test_tournament_filter():
    matches = load_international_from_string(_CSV, tournaments=["FIFA World Cup"])
    assert len(matches) == 1
    assert matches[0].home == "Qatar"


def test_sample_file_loads_and_fits():
    matches = load_international_csv(SAMPLE)
    assert len(matches) > 100
    assert any(m.neutral for m in matches)        # the dataset has neutral games
    ratings = fit_dixon_coles(matches, xi=0.0015, fit_rho=True)
    assert len(ratings.teams) > 10


def test_neutral_drops_home_advantage():
    r = TeamRatings(attack={"A": 0.2, "B": -0.2}, defence={"A": 0.1, "B": -0.1},
                    home_advantage=0.3, rho=0.0, teams=["A", "B"])
    home_h, _ = r.expected_goals("A", "B", neutral=False)
    home_n, _ = r.expected_goals("A", "B", neutral=True)
    assert home_h > home_n                         # home advantage only when not neutral


def test_fit_respects_neutral_in_likelihood():
    # All matches A-vs-B at neutral venues with A scoring more: A's attack should
    # exceed B's, and this must not be absorbed into home advantage.
    matches = [Match("A", "B", 3, 0, date(2024, 1, 1 + (i % 27)), neutral=True)
               for i in range(20)]
    matches += [Match("B", "A", 0, 3, date(2024, 2, 1 + (i % 27)), neutral=True)
                for i in range(20)]
    r = fit_dixon_coles(matches, fit_rho=False)
    assert r.attack["A"] > r.attack["B"]
