"""footyvalue — a football value-betting detection engine.

The system models the joint distribution of the final scoreline of a match
(using a Poisson / Dixon-Coles goals model) and reads every supported market
off that single distribution:

* Match Odds (1X2)
* Both Teams To Score (BTTS Yes/No)
* Over/Under total goals (any line)

It then compares the model's probabilities against the prices offered by the
market to surface positive expected-value ("value") selections, both pre-game
and in-play.
"""

from .scoreline import score_matrix, poisson_pmf_vector
from .markets import match_odds, over_under, btts, all_markets
from .inplay import inplay_markets
from .value import (
    ValueOpportunity,
    expected_value,
    edge,
    kelly_fraction,
    find_value,
)

__all__ = [
    "score_matrix",
    "poisson_pmf_vector",
    "match_odds",
    "over_under",
    "btts",
    "all_markets",
    "inplay_markets",
    "ValueOpportunity",
    "expected_value",
    "edge",
    "kelly_fraction",
    "find_value",
]

__version__ = "0.1.0"
