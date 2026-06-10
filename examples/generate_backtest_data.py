"""Generate a synthetic multi-season dataset *with odds* for the backtester.

This is illustrative, not real data. Eight teams with fixed strengths play three
double round-robin "seasons". For each match we:

* simulate the result from the teams' true Poisson goal rates,
* derive the *fair* odds from the true scoreline distribution,
* publish a **closing** price that is sharp (fair minus a small margin), and an
  **opening** price that is noisy and occasionally generous.

The deliberate inefficiency (sometimes-soft opening prices vs a sharp close) is
what a good model is meant to exploit, so the backtest shows positive CLV/ROI on
this data. Real markets are far less forgiving — see the README caveats.

Output columns mimic football-data.co.uk (B365* opening, B365C* closing).

Run:  python examples/generate_backtest_data.py
"""

import csv
import os
from datetime import date, timedelta

import numpy as np

from footyvalue.scoreline import score_matrix
from footyvalue.markets import match_odds, over_under

TEAMS = {
    "Lions":   (0.45, 0.40), "Eagles": (0.35, 0.25),
    "Bears":   (0.20, 0.15), "Wolves": (0.05, 0.05),
    "Foxes":   (-0.05, -0.05), "Otters": (-0.20, -0.10),
    "Rabbits": (-0.35, -0.25), "Moles":  (-0.45, -0.45),
}
HOME_ADV = 0.28
BASE = 0.10
N_SEASONS = 3
CLOSE_MARGIN = 0.03   # closing book margin (sharp)

OUT = os.path.join(os.path.dirname(__file__), "sample_backtest.csv")

FIELDS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5",
]


def _price(prob, inflation):
    """Decimal odds for a probability, inflated by ``inflation`` (e.g. -0.03)."""
    prob = min(max(prob, 1e-6), 0.999)
    return round((1.0 / prob) * (1.0 + inflation), 2)


def main():
    rng = np.random.default_rng(424242)
    names = list(TEAMS)
    start = date(2022, 8, 6)
    rows = []
    day = 0

    for _season in range(N_SEASONS):
        for home in names:
            for away in names:
                if home == away:
                    continue
                ah, dh = TEAMS[home]
                aa, da = TEAMS[away]
                lam_h = np.exp(BASE + HOME_ADV + ah - da)
                lam_a = np.exp(BASE + aa - dh)

                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))

                m = score_matrix(lam_h, lam_a)
                mo = match_odds(m)
                ou = over_under(m, 2.5)
                probs = {
                    "B365H": mo["home"], "B365D": mo["draw"], "B365A": mo["away"],
                    "B365>2.5": ou["over"], "B365<2.5": ou["under"],
                }

                row = {
                    "Date": (start + timedelta(days=day)).strftime("%d/%m/%Y"),
                    "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag,
                }
                for col, p in probs.items():
                    # Opening: noisy, skewed slightly generous (sometimes value).
                    u = rng.uniform(-0.04, 0.10)
                    row[col] = _price(p, u)
                    # Closing: sharp (fair minus a small book margin).
                    row["B365C" + col[len("B365"):]] = _price(p, -CLOSE_MARGIN)
                rows.append(row)
                day += 3

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matches to {OUT}")


if __name__ == "__main__":
    main()
