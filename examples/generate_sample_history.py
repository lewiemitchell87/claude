"""Generate a deterministic synthetic season for the demo / tests.

Eight teams with known attack/defence strengths play a double round-robin. Goals
are drawn from independent Poisson distributions with a home advantage, so the
fitter has genuine signal to recover. Seeded for reproducibility.

Run:  python examples/generate_sample_history.py
"""

import csv
import os
from datetime import date, timedelta

import numpy as np

# team -> (attack, defence) in log-goal units; higher attack scores more,
# higher defence concedes fewer.
TEAMS = {
    "Lions":     (0.45, 0.40),
    "Eagles":    (0.35, 0.25),
    "Bears":     (0.20, 0.15),
    "Wolves":    (0.05, 0.05),
    "Foxes":     (-0.05, -0.05),
    "Otters":    (-0.20, -0.10),
    "Rabbits":   (-0.35, -0.25),
    "Moles":     (-0.45, -0.45),
}
HOME_ADV = 0.28
BASE = 0.10  # log baseline scoring level

OUT = os.path.join(os.path.dirname(__file__), "sample_history.csv")


def main():
    rng = np.random.default_rng(20240607)
    names = list(TEAMS)
    start = date(2024, 8, 10)
    rows = []
    fixture_no = 0
    for home in names:
        for away in names:
            if home == away:
                continue
            ah, dh = TEAMS[home]
            aa, da = TEAMS[away]
            lam_home = np.exp(BASE + HOME_ADV + ah - da)
            lam_away = np.exp(BASE + aa - dh)
            hg = int(rng.poisson(lam_home))
            ag = int(rng.poisson(lam_away))
            match_day = start + timedelta(days=fixture_no * 3)
            rows.append({
                "Date": match_day.strftime("%d/%m/%Y"),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
            })
            fixture_no += 1

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matches to {OUT}")


if __name__ == "__main__":
    main()
