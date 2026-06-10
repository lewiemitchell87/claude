# footyvalue

A football **value-betting detection engine**. It estimates the "true"
probability of every outcome in a match from a goals model, compares those
probabilities against the prices the market is offering, and surfaces the
selections where you have a positive-expected-value edge — both **pre-game** and
**in-play**.

Markets covered:

- **Match Odds** (1X2 — home / draw / away)
- **Both Teams To Score** (BTTS Yes / No)
- **Over/Under total goals** (any line — 0.5, 1.5, 2.5, 3.5, …)

> ⚠️ **This is a modelling and decision-support tool, not a money printer.**
> A value signal is only as good as the model behind it, and the model is only as
> good as your data. It places no bets by itself. Expected value is a long-run
> statistical statement; variance is real and bankrolls can and do go to zero.
> Bet only what you can afford to lose, and only where it is legal for you to do
> so.

---

## The core idea

Every supported market is a different question about the **same underlying
object**: the joint probability distribution of the final scoreline,
`P[home_goals, away_goals]`.

```
                P[h, a]  (the scoreline matrix)
                   │
   ┌───────────────┼────────────────┐
   ▼               ▼                ▼
 Match Odds      BTTS            Over/Under
 (sum cells     (sum cells      (sum cells by
  by h vs a)     h≥1 & a≥1)      h + a vs line)
```

Because all three markets are read off one matrix, they are always mutually
consistent. We build that matrix from each side's **expected goals**
(`λ_home`, `λ_away`) using a Poisson model with the **Dixon-Coles** low-score
correction (which fixes the well-known mispricing of 0-0, 1-0, 0-1, 1-1).

The expected goals come from **team attack/defence ratings** fitted by
time-weighted maximum likelihood from historical results, so recent form is
weighted more heavily.

**In-play**, the final score is `current_score + remaining_goals`. We rescale the
expected goals by the fraction of the match left to play, build the matrix for
the *additional* goals, offset by the current score, and read every market live.

## Value, edge and staking

For a back bet at decimal odds `O` with model probability `p`:

- **Expected value** per unit stake: `EV = p·O − 1` (positive ⇒ value)
- **Edge**: `p − 1/O` (your probability advantage over the price)
- **Kelly stake fraction**: `(p·O − 1) / (O − 1)`

Staking uses **fractional Kelly** (a cap on the Kelly fraction, default 0.25)
because full Kelly is brutally aggressive and unforgiving of model error.

---

## Install

```bash
pip install -r requirements.txt        # numpy, scipy (+ requests for live odds)
```

Runs on Python 3.9+. The core depends only on numpy/scipy; `requests` is only
needed for the live-odds adapter.

## Quick start — offline demo (no network, no API key)

```bash
python -m footyvalue.cli demo
```

This fits ratings from a bundled synthetic season and scans bundled fixtures
(pre-match and one in-play), printing the value opportunities it finds.

## Workflow

### 1. Fit ratings from historical results

Use any CSV in [football-data.co.uk](https://www.football-data.co.uk/data.php)
format (columns `Date, HomeTeam, AwayTeam, FTHG, FTAG`):

```bash
python -m footyvalue.cli fit --history E0.csv --out ratings.json --xi 0.003
```

`--xi` is the per-day time-decay (e.g. `0.003` ≈ a ~230-day half-life). Use `0`
to weight all matches equally.

### 2. Scan fixtures for value

**From a JSON file** of fixtures + odds (see `examples/sample_fixtures.json`):

```bash
python -m footyvalue.cli scan --ratings ratings.json \
    --fixtures fixtures.json --min-ev 0.03 --bankroll 1000
```

**From live odds** via [The Odds API](https://the-odds-api.com) (free tier):

```bash
export ODDS_API_KEY=your_key_here
python -m footyvalue.cli scan --ratings ratings.json --live --sport soccer_epl
```

### 3. Backtest before risking money (CLV / ROI)

Validate a configuration on history **before** betting. The backtester replays
matches in date order, fits ratings on *only the prior matches* (no lookahead),
places the value bets the engine would have flagged, settles them, and reports
ROI plus **closing-line value** — the single most reliable sign an edge is real.

```bash
# Bundled synthetic dataset (no args needed)
python -m footyvalue.cli backtest --min-train 56 --refit-every 14 --min-ev 0.03

# Your own football-data.co.uk CSV (with odds columns)
python -m footyvalue.cli backtest --history E0.csv --min-ev 0.03 --staking flat
```

Example output on the bundled (deliberately inefficient) synthetic data:

```
Bets:          302  (won 103, push 0)
Win rate:      34.1%
ROI / yield:   +18.56%
Avg CLV:       +6.97%   (positive: 94%)
Max drawdown:  37.66
By market:
  match_odds       bets  192  ROI  +32.0%
  over_under_2.5   bets  110  ROI   -4.9%
```

> The bundled data is synthetic and intentionally exploitable to demonstrate the
> tooling. **Real markets are far harder** — expect ROI near, or below, zero
> until you have a genuine edge. Use **flat (level) stakes** to read ROI cleanly;
> use `--staking kelly` only once an edge is established (it compounds, and is
> unforgiving of model error — it can and will bust a bankroll on bad estimates).

### 4. Continuously scan pre-game fixtures for value (live)

`watch` polls an odds source on an interval, evaluates every **not-yet-started**
fixture, and prints newly-appearing value (deduplicated, so you see each price
once). This is the path for **immediate pre-game value betting**.

```bash
# Offline demo (no network / API key) — two polls against bundled fixtures
python -m footyvalue.cli watch --demo --min-ev 0.03 --bankroll 1000

# Live, via The Odds API (bookmaker back prices)
export ODDS_API_KEY=your_key_here
python -m footyvalue.cli watch --ratings ratings.json --sport soccer_epl \
    --min-ev 0.03 --bankroll 1000 --poll 60
```

Each alert shows the side, fixture, market/selection, price, model probability,
EV and recommended stake:

```
[BACK] Wolves v Bears (2026-06-11T...)  over_under_2.5/over @ 2.80  model 40.5%  EV +13.3%  stake 74.11
```

### Back **and** lay (exchange) staking

Every value check supports both sides with commission on net winnings. Laying at
odds `L` is treated as a back on the opposite outcome at `L/(L-1)`, so the EV and
Kelly maths is identical and consistent. Lay value exists when your model thinks
the selection is **less** likely than its lay price implies (`p < 1/L`).

```python
from footyvalue.value import evaluate_selection
opp = evaluate_selection("match_odds", "home", model_prob=0.30,
                         lay_odds=2.0, commission=0.02, min_ev=0.0)
print(opp.side, opp.ev, opp.stake, opp.backer_stake)   # lay 0.40 ... liability/backer
```

The bundled live source (The Odds API) carries **back** prices only. To use lay
staking live, plug in an exchange feed (e.g. a Betfair adapter) that fills the
`lay` prices on each `FixtureOdds`; pass `--commission` to model the exchange cut.

### 5. Price an in-play situation

```bash
python -m footyvalue.cli inplay --ratings ratings.json \
    --home "Arsenal" --away "Chelsea" --minute 60 --score 1-0 \
    --odds '{"match_odds": {"home": 1.5, "draw": 4.0, "away": 8.0},
             "over_under_2.5": {"over": 2.1, "under": 1.8}}'
```

(For continuous in-play scanning, feed live scores + live odds into
`scan_fixtures` with `minute`/`home_goals`/`away_goals` per fixture.)

## Fixtures JSON format

```json
[
  {
    "home": "Foxes",
    "away": "Eagles",
    "markets": {
      "match_odds":     {"home": 5.0, "draw": 3.6, "away": 2.1},
      "btts":           {"yes": 2.05, "no": 1.78},
      "over_under_2.5": {"over": 2.3, "under": 1.55}
    }
  },
  {
    "home": "Eagles", "away": "Foxes",
    "minute": 70, "home_goals": 1, "away_goals": 0,
    "markets": { "match_odds": {"home": 1.15, "draw": 12.0, "away": 60.0} }
  }
]
```

Omit `minute` for a pre-match fixture. Over/Under markets are keyed
`over_under_<line>` (e.g. `over_under_2.5`).

## Use it as a library

```python
from footyvalue.ratings import fit_dixon_coles
from footyvalue.data.football_data import load_matches_csv
from footyvalue.engine import ScanConfig, evaluate_fixture

ratings = fit_dixon_coles(load_matches_csv("E0.csv"), xi=0.003)

odds = {
    "match_odds": {"home": 2.5, "draw": 3.3, "away": 2.9},
    "over_under_2.5": {"over": 1.9, "under": 2.0},
}
result = evaluate_fixture(ratings, "Arsenal", "Chelsea", odds,
                          ScanConfig(min_ev=0.03, bankroll=1000))
for o in result.opportunities:
    print(o.market, o.selection, o.odds, f"EV {o.ev:+.1%}", f"stake {o.stake:.0f}")
```

## Project layout

```
footyvalue/
  scoreline.py    Poisson / Dixon-Coles scoreline matrix
  markets.py      Match Odds, BTTS, Over/Under derived from the matrix
  inplay.py       Live probabilities conditioned on minute + current score
  ratings.py      Time-weighted Dixon-Coles MLE of team attack/defence
  value.py        EV, edge, Kelly; back + lay (exchange) value detection
  backtest.py     Walk-forward backtester with CLV / ROI / drawdown metrics
  scanner.py      Continuous pre-game value scanner (poll, dedup, alert)
  engine.py       Orchestration (ratings + odds -> ranked value bets)
  odds.py         Odds<->prob, overround (vig) removal
  cli.py          Command-line interface (fit/scan/inplay/backtest/watch/demo)
  data/
    football_data.py   Results + odds loader (CSV / URL)
    odds_api.py        Live odds adapter (The Odds API)
examples/
  generate_sample_history.py   Reproducible synthetic season (results)
  generate_backtest_data.py    Synthetic multi-season dataset with odds
  sample_history.csv           Bundled training data
  sample_fixtures.json         Bundled fixtures + odds
  sample_backtest.csv          Bundled results+odds for backtesting
tests/                         79 tests covering the maths and pipeline
```

## Modelling notes & limitations

- **Garbage in, garbage out.** The ratings need a meaningful history (ideally a
  full season+ of the same competition). Cross-league fixtures need cross-league
  data. Promoted/new teams default to league-average until they have games.
- **What's modelled:** goal-scoring rates, home advantage, low-score dependency,
  recency. **What isn't (yet):** injuries/suspensions, lineups, weather,
  motivation, red cards, schedule congestion, market-implied information. These
  are why the *market* is hard to beat.
- **In-play** uses a time-rescaled goal model; it does not (yet) ingest live
  match state beyond the score (shots, xG, red cards). A red card or a flurry of
  chances materially changes the true rates — treat raw in-play signals with
  extra caution.
- **Beating the closing line** is the real test. Consider validating against
  closing odds (available in the football-data.co.uk files) before staking, and
  always paper-trade a new configuration first.

## Tests

```bash
python -m pytest -q
```

## Responsible gambling

If betting stops being fun, or you're chasing losses, stop. Support:
- UK: **GamCare** 0808 8020 133 — https://www.gamcare.org.uk
- **GambleAware** — https://www.begambleaware.org
- Most exchanges/books offer deposit limits and self-exclusion — use them.
