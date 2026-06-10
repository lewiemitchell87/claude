"""Command-line interface for footyvalue.

Examples
--------
    # Fit ratings from a historical CSV (football-data.co.uk format)
    python -m footyvalue.cli fit --history E0.csv --out ratings.json --xi 0.003

    # Scan fixtures + odds supplied in a JSON file
    python -m footyvalue.cli scan --ratings ratings.json --fixtures fixtures.json \
        --min-ev 0.03 --bankroll 1000

    # Price a single in-play situation
    python -m footyvalue.cli inplay --ratings ratings.json \
        --home "Arsenal" --away "Chelsea" --minute 60 --score 1-0 \
        --odds '{"match_odds": {"home": 1.5, "draw": 4.0, "away": 8.0}}'

    # Run the bundled, offline end-to-end demo (no network, no API key)
    python -m footyvalue.cli demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .ratings import TeamRatings, fit_dixon_coles
from .engine import ScanConfig, FixtureResult, evaluate_fixture, scan_fixtures
from .data.football_data import load_matches_csv, load_backtest_matches
from .value import ValueOpportunity
from .backtest import run_backtest
from .scanner import PreGameScanner, FixtureOdds, ScannerHit

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def format_opportunity(o: ValueOpportunity) -> str:
    stake = f"  stake {o.stake:,.2f}" if o.stake else ""
    return (
        f"    [{o.market:<16}] {o.selection:<6} "
        f"@ {o.odds:>6.2f}  model {o.model_prob*100:5.1f}%  "
        f"fair {o.fair_odds:>6.2f}  edge {o.edge*100:+5.1f}%  "
        f"EV {o.ev*100:+5.1f}%{stake}"
    )


def print_results(results: List[FixtureResult], show_all: bool = False) -> int:
    total = 0
    for r in results:
        if not r.opportunities and not show_all:
            continue
        state = ""
        if r.minute is not None and r.score is not None:
            state = f"  [LIVE {r.minute}'  {r.score[0]}-{r.score[1]}]"
        print(f"\n{r.home} vs {r.away}{state}")
        print(f"    xG: {r.lam_home:.2f} - {r.lam_away:.2f}")
        if not r.opportunities:
            print("    (no value found)")
            continue
        for o in r.opportunities:
            print(format_opportunity(o))
            total += 1
    print(f"\n{total} value opportunit{'y' if total == 1 else 'ies'} found.\n")
    return total


def _config_from_args(args) -> ScanConfig:
    return ScanConfig(
        min_ev=args.min_ev,
        min_edge=args.min_edge,
        kelly_cap=args.kelly_cap,
        bankroll=args.bankroll,
    )


def _parse_score(score: Optional[str]):
    if not score:
        return 0, 0
    try:
        h, a = score.split("-")
        return int(h), int(a)
    except ValueError:
        raise SystemExit(f"Invalid --score '{score}'. Use e.g. 1-0")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_fit(args) -> int:
    matches = load_matches_csv(args.history)
    if not matches:
        print(f"No matches loaded from {args.history}", file=sys.stderr)
        return 1
    ratings = fit_dixon_coles(matches, xi=args.xi, fit_rho=not args.no_rho)
    ratings.save(args.out)
    print(
        f"Fitted {len(ratings.teams)} teams from {len(matches)} matches.\n"
        f"  home advantage: {ratings.home_advantage:+.3f} (log-goals)\n"
        f"  rho:            {ratings.rho:+.3f}\n"
        f"  saved -> {args.out}"
    )
    return 0


def cmd_scan(args) -> int:
    ratings = TeamRatings.load(args.ratings)
    config = _config_from_args(args)

    if args.live:
        from .data.odds_api import OddsApiClient

        client = OddsApiClient(api_key=args.api_key)
        fx = client.fetch_odds(sport=args.sport)
        fixtures = [
            {"home": f.home, "away": f.away, "markets": f.markets} for f in fx
        ]
    else:
        with open(args.fixtures) as fh:
            fixtures = json.load(fh)

    results = scan_fixtures(ratings, fixtures, config)
    print_results(results, show_all=args.show_all)
    return 0


def cmd_inplay(args) -> int:
    ratings = TeamRatings.load(args.ratings)
    config = _config_from_args(args)
    home_goals, away_goals = _parse_score(args.score)
    odds = json.loads(args.odds) if args.odds else {}

    result = evaluate_fixture(
        ratings, args.home, args.away, odds, config,
        minute=args.minute, home_goals=home_goals, away_goals=away_goals,
    )
    print_results([result], show_all=True)
    return 0


def cmd_backtest(args) -> int:
    history = args.history or os.path.join(EXAMPLES_DIR, "sample_backtest.csv")
    matches = load_backtest_matches(history)
    priced = [m for m in matches if m["odds"]]
    if not priced:
        print(f"No matches with odds found in {history}", file=sys.stderr)
        return 1

    result = run_backtest(
        matches,
        min_train_matches=args.min_train,
        refit_every=args.refit_every,
        xi=args.xi,
        min_ev=args.min_ev,
        min_edge=args.min_edge,
        commission=args.commission,
        allow_lay=args.lay,
        staking=args.staking,
        stake_unit=args.stake_unit,
        kelly_cap=args.kelly_cap,
        starting_bankroll=args.bankroll if args.bankroll > 0 else 1000.0,
    )
    print(f"Backtest over {len(matches)} matches ({len(priced)} with odds), "
          f"staking={args.staking}, min_ev={args.min_ev}, "
          f"lay={'on' if args.lay else 'off'}, commission={args.commission}\n")
    print(result.summary_text())
    return 0


def _format_hit(hit: ScannerHit) -> str:
    o = hit.opportunity
    when = hit.commence_time or "?"
    return (
        f"[{o.side.upper()}] {hit.home} v {hit.away} ({when})  "
        f"{o.market}/{o.selection} @ {o.odds:.2f}  "
        f"model {o.model_prob*100:.1f}%  EV {o.ev*100:+.1f}%"
        + (f"  stake {o.stake:,.2f}" if o.stake else "")
    )


def cmd_watch(args) -> int:
    if args.ratings:
        ratings = TeamRatings.load(args.ratings)
    elif args.demo:
        ratings = fit_dixon_coles(
            load_matches_csv(os.path.join(EXAMPLES_DIR, "sample_history.csv")),
            xi=0.0, fit_rho=True,
        )
    else:
        print("watch requires --ratings (or use --demo)", file=sys.stderr)
        return 1

    if args.demo:
        from datetime import datetime, timedelta, timezone
        with open(os.path.join(EXAMPLES_DIR, "sample_fixtures.json")) as fh:
            raw = json.load(fh)
        soon = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        demo_fixtures = [
            FixtureOdds(event_id=str(i), home=fx["home"], away=fx["away"],
                        commence_time=soon, back=fx["markets"])
            for i, fx in enumerate(raw) if "minute" not in fx
        ]
        source = lambda: demo_fixtures
        max_polls = 2
    else:
        from .scanner import odds_api_source
        source = odds_api_source(sport=args.sport, api_key=args.api_key)
        max_polls = args.max_polls

    scanner = PreGameScanner(
        ratings, source,
        min_ev=args.min_ev, min_edge=args.min_edge, kelly_cap=args.kelly_cap,
        bankroll=args.bankroll, commission=args.commission,
        min_seconds_to_start=args.min_seconds_to_start,
    )

    poll_no = {"n": 0}

    def on_hit(hit: ScannerHit):
        print("  " + _format_hit(hit))

    def announce():
        poll_no["n"] += 1
        print(f"\n--- poll {poll_no['n']} @ {args.poll}s interval ---")

    # Run with a wrapper that announces each poll.
    import time as _t
    polls = 0
    while max_polls is None or polls < max_polls:
        announce()
        hits = scanner.poll_once()
        if not hits:
            print("  (no new value)")
        for h in hits:
            on_hit(h)
        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        _t.sleep(args.poll)
    return 0


def cmd_demo(args) -> int:
    """Self-contained end-to-end demo: fit -> price -> value, no network."""
    history = os.path.join(EXAMPLES_DIR, "sample_history.csv")
    fixtures_path = os.path.join(EXAMPLES_DIR, "sample_fixtures.json")

    print("Fitting ratings from bundled sample history...")
    matches = load_matches_csv(history)
    ratings = fit_dixon_coles(matches, xi=0.0, fit_rho=True)
    print(f"  {len(ratings.teams)} teams, {len(matches)} matches, "
          f"home advantage {ratings.home_advantage:+.3f}")

    with open(fixtures_path) as fh:
        fixtures = json.load(fh)

    config = ScanConfig(min_ev=0.02, bankroll=1000.0, kelly_cap=0.25)
    print("\nScanning bundled fixtures for value (bankroll 1000, fractional Kelly 0.25)...")
    results = scan_fixtures(ratings, fixtures, config)
    print_results(results, show_all=True)
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="footyvalue", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_thresholds(sp):
        sp.add_argument("--min-ev", type=float, default=0.02,
                        help="Minimum expected value to flag a bet (default 0.02 = +2%%)")
        sp.add_argument("--min-edge", type=float, default=0.0,
                        help="Minimum probability edge to flag a bet")
        sp.add_argument("--kelly-cap", type=float, default=0.25,
                        help="Cap on the Kelly staking fraction (fractional Kelly)")
        sp.add_argument("--bankroll", type=float, default=0.0,
                        help="Bankroll for stake sizing (0 = no stake shown)")

    # fit
    f = sub.add_parser("fit", help="Fit team ratings from a results CSV")
    f.add_argument("--history", required=True, help="CSV of historical results")
    f.add_argument("--out", default="ratings.json", help="Output ratings file")
    f.add_argument("--xi", type=float, default=0.0,
                   help="Time-decay per day (e.g. 0.003). 0 = equal weighting")
    f.add_argument("--no-rho", action="store_true", help="Disable Dixon-Coles rho")
    f.set_defaults(func=cmd_fit)

    # scan
    s = sub.add_parser("scan", help="Scan fixtures/odds for value")
    s.add_argument("--ratings", required=True, help="Ratings JSON from `fit`")
    s.add_argument("--fixtures", help="Fixtures+odds JSON file")
    s.add_argument("--live", action="store_true", help="Fetch live odds via The Odds API")
    s.add_argument("--sport", default="soccer_epl", help="Odds API sport key (with --live)")
    s.add_argument("--api-key", default=None, help="Odds API key (or ODDS_API_KEY env)")
    s.add_argument("--show-all", action="store_true", help="Show fixtures with no value too")
    add_thresholds(s)
    s.set_defaults(func=cmd_scan)

    # inplay
    i = sub.add_parser("inplay", help="Price a single in-play situation")
    i.add_argument("--ratings", required=True, help="Ratings JSON from `fit`")
    i.add_argument("--home", required=True)
    i.add_argument("--away", required=True)
    i.add_argument("--minute", type=int, required=True)
    i.add_argument("--score", default="0-0", help="Current score, e.g. 1-0")
    i.add_argument("--odds", help="Odds JSON, e.g. '{\"match_odds\":{\"home\":1.5,...}}'")
    add_thresholds(i)
    i.set_defaults(func=cmd_inplay)

    # backtest
    b = sub.add_parser("backtest", help="Walk-forward backtest with CLV/ROI metrics")
    b.add_argument("--history", help="Results+odds CSV (football-data format). "
                                     "Omit to use the bundled synthetic dataset.")
    b.add_argument("--min-train", type=int, default=60,
                   help="Matches required before betting starts")
    b.add_argument("--refit-every", type=int, default=10,
                   help="Refit ratings every N matches (speed/accuracy trade-off)")
    b.add_argument("--xi", type=float, default=0.0, help="Time-decay per day")
    b.add_argument("--staking", choices=["flat", "kelly"], default="flat",
                   help="Level stakes (flat) or compounding fractional Kelly")
    b.add_argument("--stake-unit", type=float, default=1.0, help="Flat stake size")
    b.add_argument("--lay", action="store_true",
                   help="Also consider lay bets (treats the price as a lay price)")
    b.add_argument("--commission", type=float, default=0.0,
                   help="Exchange commission on net winnings (e.g. 0.02)")
    add_thresholds(b)
    b.set_defaults(func=cmd_backtest)

    # watch (continuous pre-game scanner)
    w = sub.add_parser("watch", help="Continuously scan pre-game fixtures for value")
    w.add_argument("--ratings", help="Ratings JSON from `fit` (omit with --demo)")
    w.add_argument("--sport", default="soccer_epl", help="Odds API sport key")
    w.add_argument("--api-key", default=None, help="Odds API key (or ODDS_API_KEY env)")
    w.add_argument("--poll", type=float, default=60.0, help="Seconds between polls")
    w.add_argument("--max-polls", type=int, default=None, help="Stop after N polls")
    w.add_argument("--commission", type=float, default=0.0,
                   help="Exchange commission (for lay staking with an exchange feed)")
    w.add_argument("--min-seconds-to-start", type=float, default=0.0,
                   help="Skip fixtures starting sooner than this many seconds")
    w.add_argument("--demo", action="store_true",
                   help="Run offline against bundled fixtures (no network/API key)")
    add_thresholds(w)
    w.set_defaults(func=cmd_watch)

    # demo
    d = sub.add_parser("demo", help="Run the bundled offline end-to-end demo")
    d.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # scan needs either --fixtures or --live
    if args.command == "scan" and not args.live and not args.fixtures:
        parser.error("scan requires --fixtures or --live")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
