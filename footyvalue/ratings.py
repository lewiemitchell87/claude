"""Estimate team attack/defence strengths from historical results.

We fit a Dixon-Coles model by (time-weighted) maximum likelihood. Each team has
an ``attack`` and ``defence`` rating; the expected goals for a fixture are::

    lambda_home = exp(home_advantage + attack[home] - defence[away])
    lambda_away = exp(                 attack[away] - defence[home])

Recent matches are weighted more heavily via an exponential time-decay ``xi``
(per day), as recommended by Dixon & Coles, so the ratings track current form.

These expected goals are exactly the ``lam_home``/``lam_away`` inputs that the
scoreline engine and the in-play model consume, so one fit drives every market.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .scoreline import score_matrix
from .markets import all_markets


@dataclass
class Match:
    """A single historical result."""

    home: str
    away: str
    home_goals: int
    away_goals: int
    match_date: Optional[date] = None

    @staticmethod
    def parse_date(value) -> Optional[date]:
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(str(value), fmt).date()
            except ValueError:
                continue
        return None


@dataclass
class TeamRatings:
    """Fitted ratings plus the global home advantage and dependency parameter."""

    attack: Dict[str, float]
    defence: Dict[str, float]
    home_advantage: float
    rho: float = 0.0
    teams: List[str] = field(default_factory=list)

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        """Expected goals ``(lambda_home, lambda_away)`` for a fixture.

        Unknown teams default to league-average strength (rating 0).
        """
        ah = self.attack.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        dh = self.defence.get(home, 0.0)
        da = self.defence.get(away, 0.0)
        lam_home = np.exp(self.home_advantage + ah - da)
        lam_away = np.exp(aa - dh)
        return float(lam_home), float(lam_away)

    def market_probabilities(self, home: str, away: str, max_goals: int = 10, ou_lines=(0.5, 1.5, 2.5, 3.5)):
        """Pre-match probabilities for every market for this fixture."""
        lam_h, lam_a = self.expected_goals(home, away)
        matrix = score_matrix(lam_h, lam_a, max_goals=max_goals, rho=self.rho)
        return all_markets(matrix, ou_lines=ou_lines)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "teams": self.teams,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "TeamRatings":
        with open(path) as fh:
            data = json.load(fh)
        return cls(
            attack=data["attack"],
            defence=data["defence"],
            home_advantage=data["home_advantage"],
            rho=data.get("rho", 0.0),
            teams=data.get("teams", list(data["attack"].keys())),
        )


def _time_weights(matches: Sequence[Match], xi: float) -> np.ndarray:
    """Exponential decay weights based on each match's age in days."""
    if xi <= 0:
        return np.ones(len(matches))
    dated = [m.match_date for m in matches if m.match_date is not None]
    if not dated:
        return np.ones(len(matches))
    latest = max(dated)
    weights = []
    for m in matches:
        if m.match_date is None:
            weights.append(1.0)
        else:
            age_days = (latest - m.match_date).days
            weights.append(np.exp(-xi * age_days))
    return np.asarray(weights)


def fit_dixon_coles(
    matches: Sequence[Match],
    *,
    xi: float = 0.0,
    fit_rho: bool = True,
    reg: float = 1e-3,
    max_iter: int = 200,
) -> TeamRatings:
    """Fit team ratings by (optionally time-weighted) maximum likelihood.

    Parameters
    ----------
    matches:
        Historical results.
    xi:
        Time-decay rate per day (e.g. ``0.003`` ~ a half-life of ~230 days).
        ``0`` weights all matches equally.
    fit_rho:
        Whether to estimate the Dixon-Coles dependency parameter.
    reg:
        L2 regularisation on attack/defence ratings. Pins the otherwise
        shift-invariant ratings to a centred solution and stabilises teams with
        few games.
    """
    if not matches:
        raise ValueError("No matches supplied")

    teams = sorted({m.home for m in matches} | {m.away for m in matches})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = np.array([index[m.home] for m in matches])
    away_idx = np.array([index[m.away] for m in matches])
    hg = np.array([m.home_goals for m in matches], dtype=float)
    ag = np.array([m.away_goals for m in matches], dtype=float)
    weights = _time_weights(matches, xi)

    log_hg_fac = gammaln(hg + 1)
    log_ag_fac = gammaln(ag + 1)

    mask00 = (hg == 0) & (ag == 0)
    mask01 = (hg == 0) & (ag == 1)
    mask10 = (hg == 1) & (ag == 0)
    mask11 = (hg == 1) & (ag == 1)

    # Parameter layout: [attack(n), defence(n), home_advantage, rho]
    def unpack(params):
        attack = params[:n]
        defence = params[n : 2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1] if fit_rho else 0.0
        return attack, defence, home_adv, rho

    def neg_log_likelihood(params):
        attack, defence, home_adv, rho = unpack(params)
        lam = np.exp(home_adv + attack[home_idx] - defence[away_idx])
        mu = np.exp(attack[away_idx] - defence[home_idx])

        tau = np.ones_like(lam)
        tau[mask00] = 1.0 - lam[mask00] * mu[mask00] * rho
        tau[mask01] = 1.0 + lam[mask01] * rho
        tau[mask10] = 1.0 + mu[mask10] * rho
        tau[mask11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)

        ll = (
            np.log(tau)
            + hg * np.log(lam)
            - lam
            - log_hg_fac
            + ag * np.log(mu)
            - mu
            - log_ag_fac
        )
        penalty = reg * (np.sum(attack ** 2) + np.sum(defence ** 2))
        return -np.sum(weights * ll) + penalty

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.25  # sensible initial home advantage in log-goals
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0)]  # attack/defence, home_adv
    bounds += [(-0.2, 0.2)] if fit_rho else [(0.0, 0.0)]

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    attack, defence, home_adv, rho = unpack(result.x)
    # Centre the ratings (purely cosmetic; predictions are shift-invariant).
    attack = attack - attack.mean()
    defence = defence - defence.mean()

    return TeamRatings(
        attack={t: float(attack[i]) for t, i in index.items()},
        defence={t: float(defence[i]) for t, i in index.items()},
        home_advantage=float(home_adv),
        rho=float(rho),
        teams=teams,
    )
