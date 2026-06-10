"""Scoreline distribution engine.

The whole system is built on one object: a matrix ``P`` where ``P[h, a]`` is the
probability that the home team scores ``h`` goals and the away team scores ``a``
goals. Every market (Match Odds, BTTS, Over/Under) is a simple sum over cells of
this matrix, which guarantees the markets are mutually consistent.

Goals are modelled as (approximately) independent Poisson variables with means
``lambda_home`` and ``lambda_away`` — the *expected goals* for each side — with an
optional Dixon-Coles low-score dependency correction (``rho``) that fixes the
well-known tendency of the pure-Poisson model to misprice 0-0, 1-0, 0-1 and 1-1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

# A small floor for the Poisson mean so log/division never blows up.
_MIN_LAMBDA = 1e-12


def poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    """Return ``[P(X=0), P(X=1), ..., P(X=max_goals)]`` for ``X ~ Poisson(lam)``.

    Computed in log space for numerical stability at larger goal counts.
    """
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    lam = max(float(lam), _MIN_LAMBDA)
    k = np.arange(0, max_goals + 1)
    log_pmf = -lam + k * np.log(lam) - gammaln(k + 1)
    return np.exp(log_pmf)


def dixon_coles_correction(
    matrix: np.ndarray, lam_home: float, lam_away: float, rho: float
) -> np.ndarray:
    """Apply the Dixon-Coles low-score dependency correction in place-safe form.

    Only the four lowest-score cells are adjusted. ``rho`` is the dependency
    parameter (typically a small negative number). ``rho == 0`` recovers the
    independent-Poisson model.
    """
    if rho == 0.0:
        return matrix
    m = matrix.copy()
    # matrix is indexed [home, away]; tau follows Dixon & Coles (1997).
    m[0, 0] *= 1.0 - lam_home * lam_away * rho
    m[0, 1] *= 1.0 + lam_home * rho
    m[1, 0] *= 1.0 + lam_away * rho
    m[1, 1] *= 1.0 - rho
    return m


def score_matrix(
    lam_home: float,
    lam_away: float,
    max_goals: int = 10,
    rho: float = 0.0,
) -> np.ndarray:
    """Build the normalised joint scoreline distribution ``P[home, away]``.

    Parameters
    ----------
    lam_home, lam_away:
        Expected goals for the home and away teams.
    max_goals:
        Truncation point. With typical football scoring rates, 10 captures
        essentially all probability mass.
    rho:
        Dixon-Coles dependency parameter. Keep small (e.g. ``-0.05`` to ``0.0``);
        large magnitudes can drive the corrected low-score cells negative.

    Returns
    -------
    np.ndarray
        A ``(max_goals+1, max_goals+1)`` matrix that sums to 1.
    """
    home = poisson_pmf_vector(lam_home, max_goals)
    away = poisson_pmf_vector(lam_away, max_goals)
    matrix = np.outer(home, away)  # [home, away]

    if rho != 0.0:
        matrix = dixon_coles_correction(matrix, lam_home, lam_away, rho)
        # The correction can in principle push a cell slightly negative for
        # extreme rho; clip so we never emit a negative probability.
        np.clip(matrix, 0.0, None, out=matrix)

    total = matrix.sum()
    if total <= 0.0:
        raise ValueError("Degenerate scoreline matrix (sum <= 0)")
    return matrix / total
