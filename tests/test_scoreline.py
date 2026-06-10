import math

import numpy as np
import pytest
from scipy.stats import poisson

from footyvalue.scoreline import poisson_pmf_vector, score_matrix


def test_poisson_pmf_matches_scipy():
    lam = 1.37
    got = poisson_pmf_vector(lam, 12)
    want = poisson.pmf(np.arange(0, 13), lam)
    assert np.allclose(got, want, atol=1e-12)


def test_poisson_pmf_handles_zero_lambda():
    v = poisson_pmf_vector(0.0, 5)
    # All mass at zero goals.
    assert v[0] == pytest.approx(1.0)
    assert np.allclose(v[1:], 0.0)


def test_score_matrix_sums_to_one():
    m = score_matrix(1.6, 1.1, max_goals=10)
    assert m.sum() == pytest.approx(1.0, abs=1e-12)
    assert (m >= 0).all()


def test_independent_marginals_recovered_without_dc():
    lam_h, lam_a = 1.8, 0.9
    m = score_matrix(lam_h, lam_a, max_goals=15, rho=0.0)
    # Home marginal should match Poisson(lam_h) (within truncation).
    home_marginal = m.sum(axis=1)
    assert np.allclose(home_marginal, poisson.pmf(np.arange(16), lam_h), atol=1e-4)


def test_dixon_coles_shifts_low_scores():
    lam_h, lam_a = 1.3, 1.2
    base = score_matrix(lam_h, lam_a, rho=0.0)
    dc = score_matrix(lam_h, lam_a, rho=-0.08)
    # Negative rho increases the 0-0 and 1-1 mass relative to independence.
    assert dc[0, 0] > base[0, 0]
    assert dc[1, 1] > base[1, 1]
    assert dc.sum() == pytest.approx(1.0, abs=1e-12)
    assert (dc >= 0).all()


def test_higher_expected_goals_increases_home_scoring():
    low = score_matrix(0.8, 1.0)
    high = score_matrix(2.4, 1.0)
    # P(home scores 2+) should rise with expected goals.
    assert high[2:].sum() > low[2:].sum()
