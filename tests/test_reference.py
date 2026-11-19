"""Absolute values against scipy, an independently developed code."""

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
import scipy.stats as st
from conftest import CHOLESKY, MEAN, WEIGHTS

import priorities as pr

pytestmark = vv.case(
    "unit-distributions", "cross-code-benchmark", reference_code="scipy"
)

# Probabilities at which each reference is evaluated: both tails and the bulk.
LEVELS = np.array([0.03, 0.2, 0.5, 0.8, 0.97])
BETA = st.beta(0.867, 3.03)
MIXTURE = [st.norm(-2.0, 0.5), st.norm(0.0, 1.0), st.norm(3.0, 2.0)]

REFERENCES = {
    "uniform": (st.uniform(-1.5, 4.0).ppf, st.uniform(-1.5, 4.0)),
    "log_uniform": (st.loguniform(0.3, 40.0).ppf, st.loguniform(0.3, 40.0)),
    "normal": (st.norm(0.7, 1.3).ppf, st.norm(0.7, 1.3)),
    "log_normal": (
        st.lognorm(0.6, scale=np.exp(0.2)).ppf,
        st.lognorm(0.6, scale=np.exp(0.2)),
    ),
    "half_normal": (st.halfnorm(scale=1.7).ppf, st.halfnorm(scale=1.7)),
    "rayleigh": (st.rayleigh(scale=0.3).ppf, st.rayleigh(scale=0.3)),
    "beta": (BETA.ppf, BETA),
    "truncated_normal": (
        st.truncnorm(-1.7 / 1.3, 1.3 / 1.3, 0.7, 1.3).ppf,
        st.truncnorm(-1.7 / 1.3, 1.3 / 1.3, 0.7, 1.3),
    ),
    "truncated_upper": (
        st.truncnorm(-np.inf, 1.3 / 1.3, 0.7, 1.3).ppf,
        st.truncnorm(-np.inf, 1.3 / 1.3, 0.7, 1.3),
    ),
}


@pytest.mark.parametrize("name", list(REFERENCES))
def test_density_and_distribution_match_scipy(name, dists):
    """log_prob and cdf equal scipy.stats at five quantiles.

    Fails if a normalization constant, a parameter convention (scale against
    rate, log-space location) or the truncation normalizer is wrong.
    Tolerance: rtol 1e-12 with atol 0, floating-point; the measured worst
    case is 4e-15. Every expected log density is at least 0.01 from zero, so
    the relative comparison is well conditioned.
    """
    quantile, reference = REFERENCES[name]
    points = quantile(LEVELS)
    expected = reference.logpdf(points)
    assert np.all(np.abs(expected) > 0.01)
    dist = dists[name]
    np.testing.assert_allclose(dist.log_prob(jnp.asarray(points)), expected, 1e-12, 0)
    np.testing.assert_allclose(dist.cdf(jnp.asarray(points)), LEVELS, 1e-12, 0)


def test_truncated_beta_matches_scipy(dists):
    """A Beta cut at 0.95 is the scipy Beta renormalized by its mass below 0.95.

    Fails if the truncation normalizer uses the wrong side or the wrong distribution.
    Tolerance: rtol 1e-12 with atol 0, floating-point (measured 2e-15).
    """
    points = BETA.ppf(LEVELS * BETA.cdf(0.95))
    dist = dists["truncated_beta"]
    expected = BETA.logpdf(points) - np.log(BETA.cdf(0.95))
    np.testing.assert_allclose(dist.log_prob(jnp.asarray(points)), expected, 1e-12, 0)
    np.testing.assert_allclose(dist.cdf(jnp.asarray(points)), LEVELS, 1e-12, 0)


def test_beta_quantile_matches_scipy_deep_in_the_tail():
    """The Beta quantile equals scipy's betaincinv down to u = 1e-100.

    Fails if the root search runs on the linear axis, which loses the lower
    tail (58 percent error in the value at u = 1e-6 for a shape of 0.2).
    Tolerance: rtol 1e-12 with atol 0, spike; measured 3e-14 for the first
    shape pair and 9e-14 for the second.
    """
    unit = np.logspace(-100.0, -1.0, 34)
    for a, b, lowest in ((0.867, 3.03, 0), (0.2, 3.03, 24)):
        # For a = 0.2 the value underflows below u = 1e-30; keep what resolves.
        got = pr.Beta(a, b).icdf(jnp.asarray(unit[lowest:]))
        expected = scipy.special.betaincinv(a, b, unit[lowest:])
        assert np.all(expected > 1e-300)
        np.testing.assert_allclose(got, expected, 1e-12, 0)


def test_mixture_matches_scipy(dists):
    """A scalar mixture equals the weighted sum of scipy normal densities.

    Fails if the weights are not normalized or the component axis is reduced
    in the wrong place. Tolerance: rtol 1e-12, atol 0, floating-point.
    """
    points = np.linspace(-4.0, 7.0, 7)
    density = sum(
        w * part.pdf(points) for w, part in zip(WEIGHTS, MIXTURE, strict=True)
    )
    below = sum(w * part.cdf(points) for w, part in zip(WEIGHTS, MIXTURE, strict=True))
    dist = dists["mixture"]
    np.testing.assert_allclose(jnp.exp(dist.log_prob(points)), density, 1e-12, 0)
    np.testing.assert_allclose(dist.cdf(jnp.asarray(points)), below, 1e-12, 0)


def test_event_dists_match_scipy(dists):
    """The multivariate normal and its mixture equal scipy's log densities.

    Fails if the Cholesky factor is used as the covariance, the determinant
    term is missing, or the mixture reduces the event axis with the
    components. Tolerance: rtol 1e-12, atol 0, floating-point (measured
    6e-16).
    """
    points = np.array([[0.3, -1.0, 2.0], [1.5, 0.5, 4.0], [-2.0, -4.0, 1.0]])
    factor = np.array(CHOLESKY)
    first = st.multivariate_normal(MEAN, factor @ factor.T)
    second = st.multivariate_normal(MEAN[::-1], 4 * factor @ factor.T)
    np.testing.assert_allclose(
        dists["multivariate_normal"].log_prob(points), first.logpdf(points), 1e-12, 0
    )
    mixed = np.log(WEIGHTS[0] * first.pdf(points) + WEIGHTS[1] * second.pdf(points))
    mixed -= np.log(WEIGHTS[0] + WEIGHTS[1])
    np.testing.assert_allclose(dists["event_mixture"].log_prob(points), mixed, 1e-12, 0)
