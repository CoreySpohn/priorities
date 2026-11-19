"""Derivatives of the density, the inverse CDFs and the default transform."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import CHOLESKY, MEAN
from hypothesis import given
from hypothesis import strategies as st
from jax.test_util import check_grads

import priorities as pr

# (builder from a parameter tuple, parameters, value); all of order one, as
# the finite-difference step of check_grads is absolute.
DENSITIES = {
    "uniform": (lambda p: pr.Uniform(*p), (-1.5, 2.5), 0.4),
    "log_uniform": (lambda p: pr.LogUniform(*p), (0.3, 4.0), 1.1),
    "normal": (lambda p: pr.Normal(*p), (0.7, 1.3), 0.2),
    "log_normal": (lambda p: pr.LogNormal(*p), (0.2, 0.6), 1.4),
    "half_normal": (lambda p: pr.HalfNormal(*p), (1.7,), 0.9),
    "rayleigh": (lambda p: pr.Rayleigh(*p), (0.8,), 0.6),
    "beta": (lambda p: pr.Beta(*p), (0.867, 3.03), 0.3),
    "truncated_normal": (
        lambda p: pr.TruncatedDistribution(pr.Normal(p[0], p[1]), p[2], p[3]),
        (0.7, 1.3, -1.0, 2.0),
        0.2,
    ),
    "mixture": (
        lambda p: pr.Mixture(p[0], pr.Normal(p[1], p[2])),
        (jnp.array([0.1, 0.5]), jnp.array([-1.0, 1.0]), jnp.array([0.5, 1.5])),
        0.3,
    ),
    "multivariate_normal": (
        lambda p: pr.MultivariateNormal(p[0], jnp.tril(p[1])),
        (jnp.array(MEAN), jnp.array(CHOLESKY)),
        jnp.array([0.3, -1.0, 2.0]),
    ),
}


@pytest.mark.parametrize("name", list(DENSITIES))
def test_log_prob_gradients_match_finite_differences(name):
    """log_prob is differentiable in the value and in every parameter.

    Fails if a masking branch or a custom rule returns a wrong or zero
    derivative. check_grads to first order, forward and reverse; eps 1e-4
    with rtol and atol 1e-4, the truncation error of the central difference
    at that step for arguments of order one.
    """
    build, parameters, value = DENSITIES[name]
    parameters = tuple(jnp.asarray(p) for p in parameters)

    def log_prob(parameters, value):
        return build(parameters).log_prob(value)

    check_grads(
        log_prob,
        (parameters, jnp.asarray(value)),
        order=1,
        modes=("fwd", "rev"),
        eps=1e-4,
        atol=1e-4,
        rtol=1e-4,
    )


def test_quantile_derivative_is_the_reciprocal_density(unit_dist):
    """d icdf / du equals 1 / pdf at the returned value.

    Fails if the derivative through the Beta root search is the exact zero
    that autodiff gives for a bisection loop, which would silently remove
    reparameterized gradients of a truncated draw. Tolerance: rtol 1e-10,
    floating-point (both sides are closed forms of the same value).
    """
    unit = jnp.array([0.02, 0.3, 0.5, 0.9, 0.99])
    slope = jax.vmap(jax.grad(unit_dist.icdf))(unit)
    density = jnp.exp(unit_dist.log_prob(unit_dist.icdf(unit)))
    assert bool(jnp.all(slope > 0))
    np.testing.assert_allclose(slope * density, 1.0, rtol=1e-10, atol=0)


def test_beta_shape_gradients(dists):
    """The Beta density has shape gradients; its inverse CDFs raise instead.

    Fails if a shape gradient through the distribution function or the
    quantile silently returns zero, or if the plain Beta loses the gradient a
    hierarchical fit needs. The density gradient is the digamma closed form;
    tolerance rtol 1e-12, floating-point.
    """
    a, b, value = 0.867, 3.03, 0.3
    digamma = jax.scipy.special.digamma
    expected = jnp.log(value) - digamma(a) + digamma(a + b)
    got = jax.grad(lambda shape: pr.Beta(shape, b).log_prob(value))(a)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="not differentiable in its shapes"):
        jax.grad(lambda shape: pr.Beta(shape, b).icdf(0.3))(a)
    with pytest.raises(ValueError, match=r"[Bb]etainc"):
        jax.grad(lambda cut: cut.log_prob(value))(dists["truncated_beta"])
    cut_gradient = jax.grad(
        lambda high: pr.TruncatedDistribution(pr.Beta(a, b), high=high).log_prob(value)
    )
    assert float(cut_gradient(0.95)) < 0


BOUNDED = ["normal", "rayleigh", "truncated_upper", "beta", "truncated_beta", "mixture"]


@pytest.mark.parametrize("name", [*BOUNDED, "multivariate_normal"])
def test_log_volume_is_the_log_jacobian_of_from_real(name, dists):
    """log_volume equals the autodiff log determinant of from_real.

    Fails if a transform and its volume term disagree (a missing Jacobian in
    a sampler's density). Covers all four bounds options and the sum over the
    event axis. Tolerance: rtol 1e-10 with atol 1e-12, floating-point.
    """
    dist = dists[name]
    x = jnp.array([-1.3, 0.2, 2.1])
    value, log_volume = pr.from_real(dist, x)
    jacobian = jax.jacfwd(lambda x: pr.from_real(dist, x)[0])(x)
    expected = jnp.log(jnp.abs(jnp.diagonal(jacobian)))
    expected = expected.sum() if dist.event_dim else expected
    np.testing.assert_allclose(log_volume, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pr.to_real(dist, value), x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        pr.real_log_prob(dist, x), dist.log_prob(value) + log_volume, rtol=1e-14
    )


def test_bounds_follow_the_cut_sides(dists):
    """TruncatedDistribution reports the bounded sides of its base plus its own cuts.

    Fails if an upper-only cut is given the lower transform, which would map
    the real line outside the support.
    """
    base = dists["normal"]
    assert pr.TruncatedDistribution(base).bounds == "real"
    assert pr.TruncatedDistribution(base, low=0.0).bounds == "lower"
    assert pr.TruncatedDistribution(base, high=0.0).bounds == "upper"
    assert pr.TruncatedDistribution(base, 0.0, 1.0).bounds == "interval"
    assert pr.TruncatedDistribution(dists["rayleigh"], high=1.0).bounds == "interval"
    value, _ = pr.from_real(dists["truncated_upper"], jnp.array([-30.0, 0.0, 30.0]))
    assert bool(jnp.all(value <= 2.0)) and bool(jnp.all(jnp.diff(value) < 0))


@pytest.mark.parametrize("name", BOUNDED)
@given(x=st.floats(min_value=-20.0, max_value=20.0))
def test_from_real_lands_in_the_support(name, dists, x):
    """Any real coordinate maps into the support and back to itself.

    Fails if a transform leaves the support or the pair is not an inverse.
    The round trip is compared in the value, where it is well conditioned;
    tolerance rtol 1e-12, floating-point.
    """
    dist = dists[name]
    low, high = dist.support()
    value, log_volume = pr.from_real(dist, jnp.asarray(x))
    assert bool((value >= low) & (value <= high)) and bool(jnp.isfinite(log_volume))
    again, _ = pr.from_real(dist, pr.to_real(dist, value))
    np.testing.assert_allclose(again, value, rtol=1e-12, atol=0)
