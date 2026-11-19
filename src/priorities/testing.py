"""Contract checks for a distribution, for this library's tests and for subclass owners.

``check_distribution`` ties ``sample``, ``cdf`` and ``icdf`` to ``log_prob``
without restating any formula, so it applies unchanged to a new subclass of
``AbstractDistribution``. The absolute scale of ``log_prob`` is not covered here: pin
it against an independent reference.
"""

import jax
import jax.numpy as jnp
from jax import Array

from priorities._distribution import AbstractDistribution
from priorities._transform import from_real, real_log_prob, to_real

# sup-norm quantile of the Kolmogorov distribution: P(sqrt(n) D > 2.69) = 1e-6.
KS_SCALE = 2.69


def ks_distance(dist: AbstractDistribution, draws: Array) -> Array:
    """Return the Kolmogorov-Smirnov distance of draws to ``dist.cdf``.

    Args:
        dist: A distribution whose ``cdf`` is a distribution function (scalar)
            or a map to independent uniforms (event distribution with an inverse CDF).
        draws: Values of shape ``(n,) + event``.

    Returns:
        The largest distance between the empirical and the uniform
        distribution functions of ``cdf(draws)``, over the coordinates.
    """
    count = draws.shape[0]
    unit = jnp.sort(dist.cdf(draws).reshape(count, -1), axis=0)
    rank = jnp.arange(count)[:, None]
    return jnp.max(jnp.maximum((rank + 1) / count - unit, unit - rank / count))


def icdf_round_trip(dist: AbstractDistribution, dtype=None, ulps: float = 4.0) -> Array:
    """Return the scaled error of ``cdf(icdf(u))`` over both tails.

    The grid is logarithmic in ``u`` from 1e-30 to 0.5 and in ``1 - u`` from
    0.5 down to the resolution of the dtype. The error is divided by
    ``min(u, 1 - u)`` after removing what the inputs cannot resolve:
    ``ulps`` units in the last place of ``u`` and of the value (the latter
    mapped through the density). An absolute tolerance cannot see a lost tail;
    this one can.

    Args:
        dist: A scalar distribution with an inverse CDF.
        dtype: Floating dtype of the grid; the default float when ``None``.
        ulps: Allowance for the representation of ``u`` and of the value.

    Returns:
        The relative error at each grid point, to compare with a tolerance.
    """
    dtype = jnp.zeros((), dtype).dtype
    eps = jnp.finfo(dtype).eps
    lower = jnp.logspace(-30.0, jnp.log10(0.5), 61, dtype=dtype)
    upper = 1 - jnp.logspace(jnp.log10(1e4 * eps), jnp.log10(0.5), 25, dtype=dtype)
    unit = jnp.concatenate([lower, upper])
    value = dist.icdf(unit)
    floor = ulps * eps * (unit + jnp.abs(value) * jnp.exp(dist.log_prob(value)))
    error = jnp.abs(dist.cdf(value) - unit)
    return jnp.maximum(error - floor, 0.0) / jnp.minimum(unit, 1 - unit)


def _check_support(dist, low, high):
    grad = jax.vmap(jax.grad(dist.log_prob))
    finfo = jnp.finfo(jnp.result_type(low, high))
    # One spacing beyond each bound, kept a normal float: subnormals can flush.
    step = jnp.maximum(finfo.eps * jnp.abs(jnp.stack([low, high])), finfo.tiny)
    outside = jnp.stack([low - 1.0, low - step[0], high + step[1], high + 1.0])
    outside = outside[jnp.isfinite(outside)]
    assert bool(jnp.all(dist.log_prob(outside) == -jnp.inf)), "density off support"
    assert bool(jnp.all(grad(outside) == 0.0)), "gradient off support"
    assert bool(jnp.isnan(dist.log_prob(jnp.nan))), "NaN value must give NaN"


def _check_normalization(dist, tol, half_width=40.0, count=8001):
    x = jnp.linspace(-half_width, half_width, count)
    total = jnp.sum(jnp.exp(real_log_prob(dist, x))) * (x[1] - x[0])
    assert abs(float(total) - 1.0) < tol, f"density integrates to {total}"


def check_distribution(
    dist: AbstractDistribution,
    key: Array,
    *,
    n_draws: int = 200_000,
    rtol: float = 1e-10,
    unit_rtol: float = 1e-11,
) -> None:
    """Assert that the methods of an unbatched distribution agree with its density.

    The checks, in order:

    - draws lie in the support and survive the default transform round trip;
    - a scalar distribution is exactly ``-inf`` with zero gradient outside its support
      and NaN at NaN;
    - a scalar distribution without a distribution function integrates to one over
      the unconstrained axis (trapezoid rule on ``[-40, 40]``);
    - otherwise ``cdf`` is 0 and 1 at the support ends and its derivative
      is ``exp(log_prob)`` at 33 quantiles of the draws;
    - with an inverse CDF, ``cdf(icdf(u))`` returns ``u`` to ``unit_rtol``
      relative to ``min(u, 1 - u)`` (see ``icdf_round_trip``);
    - the draws follow ``cdf``: the Kolmogorov-Smirnov distance stays
      below ``2.69 / sqrt(n_draws)``, a false-rejection rate of 1e-6 per
      coordinate.

    The defaults are for float64. An event distribution without an inverse CDF gets the
    first check only.

    Args:
        dist: The distribution to check, with an empty batch shape.
        key: PRNG key for the draws.
        n_draws: Number of draws. A deviation of 5 percent in a scale needs
            200,000 draws to be rejected; fewer suffice when ``sample`` is the
            quantile of uniforms.
        rtol: Relative tolerance of the derivative and normalization checks.
        unit_rtol: Relative tolerance of the unit round trip.

    Raises:
        AssertionError: If a check fails.
    """
    low, high = dist.support()
    draws = dist.sample(key, (n_draws,))
    assert bool(jnp.all(jnp.isfinite(dist.log_prob(draws)))), "draw off support"
    back, _ = from_real(dist, to_real(dist, draws[:64]))
    assert bool(jnp.allclose(back, draws[:64], rtol=rtol, atol=0.0)), "transform"
    if dist.event_dim == 0:
        _check_support(dist, low, high)
        try:
            ends = dist.cdf(jnp.stack([low, high]))
        except NotImplementedError:
            return _check_normalization(dist, rtol)
        assert bool(jnp.all(ends == jnp.array([0.0, 1.0]))), "cdf at the ends"
        points = jnp.quantile(draws, jnp.linspace(0.01, 0.99, 33))
        slope = jax.vmap(jax.grad(dist.cdf))(points)
        density = jnp.exp(dist.log_prob(points))
        assert bool(jnp.allclose(slope, density, rtol=rtol, atol=0.0)), "d cdf"
    elif not dist.has_icdf:
        return None
    if dist.has_icdf and dist.event_dim == 0:
        worst = float(jnp.max(icdf_round_trip(dist)))
        assert worst < unit_rtol, f"unit round trip off by {worst}"
    distance = float(ks_distance(dist, draws))
    assert distance < KS_SCALE / n_draws**0.5, f"draws: KS distance {distance}"
