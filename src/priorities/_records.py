"""Elementwise records: one distribution per class, array parameters only."""

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import betainc, betaln, erf, erfinv, ndtr, ndtri

from priorities._distribution import (
    AbstractDistribution,
    _inexact_arg,
    _mask_fill,
    as_float,
    bounded,
    draw_spec,
    finish,
    quantile,
    unit_draws,
)

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_SQRT_2 = math.sqrt(2.0)


class Uniform(AbstractDistribution):
    """Uniform distribution on the closed interval ``[low, high]``.

    Attributes:
        low: Lower bound, finite.
        high: Upper bound, finite and above ``low``.
    """

    low: Array = eqx.field(converter=as_float)
    high: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        valid = (self.low < self.high) & jnp.isfinite(self.high - self.low)
        return bounded(self, valid, self.low, self.high)

    def log_prob(self, value):
        inside = (value >= self.low) & (value <= self.high)
        return finish(self, -jnp.log(self.high - self.low), value, inside, -jnp.inf)

    def sample(self, key, sample_shape=()):
        return self.icdf(unit_draws(self, key, sample_shape))

    def cdf(self, value):
        clipped = jnp.clip(value, self.low, self.high)
        return finish(self, (clipped - self.low) / (self.high - self.low), value)

    def icdf(self, q):
        return quantile(self, self.low + q * (self.high - self.low), q)


class LogUniform(AbstractDistribution):
    """Distribution with density proportional to ``1 / value`` on ``[low, high]``.

    Attributes:
        low: Lower bound, positive.
        high: Upper bound, finite and above ``low``.
    """

    low: Array = eqx.field(converter=as_float)
    high: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        valid = (self.low > 0) & (self.low < self.high) & jnp.isfinite(self.high)
        return bounded(self, valid, self.low, self.high)

    def _log_ratio(self):
        return jnp.log(self.high / self.low)

    def log_prob(self, value):
        """Return the log density, using a safe log argument off support.

        Args:
            value: Values at which to evaluate the density.

        Returns:
            Log densities with negative infinity outside the support.
        """
        inside = (value >= self.low) & (value <= self.high)
        safe = _mask_fill(inside, _inexact_arg(value, self.low, self.high), 1.0)
        out = -jnp.log(safe) - jnp.log(self._log_ratio())
        return finish(self, out, value, inside, -jnp.inf)

    def sample(self, key, sample_shape=()):
        return self.icdf(unit_draws(self, key, sample_shape))

    def cdf(self, value):
        clipped = jnp.clip(value, self.low, self.high)
        return finish(self, jnp.log(clipped / self.low) / self._log_ratio(), value)

    def icdf(self, q):
        return quantile(self, self.low * jnp.exp(q * self._log_ratio()), q)


class Normal(AbstractDistribution):
    """Normal distribution.

    Attributes:
        loc: Mean.
        scale: Standard deviation, positive.
    """

    loc: Array = eqx.field(converter=as_float)
    scale: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "real"
    has_icdf: ClassVar[bool] = True

    def support(self):
        return bounded(self, self.scale > 0, -jnp.inf, jnp.inf)

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        out = -0.5 * z**2 - jnp.log(self.scale) - _LOG_SQRT_2PI
        return finish(self, out, value, fill=-jnp.inf)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, *draw_spec(self, sample_shape))
        return finish(self, self.loc + self.scale * noise)

    def cdf(self, value):
        return finish(self, ndtr((value - self.loc) / self.scale), value)

    def icdf(self, q):
        return quantile(self, self.loc + self.scale * ndtri(q), q)


class LogNormal(AbstractDistribution):
    """Distribution of ``exp(x)`` for a normal ``x``, on ``(0, inf)``.

    Attributes:
        loc: Mean of ``log(value)``.
        scale: Standard deviation of ``log(value)``, positive.
    """

    loc: Array = eqx.field(converter=as_float)
    scale: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "lower"
    has_icdf: ClassVar[bool] = True

    def support(self):
        return bounded(self, self.scale > 0, 0.0, jnp.inf)

    def _z(self, value):
        """Standardized ``log(value)`` and the log itself, safe off-support."""
        value = _inexact_arg(value, self.loc, self.scale)
        log_value = jnp.log(_mask_fill(value > 0, value, 1.0))
        return (log_value - self.loc) / self.scale, log_value

    def log_prob(self, value):
        z, log_value = self._z(value)
        out = -0.5 * z**2 - jnp.log(self.scale) - _LOG_SQRT_2PI - log_value
        return finish(self, out, value, value > 0, -jnp.inf)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, *draw_spec(self, sample_shape))
        return finish(self, jnp.exp(self.loc + self.scale * noise))

    def cdf(self, value):
        """Evaluate the cumulative probability below each value.

        Args:
            value: Values at which to evaluate the CDF.

        Returns:
            Cumulative probabilities, including zero for nonpositive values.
        """
        return finish(self, _mask_fill(value > 0, ndtr(self._z(value)[0]), 0.0), value)

    def icdf(self, q):
        return quantile(self, jnp.exp(self.loc + self.scale * ndtri(q)), q)


class HalfNormal(AbstractDistribution):
    """Distribution of ``|x|`` for a zero-mean normal ``x``, on ``[0, inf)``.

    Attributes:
        scale: Standard deviation of the underlying normal, positive.
    """

    scale: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "lower"
    has_icdf: ClassVar[bool] = True

    def support(self):
        return bounded(self, self.scale > 0, 0.0, jnp.inf)

    def log_prob(self, value):
        z = jnp.maximum(value, 0.0) / self.scale
        out = math.log(2.0) - 0.5 * z**2 - jnp.log(self.scale) - _LOG_SQRT_2PI
        return finish(self, out, value, value >= 0, -jnp.inf)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, *draw_spec(self, sample_shape))
        return finish(self, self.scale * jnp.abs(noise))

    def cdf(self, value):
        z = jnp.maximum(value, 0.0) / self.scale
        return finish(self, erf(z / _SQRT_2), value)

    def icdf(self, q):
        return quantile(self, self.scale * _SQRT_2 * erfinv(q), q)


class Rayleigh(AbstractDistribution):
    """Rayleigh distribution on ``(0, inf)``; the density vanishes at zero.

    Attributes:
        scale: Mode of the density, positive.
    """

    scale: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "lower"
    has_icdf: ClassVar[bool] = True

    def support(self):
        return bounded(self, self.scale > 0, 0.0, jnp.inf)

    def log_prob(self, value):
        """Evaluate the log density on the open positive half-line.

        Args:
            value: Values at which to evaluate the density.

        Returns:
            Log densities with negative infinity outside the support.
        """
        inside = (value > 0) & (value < jnp.inf)
        safe = _mask_fill(inside, _inexact_arg(value, self.scale), 1.0)
        out = jnp.log(safe) - 2 * jnp.log(self.scale) - 0.5 * (safe / self.scale) ** 2
        return finish(self, out, value, inside, -jnp.inf)

    def sample(self, key, sample_shape=()):
        return self.icdf(unit_draws(self, key, sample_shape))

    def cdf(self, value):
        z = jnp.maximum(value, 0.0) / self.scale
        return finish(self, -jnp.expm1(-0.5 * z**2), value)

    def icdf(self, q):
        return quantile(self, self.scale * jnp.sqrt(-2 * jnp.log1p(-q)), q)


def _beta_log_kernel(a, b, value):
    """Beta log density at ``value`` in ``[0, 1]``, finite off the open interval."""
    log_low = jnp.log(_mask_fill(value > 0, value, 1.0))
    log_high = jnp.log1p(-_mask_fill(value < 1, value, 0.0))
    # betaln may drop the weak type of Python-number shapes, which would
    # promote a float32 value to float64; cast it to the promoted dtype.
    log_norm = betaln(a, b).astype(jnp.result_type(a, b, value))
    return (a - 1) * log_low + (b - 1) * log_high - log_norm


@jax.custom_jvp
def _beta_quantile(a, b, unit):
    """Solve ``betainc(a, b, x) = unit`` by bisection on ``log(x)``.

    Bisecting the logarithm keeps relative accuracy in the lower tail, where
    ``x`` spans many decades for a shape below one. The iteration count is
    static: twice the bit width closes the bracket from ``log(tiny)`` to
    adjacent floats anywhere in ``(0, 1)``.
    """
    dtype = jnp.result_type(a, b, unit)
    finfo = jnp.finfo(dtype)
    a, b, unit = jnp.broadcast_arrays(a, b, jnp.asarray(unit, dtype))

    def halve(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        below = betainc(a, b, jnp.exp(middle)) < unit
        return jnp.where(below, middle, low), jnp.where(below, high, middle)

    bracket = (jnp.full(unit.shape, jnp.log(finfo.tiny)), jnp.zeros(unit.shape, dtype))
    low, high = jax.lax.fori_loop(0, 2 * finfo.bits, halve, bracket)
    return jnp.exp(0.5 * (low + high))


def _beta_quantile_jvp(primals, tangents):
    """Differentiate the quantile in ``unit`` only: ``dx/du = 1 / pdf(x)``."""
    a, b, unit = primals
    a_dot, b_dot, unit_dot = tangents
    if not (isinstance(a_dot, SymbolicZero) and isinstance(b_dot, SymbolicZero)):
        raise ValueError("the Beta quantile is not differentiable in its shapes")
    root = _beta_quantile(a, b, unit)
    if isinstance(unit_dot, SymbolicZero):
        return root, jnp.zeros_like(root)
    return root, unit_dot * jnp.exp(-_beta_log_kernel(a, b, root))


_beta_quantile.defjvp(_beta_quantile_jvp, symbolic_zeros=True)


class Beta(AbstractDistribution):
    """Beta distribution on ``(0, 1)``.

    An endpoint is in the support only when its shape is exactly one, where
    the density there is finite and positive. The density is differentiable in
    the shapes. The distribution function and the quantile are not: JAX has no
    derivative of ``betainc`` in its shapes, so differentiating ``cdf`` or
    ``icdf`` in ``a`` or ``b`` raises ``ValueError``. The quantile is
    differentiable in its argument.

    Attributes:
        a: First shape, positive.
        b: Second shape, positive.
    """

    concentration1: Array = eqx.field(converter=as_float)
    concentration0: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        return bounded(
            self, (self.concentration1 > 0) & (self.concentration0 > 0), 0.0, 1.0
        )

    def log_prob(self, value):
        low = (value > 0) | ((value == 0) & (self.concentration1 == 1))
        high = (value < 1) | ((value == 1) & (self.concentration0 == 1))
        out = _beta_log_kernel(self.concentration1, self.concentration0, value)
        return finish(self, out, value, low & high, -jnp.inf)

    def sample(self, key, sample_shape=()):
        """Draw values using the positive Beta shapes.

        Args:
            key: PRNG key.
            sample_shape: Shape preceding the broadcast batch.

        Returns:
            Beta draws, with NaN where the shapes are invalid.
        """
        sample_shape, dtype = draw_spec(self, sample_shape)
        # The gamma rejection sampler behind the draw needs positive shapes.
        a, b = (
            _mask_fill(shape_ > 0, shape_, 1.0)
            for shape_ in (self.concentration1, self.concentration0)
        )
        return finish(self, jax.random.beta(key, a, b, sample_shape, dtype))

    def cdf(self, value):
        """Evaluate the cumulative probability below each value.

        Args:
            value: Values at which to evaluate the CDF.

        Returns:
            Cumulative probabilities, clipped to the lower and upper tails.
        """
        unit = jnp.minimum(jnp.maximum(value, 0.0), 1.0)
        out = betainc(self.concentration1, self.concentration0, unit)
        return finish(self, out, value)

    def icdf(self, q):
        return quantile(
            self, _beta_quantile(self.concentration1, self.concentration0, q), q
        )
