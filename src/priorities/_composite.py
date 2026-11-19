"""Records built on other distributions or on one event axis."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp, ndtr, ndtri

from priorities._distribution import (
    AbstractDistribution,
    as_float,
    debug_check,
    finish,
    is_valid,
    quantile,
    unit_draws,
)
from priorities._records import _LOG_SQRT_2PI

_BOUNDS = {
    (False, False): "real",
    (True, False): "lower",
    (False, True): "upper",
    (True, True): "interval",
}
_solve_lower = jnp.vectorize(
    lambda cholesky, rhs: solve_triangular(cholesky, rhs, lower=True),
    signature="(d,d),(d)->(d)",
)


def _optional(x):
    return None if x is None else as_float(x)


class TruncatedDistribution(AbstractDistribution):
    """A scalar distribution conditioned on ``low <= value <= high``.

    Which sides are cut is static: a side left as ``None`` is not cut and adds
    no leaf, so the uncut distribution keeps its own gradients and its own draw. The
    support is closed at a finite cut. Cuts far in a tail lose precision,
    because the normalizer is a difference of distribution-function values.

    A truncated ``Beta`` is not differentiable in the Beta shapes (see
    ``Beta``). Taking a gradient with respect to a pytree that holds one
    raises ``ValueError`` even if those gradients are unused, so keep such a
    distribution out of the differentiated partition.

    Attributes:
        base: Scalar distribution with an inverse CDF.
        low: Lower cut, or ``None``.
        high: Upper cut, or ``None``.
    """

    base_dist: AbstractDistribution
    low: Array | None = eqx.field(default=None, converter=_optional)
    high: Array | None = eqx.field(default=None, converter=_optional)
    event_dim: ClassVar[int] = 0
    has_icdf: ClassVar[bool] = True

    def __check_init__(self):
        if self.base_dist.event_dim != 0 or not self.base_dist.has_icdf:
            raise ValueError(
                "TruncatedDistribution needs a scalar base_dist with an inverse CDF"
            )

    @property
    def bounds(self) -> str:
        """Bounded sides of the support: those of the base plus the cuts."""
        lower = self.low is not None or self.base_dist.bounds in ("lower", "interval")
        upper = self.high is not None or self.base_dist.bounds in ("upper", "interval")
        return _BOUNDS[lower, upper]

    def _mass(self):
        """Base probability below the lower cut, and between the cuts."""
        below = 0.0 if self.low is None else self.base_dist.cdf(self.low)
        above = 1.0 if self.high is None else self.base_dist.cdf(self.high)
        return below, above - below

    def support(self):
        low, high = self.base_dist.support()
        low = low if self.low is None else jnp.maximum(low, self.low)
        high = high if self.high is None else jnp.minimum(high, self.high)
        valid = self._mass()[1] > 0
        low, high = (jnp.where(valid, bound, jnp.nan) for bound in (low, high))
        return tuple(jnp.broadcast_arrays(low, high))

    def log_prob(self, value):
        low, high = self.support()
        out = self.base_dist.log_prob(value) - jnp.log(self._mass()[1])
        return finish(self, out, value, (value >= low) & (value <= high), -jnp.inf)

    def sample(self, key, sample_shape=()):
        return self.icdf(unit_draws(self, key, sample_shape))

    def cdf(self, value):
        low, high = self.support()
        below, mass = self._mass()
        out = (self.base_dist.cdf(jnp.clip(value, low, high)) - below) / mass
        out = jnp.where(value >= high, 1.0, jnp.clip(out, 0.0, 1.0))
        return finish(self, out, value)

    def icdf(self, q):
        below, mass = self._mass()
        base_unit = jnp.clip(below + q * mass, 0.0, 1.0)
        return quantile(self, self.base_dist.icdf(base_unit), q)


class MultivariateNormal(AbstractDistribution):
    """Normal distribution over one event axis of length ``d``.

    For an event distribution ``cdf`` and ``icdf`` map to and from the unit
    cube rather than computing the joint distribution function: ``cdf``
    applies the standard normal distribution function to
    ``solve(scale_tril, value - loc)``, and ``icdf`` inverts it.

    Attributes:
        loc: Mean, shape ``(..., d)``.
        scale_tril: Lower-triangular factor of the covariance with a positive
            diagonal, shape ``(..., d, d)``.
    """

    loc: Array = eqx.field(converter=as_float)
    scale_tril: Array = eqx.field(converter=as_float)
    event_dim: ClassVar[int] = 1
    bounds: ClassVar[str] = "real"
    has_icdf: ClassVar[bool] = True

    def __check_init__(self):
        size = self.loc.shape[-1:]
        if self.loc.ndim < 1 or self.scale_tril.shape[-2:] != size + size:
            raise ValueError("loc must be (..., d) and scale_tril (..., d, d)")

    def _diagonal(self):
        return jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)

    def support(self):
        triangular = jnp.all(jnp.triu(self.scale_tril, 1) == 0, (-2, -1))
        valid = triangular & jnp.all(self._diagonal() > 0, -1)
        shape = jnp.broadcast_shapes(self.loc.shape, self.scale_tril.shape[:-1])
        high = jnp.broadcast_to(jnp.where(valid[..., None], jnp.inf, jnp.nan), shape)
        return -high, high

    def _status(self, arg):
        """Batch mask: 0 if usable, -inf if invalid, NaN for a NaN input."""
        nan = jnp.isnan(arg).any(-1) | jnp.isnan(self.loc).any(-1)
        nan = nan | jnp.isnan(self.scale_tril).any((-2, -1))
        status = jnp.where(is_valid(self), 0.0, -jnp.inf)
        return jnp.where(nan, jnp.nan, debug_check(self, status))

    def _mask(self, out, arg):
        return jnp.where(self._status(arg)[..., None] == 0, out, jnp.nan)

    def _whiten(self, value):
        return _solve_lower(self.scale_tril, value - self.loc)

    def _color(self, noise):
        return self.loc + jnp.einsum("...ij,...j->...i", self.scale_tril, noise)

    def log_prob(self, value):
        z = self._whiten(value)
        log_det = jnp.sum(jnp.log(self._diagonal()), -1)
        out = -0.5 * jnp.sum(z**2, -1) - log_det - z.shape[-1] * _LOG_SQRT_2PI
        status = self._status(value)
        return jnp.where(status == 0, out, status)

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape) + self.support()[0].shape
        dtype = jnp.result_type(self.loc, self.scale_tril)
        noise = jax.random.normal(key, sample_shape, dtype)
        return self._mask(self._color(noise), noise)

    def cdf(self, value):
        return self._mask(ndtr(self._whiten(value)), value)

    def icdf(self, q):
        inside = jnp.where((q >= 0) & (q <= 1), q, jnp.nan)
        return self._mask(self._color(ndtri(inside)), q)


class Mixture(AbstractDistribution):
    """Weighted mixture of ``K`` distributions of one family.

    The components are one distribution whose batch has a last axis of length ``K``,
    for example ``Normal(loc, scale)`` with parameters of shape ``(K,)``. A
    scalar mixture has a distribution function (``cdf``) but no quantile,
    so ``has_icdf`` is false and ``icdf`` raises.

    Attributes:
        mixing_logits: Unnormalized log weights, shape ``(..., K)``. They are
            normalized internally.
        components: Distribution holding the ``K`` components.
    """

    mixing_logits: Array = eqx.field(converter=as_float)
    component_distribution: AbstractDistribution
    has_icdf: ClassVar[bool] = False

    @property
    def event_dim(self) -> int:
        """Event rank of the components."""
        return self.component_distribution.event_dim

    @property
    def bounds(self) -> str:
        """Bounded sides of the components."""
        return self.component_distribution.bounds

    def _log_mix(self):
        return jax.nn.log_softmax(self.mixing_logits, -1)

    def _per_component(self, value):
        """Insert the component axis in front of the event axes."""
        return jnp.expand_dims(value, -1 - self.event_dim)

    def _over_event(self, mask):
        return mask[..., None] if self.event_dim else mask

    def support(self):
        low, high = self.component_distribution.support()
        axis = -1 - self.event_dim
        valid = jnp.isfinite(logsumexp(self.mixing_logits, -1))
        valid = self._over_event(valid)
        low = jnp.where(valid, low.min(axis), jnp.nan)
        return low, jnp.where(valid, high.max(axis), jnp.nan)

    def log_prob(self, value):
        parts = self.component_distribution.log_prob(self._per_component(value))
        terms = self._log_mix() + parts
        some = jnp.any(terms > -jnp.inf, -1) & is_valid(self)
        out = logsumexp(jnp.where(some[..., None], terms, 0.0), -1)
        out = jnp.where(some, debug_check(self, out), -jnp.inf)
        nan = jnp.isnan(parts).any(-1) | jnp.isnan(self.mixing_logits).any(-1)
        return jnp.where(nan, jnp.nan, out)

    def sample(self, key, sample_shape=()):
        key_index, key_draw = jax.random.split(key)
        draws = self.component_distribution.sample(key_draw, sample_shape)
        lead = draws.shape[: draws.ndim - 1 - self.event_dim]
        index = jax.random.categorical(key_index, self.mixing_logits, shape=lead)
        index = self._over_event(index[..., None])
        out = jnp.take_along_axis(draws, index, -1 - self.event_dim)
        out = out.squeeze(-1 - self.event_dim)
        return jnp.where(self._over_event(is_valid(self)), out, jnp.nan)

    def cdf(self, value):
        if self.event_dim:
            raise NotImplementedError("a Mixture over an event axis has no cdf")
        parts = self.component_distribution.cdf(self._per_component(value))
        out = jnp.sum(jnp.exp(self._log_mix()) * parts, -1)
        return jnp.where(is_valid(self), debug_check(self, out), jnp.nan)

    def icdf(self, q):
        raise NotImplementedError("Mixture has no quantile; has_icdf is False")
