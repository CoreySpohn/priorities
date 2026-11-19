"""The distribution interface and the helpers shared by the scalar records."""

import os
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

# Read once at import. When on, a distribution evaluated with parameters outside their
# domain raises through eqx.error_if instead of returning -inf or NaN.
DEBUG_CHECKS = os.environ.get("PRIORITIES_DEBUG_CHECKS", "0") == "1"


class AbstractDistribution(eqx.Module):
    """A probability distribution: a density against ``d(value)`` with its draws.

    A distribution is a record of array parameters. It is elementwise (``event_dim``
    0) or has one event axis (``event_dim`` 1). Shapes follow one rule: a
    value is ``sample + batch + event``, where ``batch`` is the broadcast of
    the parameter shapes. The output dtype is the promotion of the argument
    and the parameters, so a distribution built from Python floats returns the dtype
    of its argument.

    Subclass this directly when the mathematics is new. A named distribution that only
    fixes defaults of an existing record is a function returning that record.

    Attributes:
        event_dim: Number of event axes, 0 or 1. Static.
        bounds: Which sides of the support are bounded: ``"real"``,
            ``"lower"``, ``"upper"`` or ``"interval"``. Static, so the default
            transform is chosen before tracing.
        has_icdf: Whether ``cdf`` and ``icdf`` form a bijection
            between the support and the unit cube. Static; test it before
            building a sampler that needs the map.
    """

    event_dim: eqx.AbstractVar[int]
    bounds: eqx.AbstractVar[str]
    has_icdf: eqx.AbstractVar[bool]

    def __check_init__(self):
        """Refuse concrete parameters outside their domain; skip tracers."""
        valid = is_valid(self)
        if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
            raise ValueError(f"parameters outside their domain: {self}")

    @abstractmethod
    def support(self) -> tuple[Array, Array]:
        """Return the bounds ``(low, high)`` of the support.

        Each has shape ``batch + event`` and is infinite on an unbounded side.
        An endpoint belongs to the support when the density there is finite.
        Both are NaN where the parameters are outside their domain.
        """

    @abstractmethod
    def log_prob(self, value: Array) -> Array:
        """Return the log density against ``d(value)``, reduced over the event.

        The result is exactly ``-inf`` outside the support, with a zero
        gradient there, and for parameters outside their domain. It is NaN
        when the value or a parameter is NaN.
        """

    @abstractmethod
    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw values of shape ``sample_shape + batch + event``.

        The dtype is the promotion of the parameters. Draws are NaN for
        parameters outside their domain.
        """

    @abstractmethod
    def cdf(self, value: Array) -> Array:
        """Map a value to the unit cube of the event shape.

        For a scalar distribution this is the cumulative distribution function. NaN for
        parameters outside their domain. Raises ``NotImplementedError`` when
        the distribution has no such function.
        """

    @abstractmethod
    def icdf(self, q: Array) -> Array:
        """Map a point of the unit cube to a value: uniform in, the distribution out.

        For a scalar distribution this is the quantile function. It returns the support
        endpoints at 0 and 1 and NaN outside ``[0, 1]`` or for parameters
        outside their domain. Raises ``NotImplementedError`` when
        ``has_icdf`` is false.
        """


def as_float(x) -> Array:
    """Convert a parameter to a floating array, keeping weak types."""
    x = jnp.asarray(x)
    return x if jnp.issubdtype(x.dtype, jnp.inexact) else x * 1.0


def is_valid(dist: AbstractDistribution) -> Array:
    """Return the batch-shaped mask of parameters inside their domain."""
    low, high = dist.support()
    invalid = jnp.isnan(low) | jnp.isnan(high)
    return ~(invalid.any(-1) if dist.event_dim else invalid)


def debug_check(dist: AbstractDistribution, out: Array) -> Array:
    """Raise at run time for invalid parameters when the debug flag is on."""
    if DEBUG_CHECKS:
        message = f"{type(dist).__name__}: parameters outside their domain"
        out = eqx.error_if(out, ~is_valid(dist), message)
    return out


def draw_spec(dist: AbstractDistribution, shape: tuple[int, ...]):
    """Return the shape ``shape + batch`` and dtype of scalar-distribution draws."""
    leaves = jax.tree_util.tree_leaves(dist)
    batch = jnp.broadcast_shapes(*(leaf.shape for leaf in leaves))
    return tuple(shape) + batch, jnp.result_type(*leaves)


def bounded(dist: AbstractDistribution, valid, low, high) -> tuple[Array, Array]:
    """Broadcast scalar-distribution bounds over the batch; NaN where not ``valid``."""
    shape, _ = draw_spec(dist, ())
    low, high = (jnp.where(valid, bound, jnp.nan) for bound in (low, high))
    return jnp.broadcast_to(low, shape), jnp.broadcast_to(high, shape)


def finish(
    dist: AbstractDistribution, out, arg=0.0, inside=True, fill=jnp.nan
) -> Array:
    """Mask the result of a scalar distribution.

    ``fill`` replaces ``out`` outside ``inside`` and for invalid parameters;
    NaN replaces it where ``arg`` or a parameter is NaN.
    """
    inputs = (arg, *jax.tree_util.tree_leaves(dist))
    nan = sum(jnp.isnan(x) for x in inputs) > 0
    out = jnp.where(is_valid(dist) & inside, debug_check(dist, out), fill)
    # Adding typed zeros promotes to the inputs' dtype and keeps weak types.
    return jnp.where(nan, jnp.nan, out) + sum(jnp.zeros_like(x) for x in inputs)


def unit_draws(dist: AbstractDistribution, key: Array, shape: tuple[int, ...]) -> Array:
    """Draw uniforms on the open unit interval for a quantile-based sample."""
    shape, dtype = draw_spec(dist, shape)
    tiny = jnp.finfo(dtype).tiny
    return jax.random.uniform(key, shape, dtype, minval=tiny)


def quantile(dist: AbstractDistribution, out, unit) -> Array:
    """Finish a scalar ``icdf``: exact endpoints, NaN off ``[0, 1]``."""
    low, high = dist.support()
    out = jnp.where(
        unit <= 0, low, jnp.where(unit >= 1, high, jnp.clip(out, low, high))
    )
    return finish(dist, out, unit, (unit >= 0) & (unit <= 1))
