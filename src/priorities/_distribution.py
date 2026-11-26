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

        An endpoint belongs to the support when the density there is finite.

        Returns:
            Lower and upper bounds of shape ``batch + event``, infinite on
            unbounded sides and NaN where the parameters are outside their domain.
        """

    @abstractmethod
    def log_prob(self, value: Array) -> Array:
        """Return the log density against ``d(value)``, reduced over the event.

        Args:
            value: Points of shape ``sample + batch + event`` at which to evaluate
                the density.

        Returns:
            Log densities of shape ``sample + batch``. The result is exactly
            ``-inf`` outside the support, with a zero gradient there, and for
            parameters outside their domain. It is NaN when the value or a
            parameter is NaN.
        """

    @abstractmethod
    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw values of shape ``sample_shape + batch + event``.

        Args:
            key: PRNG key.
            sample_shape: Shape of the sample axes preceding the batch and event axes.

        Returns:
            Draws in the promoted dtype of the parameters, with NaN for parameters
            outside their domain.
        """

    @abstractmethod
    def cdf(self, value: Array) -> Array:
        """Map a value to the unit cube of the event shape.

        For a scalar distribution this is the cumulative distribution function.

        Args:
            value: Points to map to the unit cube.

        Returns:
            Unit coordinates with NaN for parameters outside their domain.

        Raises:
            NotImplementedError: The distribution has no CDF.
        """

    @abstractmethod
    def icdf(self, q: Array) -> Array:
        """Map a point of the unit cube to a value: uniform in, the distribution out.

        For a scalar distribution this is the quantile function.

        Args:
            q: Unit coordinates to map to the support.

        Returns:
            Values on the support, including exact endpoints at 0 and 1, with
            NaN outside ``[0, 1]`` or for parameters outside their domain.

        Raises:
            NotImplementedError: The distribution has no inverse CDF.
        """


def as_float(x) -> Array:
    """Convert a parameter to a floating array, keeping weak types.

    Args:
        x: Scalar or array-like distribution parameter.

    Returns:
        An array with an inexact dtype. Integer and boolean inputs are promoted.
    """
    x = jnp.asarray(x)
    return x if jnp.issubdtype(x.dtype, jnp.inexact) else x * 1.0


def is_valid(dist: AbstractDistribution) -> Array:
    """Check whether the distribution parameters are inside their domain.

    Args:
        dist: Distribution whose support encodes parameter validity.

    Returns:
        A boolean mask over the batch, true where all event bounds are valid.
    """
    low, high = dist.support()
    invalid = jnp.isnan(low) | jnp.isnan(high)
    return ~(invalid.any(-1) if dist.event_dim else invalid)


def debug_check(dist: AbstractDistribution, out: Array) -> Array:
    """Check parameter validity when runtime debugging is enabled.

    Args:
        dist: Distribution supplying the parameter domain.
        out: Result to carry the runtime check.

    Returns:
        The supplied result with the debug check attached when enabled.

    Raises:
        RuntimeError: Debug checks are enabled and a parameter is invalid.
    """
    if DEBUG_CHECKS:
        message = f"{type(dist).__name__}: parameters outside their domain"
        out = eqx.error_if(out, ~is_valid(dist), message)
    return out


def draw_spec(dist: AbstractDistribution, shape: tuple[int, ...]):
    """Determine the shape and dtype of scalar-distribution draws.

    Args:
        dist: Scalar distribution supplying the batch parameters.
        shape: Sample shape preceding the batch.

    Returns:
        The shape ``shape + batch`` and promoted parameter dtype.
    """
    leaves = jax.tree_util.tree_leaves(dist)
    batch = jnp.broadcast_shapes(*(leaf.shape for leaf in leaves))
    return tuple(shape) + batch, jnp.result_type(*leaves)


def _constant(ref, value):
    """Build a scalar at the reference dtype, preserving its weak type."""
    return jnp.full_like(ref, jnp.result_type(ref).type(value), shape=())


def _mask_fill(mask, value, fill):
    return jnp.where(mask, value, _constant(value, fill))


def _inexact_arg(value, *parameters):
    """Promote an argument before a floating operation or sentinel fill."""
    return jnp.asarray(value, dtype=jnp.result_type(value, *parameters, 1.0))


def _uniform(key, shape, dtype, minval):
    """Draw units with both endpoints typed to match the draw."""
    low, high = dtype.type(minval), dtype.type(1)
    return jax.random.uniform(key, shape, dtype, minval=low, maxval=high)


def bounded(dist: AbstractDistribution, valid, low, high) -> tuple[Array, Array]:
    """Broadcast scalar-distribution bounds over the batch.

    Args:
        dist: Distribution supplying the batch shape and precision.
        valid: Mask of parameters within the distribution's domain.
        low: Lower bound, scalar or array.
        high: Upper bound, scalar or array.

    Returns:
        Lower and upper bounds, each NaN where the parameters are invalid.
    """
    shape, _ = draw_spec(dist, ())
    ref = sum(jnp.zeros_like(x, shape=()) for x in jax.tree_util.tree_leaves(dist))
    bounds = (x if isinstance(x, Array) else _constant(ref, x) for x in (low, high))
    return tuple(jnp.broadcast_to(_mask_fill(valid, x, jnp.nan), shape) for x in bounds)


def finish(
    dist: AbstractDistribution, out, arg=None, inside=True, fill=jnp.nan
) -> Array:
    """Mask the result of a scalar distribution.

    Args:
        dist: Distribution supplying the parameter domain.
        out: Unmasked result.
        arg: Evaluation argument, or None for a draw.
        inside: Mask of values inside the domain of the evaluation.
        fill: Value outside that domain or for invalid parameters.

    Returns:
        The masked result, with NaN where the argument or a parameter is NaN.
    """
    inputs = jax.tree.leaves((arg, dist))
    nan = sum(jnp.isnan(x) for x in inputs) > 0
    out = _mask_fill(is_valid(dist) & inside, debug_check(dist, out), fill)
    # Adding typed zeros promotes to the inputs' dtype and keeps weak types.
    return _mask_fill(~nan, out, jnp.nan) + sum(jnp.zeros_like(x) for x in inputs)


def unit_draws(dist: AbstractDistribution, key: Array, shape: tuple[int, ...]) -> Array:
    """Draw uniforms on the open unit interval for a quantile-based sample.

    Args:
        dist: Distribution supplying the batch shape and precision.
        key: PRNG key.
        shape: Sample shape preceding the batch.

    Returns:
        Uniform samples in the distribution's dtype.
    """
    shape, dtype = draw_spec(dist, shape)
    return _uniform(key, shape, dtype, jnp.finfo(dtype).tiny)


def quantile(dist: AbstractDistribution, out, unit) -> Array:
    """Finish a scalar inverse CDF.

    Args:
        dist: Distribution supplying the support and parameter domain.
        out: Unmasked quantiles.
        unit: Unit probabilities at which the quantiles were evaluated.

    Returns:
        Quantiles clipped to the support, with exact endpoints and NaN off
        the unit interval.
    """
    low, high = dist.support()
    inside = jnp.clip(out, low, high)
    out = jnp.where(unit <= 0, low, jnp.where(unit >= 1, high, inside))
    return finish(dist, out, unit, (unit >= 0) & (unit <= 1))
