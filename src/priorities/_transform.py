"""The default transform between a distribution's support and the real line."""

import jax
import jax.numpy as jnp
from jax import Array

from priorities._distribution import AbstractDistribution


def _check_bounds(dist: AbstractDistribution) -> str:
    if dist.bounds not in ("real", "lower", "upper", "interval"):
        raise ValueError(f"unknown bounds {dist.bounds!r} on {type(dist).__name__}")
    return dist.bounds


def to_real(dist: AbstractDistribution, value: Array) -> Array:
    """Map a value in the support of ``dist`` to unconstrained coordinates.

    The transform follows ``dist.bounds``: identity for ``"real"``,
    ``log(value - low)`` for ``"lower"``, ``log(high - value)`` for
    ``"upper"`` and the logit of the fractional position for ``"interval"``.
    It is elementwise, also for a distribution with an event axis.

    Args:
        dist: The distribution whose support is mapped.
        value: Values inside the support.

    Returns:
        Unconstrained coordinates of the shape of ``value``.
    """
    bounds = _check_bounds(dist)
    low, high = dist.support()
    if bounds == "real":
        return value
    if bounds == "lower":
        return jnp.log(value - low)
    if bounds == "upper":
        return jnp.log(high - value)
    return jnp.log(value - low) - jnp.log(high - value)


def from_real(dist: AbstractDistribution, x: Array) -> tuple[Array, Array]:
    """Map unconstrained coordinates to a value in the support of ``dist``.

    This inverts ``to_real``: identity, ``low + exp(x)``, ``high - exp(x)`` or
    ``low + (high - low) * sigmoid(x)``.

    Args:
        dist: The distribution whose support is the target.
        x: Unconstrained coordinates.

    Returns:
        The value and ``log_volume``, the log of ``|d value / d x|``, summed
        over the event axis when the distribution has one.
    """
    bounds = _check_bounds(dist)
    low, high = dist.support()
    if bounds == "real":
        value, log_volume = x, jnp.zeros_like(x)
    elif bounds == "lower":
        value, log_volume = low + jnp.exp(x), x
    elif bounds == "upper":
        value, log_volume = high - jnp.exp(x), x
    else:
        width = high - low
        value = low + width * jax.nn.sigmoid(x)
        log_volume = jnp.log(width) + jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x)
    log_volume = jnp.broadcast_to(log_volume, value.shape)
    return value, log_volume.sum(-1) if dist.event_dim else log_volume


def real_log_prob(dist: AbstractDistribution, x: Array) -> Array:
    """Return the log density of ``dist`` against ``d(x)``.

    This is ``dist.log_prob(value) + log_volume`` for the default transform,
    the density a gradient sampler reads in unconstrained coordinates.

    Args:
        dist: The distribution to evaluate.
        x: Unconstrained coordinates.

    Returns:
        The log density, reduced over the event axis.
    """
    value, log_volume = from_real(dist, x)
    return dist.log_prob(value) + log_volume
