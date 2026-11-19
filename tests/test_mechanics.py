"""JAX mechanics: keys, batching, tracing and precision."""

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import priorities as pr
from priorities.testing import icdf_round_trip


def test_same_key_gives_the_same_draws(dist):
    """Draws are a pure function of the key.

    Fails if a record reuses a key internally or reads global state. Exact.
    """
    first = dist.sample(jax.random.key(0), (5,))
    np.testing.assert_array_equal(first, dist.sample(jax.random.key(0), (5,)))
    assert not bool(jnp.all(first == dist.sample(jax.random.key(1), (5,))))


def test_vmap_and_jit_equal_plain_calls(dist):
    """vmap over values equals a loop, and jit equals eager.

    Fails if a method branches on a value in Python or depends on the batch
    it is called in. Tolerance: 4 units in the last place, the difference XLA
    fusion can introduce between compiled and eager code.
    """
    values = dist.sample(jax.random.key(0), (4,))
    looped = jnp.stack([dist.log_prob(value) for value in values])
    for batched in (jax.vmap(dist.log_prob), eqx.filter_jit(dist.log_prob)):
        np.testing.assert_array_max_ulp(batched(values), looped, maxulp=4)


def test_value_shape_is_sample_then_batch_then_event():
    """Shapes follow sample + batch + event, with batch the parameter broadcast.

    Fails if a record drops a parameter axis, orders the axes differently or
    places the mixture component axis anywhere but last in the batch.
    """
    key = jax.random.key(0)
    normal = pr.Normal(jnp.zeros((3, 1)), jnp.ones(4))
    assert normal.sample(key, (2,)).shape == (2, 3, 4)
    assert normal.support()[0].shape == (3, 4)
    assert normal.log_prob(jnp.zeros((2, 3, 4))).shape == (2, 3, 4)
    cut = pr.TruncatedDistribution(pr.Rayleigh(jnp.ones(3)), high=jnp.full((2, 1), 2.0))
    assert cut.sample(key, (5,)).shape == (5, 2, 3)
    factor = jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
    event = pr.MultivariateNormal(jnp.zeros(3), factor)
    assert event.sample(key, (2,)).shape == (2, 4, 3)
    assert event.log_prob(jnp.zeros((2, 4, 3))).shape == (2, 4)
    mixture = pr.Mixture(jnp.zeros((4, 3)), pr.Normal(jnp.zeros(3), jnp.ones((4, 3))))
    assert mixture.sample(key, (2,)).shape == (2, 4)
    assert mixture.log_prob(jnp.zeros((2, 4))).shape == (2, 4)
    assert mixture.support()[0].shape == (4,)
    stacked = pr.Mixture(jnp.zeros(4), event)
    assert stacked.sample(key, (2,)).shape == (2, 3)
    assert stacked.log_prob(jnp.zeros((2, 3))).shape == (2,)


def test_batched_parameters_equal_a_map_over_dists():
    """A distribution with a parameter axis equals vmap over scalar distributions.

    Fails if broadcasting pairs a value with the wrong parameter. Tolerance:
    4 units in the last place (compiled against eager).
    """
    scales = jnp.array([0.5, 1.0, 2.0])
    batched = pr.Rayleigh(scales).log_prob(0.7)
    mapped = jax.vmap(lambda scale: pr.Rayleigh(scale).log_prob(0.7))(scales)
    np.testing.assert_array_max_ulp(batched, mapped, maxulp=4)


def test_parameters_are_array_leaves_and_do_not_retrace():
    """A two-parameter distribution has two leaves, and new values reuse one trace.

    Fails if a parameter is stored as a Python float (static: every value
    recompiles) or if an unset cut adds a leaf.
    """
    leaves = jax.tree_util.tree_leaves
    assert len(leaves(pr.Beta(0.867, 3.03))) == 2
    assert len(leaves(pr.TruncatedDistribution(pr.Beta(0.867, 3.03)))) == 2
    assert len(leaves(pr.TruncatedDistribution(pr.Beta(0.867, 3.03), high=0.95))) == 3
    traces = []

    @eqx.filter_jit
    def log_prob(dist, value):
        traces.append(1)
        return dist.log_prob(value)

    for a, b in ((0.867, 3.03), (2.0, 5.0), (0.5, 0.5)):
        log_prob(pr.Beta(a, b), 0.3)
    assert len(traces) == 1
    updated = eqx.tree_at(
        lambda dist: dist.concentration1, pr.Beta(0.867, 3.03), jnp.asarray(2.0)
    )
    gradient = jax.grad(lambda dist: dist.log_prob(0.3))(updated)
    assert (
        float(gradient.concentration1) != 0.0 and float(gradient.concentration0) != 0.0
    )


FLOAT32_DISTRIBUTIONS = {
    "uniform": lambda: pr.Uniform(0.0, 1.0),
    "log_uniform": lambda: pr.LogUniform(0.1, 2.0),
    "normal": lambda: pr.Normal(0.0, 1.0),
    "log_normal": lambda: pr.LogNormal(0.0, 1.0),
    "half_normal": lambda: pr.HalfNormal(1.0),
    "rayleigh": lambda: pr.Rayleigh(1.0),
    "beta": lambda: pr.Beta(0.867, 3.03),
    "truncated": lambda: pr.TruncatedDistribution(pr.Beta(0.867, 3.03), high=0.95),
}


@pytest.mark.parametrize("name", list(FLOAT32_DISTRIBUTIONS))
def test_float32_in_gives_float32_out(name):
    """With x64 on, a record built from Python numbers returns its argument's dtype.

    Fails if a record forces float64 (a constant typed as an array, or a
    support bound that loses its weak type), which would silently double the
    precision and the cost of a float32 pipeline.
    """
    dist = FLOAT32_DISTRIBUTIONS[name]()
    value = jnp.asarray(0.3, jnp.float32)
    outputs = [dist.log_prob(value), dist.cdf(value), pr.to_real(dist, value)]
    outputs += [*pr.from_real(dist, value), pr.real_log_prob(dist, value)]
    if dist.has_icdf:
        outputs.append(dist.icdf(value))
    assert {out.dtype for out in outputs} == {jnp.dtype(jnp.float32)}
    assert dist.sample(jax.random.key(0)).dtype == jnp.float64


def test_strongly_typed_parameters_set_the_dtype():
    """The output dtype is the promotion of the argument and the parameters.

    Fails if float32 parameters are promoted by a draw, or float64 parameters
    are truncated to a float32 argument.
    """
    key = jax.random.key(0)
    single = pr.Normal(jnp.float32(0.0), jnp.float32(1.0))
    assert single.sample(key).dtype == jnp.float32
    assert single.log_prob(jnp.float64(0.3)).dtype == jnp.float64
    double = pr.Normal(jnp.float64(0.0), jnp.float64(1.0))
    assert double.log_prob(jnp.float32(0.3)).dtype == jnp.float64
    # A list parameter is a strongly typed array, so a float32 mixture is
    # built from float32 arrays.
    weights, locs = jnp.zeros(2, jnp.float32), jnp.array([0.0, 1.0], jnp.float32)
    mixture = pr.Mixture(weights, pr.Normal(locs, 1.0))
    value = jnp.float32(0.3)
    for out in (mixture.log_prob(value), mixture.cdf(value), mixture.sample(key)):
        assert out.dtype == jnp.float32
    whole = pr.Uniform(0, 1)
    assert jnp.issubdtype(whole.sample(key).dtype, jnp.floating)


def test_float32_beta_quantile_floor():
    """The float32 Beta quantile round trip stays at its measured floor.

    Fails if the float32 path loses more than rounding of the inputs
    explains. Tolerance: 5e-5 relative to min(u, 1 - u) after the 4-ulp input
    allowance, spike (measured 6.5e-6 for both distributions).
    """
    shapes = jnp.float32(0.867), jnp.float32(3.03)
    beta = pr.Beta(*shapes)
    error = icdf_round_trip(beta, jnp.float32)
    assert error.dtype == jnp.float32
    assert float(jnp.max(error)) < 5e-5
    cut = pr.TruncatedDistribution(beta, high=jnp.float32(0.95))
    assert float(jnp.max(icdf_round_trip(cut, jnp.float32))) < 5e-5


def test_no_dist_or_array_exists_at_import():
    """Importing the package builds no distribution and no array.

    Fails if a module-level default is converted before the caller enables
    x64, which would freeze it at float32 (0.867 would be off by 1.6e-8).
    """
    modules = [m for name, m in sys.modules.items() if name.startswith("priorities")]
    assert len(modules) >= 5
    for module in modules:
        for name, item in vars(module).items():
            assert not isinstance(item, jax.Array | pr.AbstractDistribution), name
