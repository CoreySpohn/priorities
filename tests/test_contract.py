"""The distribution contract: draws, inverse CDFs and support against the density."""

import dataclasses
import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import BUILDERS, CHOLESKY, DIRECT_DRAWS, MEAN, WEIGHTS

import priorities as pr
from priorities.testing import (
    KS_SCALE,
    check_distribution,
    icdf_round_trip,
    ks_distance,
)

# Draw count at which a 5 percent scale error is rejected in every seed tried.
N_DRAWS = 200_000
QUANTILE_DRAWS = 2000
CHECKED_BY_DRAWS = [*DIRECT_DRAWS, "multivariate_normal", "event_mixture"]


@pytest.mark.parametrize("name", list(BUILDERS))
def test_distribution_contract(name, dists):
    """Every record passes check_distribution (see its docstring for the checks).

    Fails if sample, cdf or icdf disagrees with log_prob, if the
    density leaks outside the support, or if a tail of the quantile is lost.
    Tolerances: derivative rtol 1e-10, floating-point; unit round trip 1e-11
    relative to min(u, 1 - u), spike (worst measured 1.2e-14, for the Beta
    distributions); Kolmogorov-Smirnov threshold 2.69 / sqrt(n), monte-carlo, a
    false-rejection rate of 1e-6 per distribution, seed 0. Records that draw through
    their own quantile use 2000 draws, because the round trip already covers
    them; the others use 200,000.
    """
    count = N_DRAWS if name in CHECKED_BY_DRAWS else QUANTILE_DRAWS
    check_distribution(dists[name], jax.random.key(0), n_draws=count)


MUTANTS = {
    "normal": lambda: pr.Normal(0.7, 1.3 * 1.05),
    "log_normal": lambda: pr.LogNormal(0.2, 0.6 * 1.05),
    "half_normal": lambda: pr.HalfNormal(1.7 * 1.05),
    "beta": lambda: pr.Beta(0.867 * 1.05, 3.03),
    "multivariate_normal": lambda: pr.MultivariateNormal(
        MEAN, jnp.array(CHOLESKY) * 1.05
    ),
    # 5 percent on every component scale reaches only 2.8 to 4.1 here; the
    # control for the mixture is 10 percent (5.7 to 7.1 over five seeds).
    "mixture": lambda: pr.Mixture(
        jnp.log(jnp.array(WEIGHTS)),
        pr.Normal(jnp.array([-2.0, 0.0, 3.0]), jnp.array([0.5, 1.0, 2.0]) * 1.1),
    ),
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_draw_check_rejects_a_wrong_scale(name, dists):
    """Mutation control: draws with a scale 5 percent off fail the draw check.

    Fails if the Kolmogorov-Smirnov check cannot see a 5 percent scale error,
    which is the case at 20,000 draws (scaled distance 1.6 to 2.1 against the
    threshold 2.69). At 200,000 draws the scaled distance is 5.4 to 11 over
    five seeds for every distribution here. Basis: mutation; seed 0.
    """
    draws = MUTANTS[name]().sample(jax.random.key(0), (N_DRAWS,))
    assert float(ks_distance(dists[name], draws)) > KS_SCALE / math.sqrt(N_DRAWS)


def test_event_mixture_draws_follow_a_projection(dists):
    """Projected draws of the event mixture follow the projected mixture.

    A fixed direction maps a mixture of multivariate normals to a scalar
    normal mixture with a closed-form distribution function. Fails if the
    component index is applied on the wrong axis or the weights are ignored
    (equal weights give a scaled distance near 30). Threshold 2.69 / sqrt(n),
    monte-carlo, 200,000 draws, seed 0.
    """
    dist = dists["event_mixture"]
    direction = jnp.array([0.6, -0.3, 0.74])
    parts = dist.component_distribution
    scale = jnp.linalg.norm(
        jnp.einsum("i,kij->kj", direction, parts.scale_tril), axis=1
    )
    projected = pr.Mixture(dist.mixing_logits, pr.Normal(parts.loc @ direction, scale))
    draws = dist.sample(jax.random.key(0), (N_DRAWS,)) @ direction
    assert float(ks_distance(projected, draws)) < KS_SCALE / math.sqrt(N_DRAWS)
    equal = eqx.tree_at(lambda mix: mix.mixing_logits, dist, jnp.zeros(2))
    wrong = equal.sample(jax.random.key(0), (N_DRAWS,)) @ direction
    assert float(ks_distance(projected, wrong)) > KS_SCALE / math.sqrt(N_DRAWS)


def test_beta_quantile_keeps_the_lower_tail_for_a_small_shape():
    """The unit round trip holds to 1e-30 for a Beta with shape 0.2.

    Fails for a root search on the linear axis (10 percent error in
    probability at u = 1e-6). Tolerance: 1e-11 relative to min(u, 1 - u),
    spike (measured 8e-15).
    """
    assert float(jnp.max(icdf_round_trip(pr.Beta(0.2, 3.03)))) < 1e-11


class Logistic(pr.AbstractDistribution):
    """A distribution defined outside the library, without a distribution function."""

    scale: jax.Array = eqx.field(converter=jnp.asarray)
    factor: float = 1.0
    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "real"
    has_icdf: ClassVar[bool] = False

    def support(self):
        bound = jnp.where(self.scale > 0, jnp.inf, jnp.nan)
        return -bound, bound

    def log_prob(self, value):
        z = value / self.scale
        return math.log(self.factor) - z - 2 * jax.nn.softplus(-z) - jnp.log(self.scale)

    def sample(self, key, sample_shape=()):
        return self.scale * jax.random.logistic(key, sample_shape)

    def cdf(self, value):
        raise NotImplementedError("Logistic has no cdf here")

    def icdf(self, q):
        raise NotImplementedError("Logistic has no icdf here")


def test_subclass_without_distribution_function_is_checked_by_quadrature():
    """check_distribution integrates a subclass that has no cdf, and refuses a mutant.

    Fails if the fallback is skipped: a density 0.1 percent too large must be
    rejected. Tolerance 1e-10 on the integral, floating-point: the trapezoid
    rule converges geometrically for this integrand (the three-level ladder
    below reaches 1e-13 at the default node count).
    """
    key = jax.random.key(0)
    check_distribution(Logistic(0.8), key, n_draws=QUANTILE_DRAWS)
    with pytest.raises(AssertionError, match="integrates"):
        check_distribution(Logistic(0.8, factor=1.001), key, n_draws=QUANTILE_DRAWS)
    with pytest.raises(ValueError, match="outside their domain"):
        Logistic(-1.0)


def test_unknown_bounds_are_refused():
    """A distribution declaring an unknown bounds string is refused by the transform.

    Fails if a typo in a subclass silently selects the interval transform.
    """

    class Odd(Logistic):
        bounds: ClassVar[str] = "open"

    with pytest.raises(ValueError, match="unknown bounds"):
        pr.to_real(Odd(0.8), 0.3)


ENDPOINTS = [
    ("uniform", -1.5, -math.log(4.0)),
    ("uniform", 2.5, -math.log(4.0)),
    ("log_uniform", 0.3, -math.log(0.3 * math.log(40.0 / 0.3))),
    ("half_normal", 0.0, 0.5 * math.log(2 / math.pi) - math.log(1.7)),
    ("rayleigh", 0.0, -math.inf),
    ("log_normal", 0.0, -math.inf),
    ("beta", 0.0, -math.inf),
    ("beta", 1.0, -math.inf),
    ("truncated_beta", 0.0, -math.inf),
    ("normal", math.inf, -math.inf),
    ("rayleigh", math.inf, -math.inf),
]


@pytest.mark.parametrize(("name", "value", "expected"), ENDPOINTS)
def test_endpoint_is_in_the_support_when_the_density_is_finite(
    name, value, expected, dists
):
    """log_prob at each support endpoint matches the per-record table.

    Fails if a closed endpoint is excluded (a draw equal to the bound would
    get zero density) or an open one included (NaN or +inf at a singular
    point). Finite entries are closed forms typed here; tolerance rtol 1e-14,
    floating-point. Infinite entries are exact, eager and under jit.
    """
    dist = dists[name]
    for evaluate in (dist.log_prob, eqx.filter_jit(dist.log_prob)):
        got = float(evaluate(jnp.asarray(value)))
        if math.isinf(expected):
            assert got == expected
        else:
            np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0)


def test_cut_and_unit_shape_endpoints_are_closed(dists):
    """A finite cut is closed, and a Beta endpoint with shape one is closed.

    Fails if TruncatedDistribution drops the density at its own cut, so that
    icdf(1) lands on a point of zero density. Expected values are the
    base density minus the log mass, typed from closed forms; rtol 1e-12,
    floating-point.
    """
    cut = dists["truncated_beta"]
    assert float(cut.icdf(1.0)) == 0.95
    assert math.isfinite(float(cut.log_prob(cut.icdf(1.0))))
    two_sided = dists["truncated_normal"]
    ends = two_sided.icdf(jnp.array([0.0, 1.0]))
    np.testing.assert_array_equal(ends, [-1.0, 2.0])
    assert bool(jnp.all(jnp.isfinite(two_sided.log_prob(ends))))
    np.testing.assert_allclose(pr.Beta(1.0, 3.0).log_prob(0.0), math.log(3.0), 1e-12)
    np.testing.assert_allclose(pr.Beta(2.0, 1.0).log_prob(1.0), math.log(2.0), 1e-12)


def test_icdf_endpoints_and_range(unit_dist):
    """icdf gives the support ends at 0 and 1 and NaN outside [0, 1].

    Fails if the quantile overshoots a bound by rounding or clips an invalid
    unit into the support. Exact.
    """
    low, high = unit_dist.support()
    got = unit_dist.icdf(jnp.array([0.0, 1.0, -0.1, 1.1, jnp.nan]))
    np.testing.assert_array_equal(got[:2], jnp.stack([low, high]))
    assert bool(jnp.all(jnp.isnan(got[2:])))


# One parameter of each record, moved outside its domain.
INVALID = {
    "uniform": (lambda m: m.high, -2.0),
    "log_uniform": (lambda m: m.low, -0.3),
    "normal": (lambda m: m.scale, -1.3),
    "log_normal": (lambda m: m.scale, -0.6),
    "half_normal": (lambda m: m.scale, -1.7),
    "rayleigh": (lambda m: m.scale, 0.0),
    "beta": (lambda m: m.concentration1, -0.867),
    "truncated_normal": (lambda m: m.low, 3.0),
    "truncated_upper": (lambda m: m.base_dist.scale, -1.3),
    "truncated_beta": (lambda m: m.base_dist.concentration1, -0.867),
    "mixture": (lambda m: m.component_distribution.scale, [0.5, -1.0, 2.0]),
    "multivariate_normal": (lambda m: m.scale_tril, -np.array(CHOLESKY)),
    "event_mixture": (lambda m: m.mixing_logits, [-np.inf, -np.inf]),
}


def _invalid(name, dist):
    where, value = INVALID[name]
    return eqx.tree_at(where, dist, jnp.asarray(value))


def _interior(dist):
    return dist.sample(jax.random.key(0))


@pytest.mark.parametrize("name", list(INVALID))
def test_invalid_parameters_give_zero_mass(name, dists, no_debug):
    """Parameters outside their domain give -inf from log_prob and NaN elsewhere.

    Fails if an invalid parameter gives a finite density (a sampler would
    accept the step) or raises inside a compiled chain. The parameters reach
    the distribution as traced arguments. A NaN parameter gives NaN, not -inf. Exact.
    """
    dist = dists[name]
    value = _interior(dist)
    bad = _invalid(name, dist)

    @eqx.filter_jit
    def evaluate(record, point):
        low, _ = record.support()
        return record.log_prob(point), record.sample(jax.random.key(0)), low

    log_prob, draw, low = evaluate(bad, value)
    assert float(log_prob) == -math.inf
    assert bool(jnp.all(jnp.isnan(draw))) and bool(jnp.all(jnp.isnan(low)))
    if dist.event_dim == 0:
        assert bool(jnp.isnan(bad.cdf(value)))
    if dist.has_icdf:
        assert bool(jnp.all(jnp.isnan(bad.icdf(jnp.full(value.shape, 0.3)))))
    nan = jax.tree_util.tree_map(lambda leaf: leaf * jnp.nan, dist)
    assert bool(jnp.isnan(evaluate(nan, value)[0]))
    assert bool(jnp.isnan(dist.log_prob(value * jnp.nan)))
    with pytest.raises(ValueError, match="outside their domain"):
        type(bad)(*(getattr(bad, field.name) for field in dataclasses.fields(bad)))


def test_debug_flag_raises_for_invalid_parameters(dists):
    """With the debug flag on, invalid traced parameters raise at run time.

    Fails if the flag is ignored, which would let a masked parameter bug
    through the test suite unseen.
    """
    bad = _invalid("normal", dists["normal"])
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside their domain"):
        eqx.filter_jit(lambda record: record.log_prob(0.3))(bad)


def test_truncated_refuses_a_base_without_a_scalar_icdf(dists):
    """TruncatedDistribution refuses an event distribution or a mixture at construction.

    Fails if the refusal moves into a traced method, far from the mistake.
    """
    for name in ("multivariate_normal", "mixture"):
        with pytest.raises(ValueError, match="scalar base_dist"):
            pr.TruncatedDistribution(dists[name], high=1.0)


def test_mixture_has_no_quantile(dists):
    """A mixture declares no inverse CDF and raises from icdf.

    Fails if a consumer could build a unit-cube sampler on a distribution without a
    quantile and only find out inside the sampler.
    """
    assert not dists["mixture"].has_icdf
    with pytest.raises(NotImplementedError, match="has_icdf"):
        dists["mixture"].icdf(0.5)
    with pytest.raises(NotImplementedError, match="event axis"):
        dists["event_mixture"].cdf(jnp.zeros(3))
