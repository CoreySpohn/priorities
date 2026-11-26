"""Shared distributions and settings for the priorities tests."""

import gc
import importlib.util
import os

import jax
import jax.numpy as jnp
import pytest
from hypothesis import settings

# Read once when priorities is imported: invalid parameters raise at run time.
os.environ.setdefault("PRIORITIES_DEBUG_CHECKS", "1")

import priorities as pr

# The budget and text-guard plugins ship with hwoutils, which is optional here.
pytest_plugins = []
if importlib.util.find_spec("hwoutils") is not None:
    if importlib.util.find_spec("hwoutils.testbudget") is not None:
        pytest_plugins.append("hwoutils.testbudget.plugin")
    if importlib.util.find_spec("hwoutils.internal_refs") is not None:
        pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

MEAN = (0.1, -2.0, 3.0)
CHOLESKY = ((1.0, 0.0, 0.0), (0.5, 2.0, 0.0), (-0.3, 0.1, 0.7))
WEIGHTS = (0.2, 0.5, 0.3)

# Every record, built on demand after the precision lane has been set. The
# shapes of Beta(0.867, 3.03) are the Kipping (2013) eccentricity fit.
BUILDERS = {
    "uniform": lambda: pr.Uniform(-1.5, 2.5),
    "log_uniform": lambda: pr.LogUniform(0.3, 40.0),
    "normal": lambda: pr.Normal(0.7, 1.3),
    "log_normal": lambda: pr.LogNormal(0.2, 0.6),
    "half_normal": lambda: pr.HalfNormal(1.7),
    "rayleigh": lambda: pr.Rayleigh(0.3),
    "beta": lambda: pr.Beta(0.867, 3.03),
    "truncated_normal": lambda: pr.TruncatedDistribution(
        pr.Normal(0.7, 1.3), -1.0, 2.0
    ),
    "truncated_upper": lambda: pr.TruncatedDistribution(pr.Normal(0.7, 1.3), high=2.0),
    "truncated_beta": lambda: pr.TruncatedDistribution(pr.Beta(0.867, 3.03), high=0.95),
    "mixture": lambda: pr.Mixture(
        jnp.log(jnp.array(WEIGHTS)),
        pr.Normal(jnp.array([-2.0, 0.0, 3.0]), jnp.array([0.5, 1.0, 2.0])),
    ),
    "multivariate_normal": lambda: pr.MultivariateNormal(MEAN, CHOLESKY),
    "event_mixture": lambda: pr.Mixture(
        jnp.log(jnp.array(WEIGHTS[:2])),
        pr.MultivariateNormal(
            jnp.array([MEAN, MEAN[::-1]]),
            jnp.array([CHOLESKY, CHOLESKY]) * jnp.array([1.0, 2.0])[:, None, None],
        ),
    ),
}
SCALAR = [name for name in BUILDERS if "mult" not in name and "event" not in name]
HAS_ICDF = [name for name in SCALAR if name != "mixture"]
# Distributions whose draws do not come from their own quantile function.
DIRECT_DRAWS = ["normal", "log_normal", "half_normal", "beta", "mixture"]


@pytest.fixture(scope="session")
def dists():
    return {name: build() for name, build in BUILDERS.items()}


@pytest.fixture(params=list(BUILDERS))
def dist(request, dists):
    return dists[request.param]


@pytest.fixture(params=SCALAR)
def scalar_dist(request, dists):
    return dists[request.param]


@pytest.fixture(params=HAS_ICDF)
def unit_dist(request, dists):
    return dists[request.param]


@pytest.fixture
def no_debug(monkeypatch):
    """Turn the run-time parameter check off, as in production."""
    monkeypatch.setattr("priorities._distribution.DEBUG_CHECKS", False)


@pytest.fixture(scope="module", autouse=True)
def clear_compile_cache():
    yield
    jax.clear_caches()
    gc.collect()
