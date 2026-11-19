# priorities

<p align="center">
  <a href="https://github.com/CoreySpohn/priorities/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/priorities/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/priorities/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/priorities/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
</p>

Probability distributions as [Equinox](https://docs.kidger.site/equinox/) records for
JAX: a density, its draws and its map to the unit cube, held as array
parameters that jit, vmap and differentiate like any other pytree. The
runtime dependencies are `jax` and `equinox`.

## Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import priorities as pr

# An eccentricity prior: a Beta distribution cut at 0.95.
prior = pr.TruncatedDistribution(pr.Beta(0.867, 3.03), high=0.95)

draws = prior.sample(jax.random.key(0), (1000,))
assert draws.shape == (1000,)
assert float(draws.max()) <= 0.95

# The density, exactly -inf outside the support.
assert jnp.isfinite(prior.log_prob(0.2))
assert prior.log_prob(0.97) == -jnp.inf

# The inverse CDF: uniform in, the distribution out, and back.
value = prior.icdf(0.25)
assert jnp.allclose(prior.cdf(value), 0.25)

# An unconstrained coordinate for a gradient-based sampler. real_log_prob
# is the density of that coordinate, including the volume term.
z = pr.to_real(prior, value)
back, log_volume = pr.from_real(prior, z)
assert jnp.allclose(back, value)
gradient = jax.grad(lambda z: pr.real_log_prob(prior, z))(z)
assert jnp.isfinite(gradient)

# Parameters are leaves: one compiled function serves every value.
score = jax.jit(jax.grad(lambda dist: dist.log_prob(0.2)))(pr.Beta(0.867, 3.03))
assert jnp.isfinite(score.concentration1)
```

## What it holds

| Record | Support | `icdf` |
| --- | --- | --- |
| `Uniform(low, high)` | `[low, high]` | yes |
| `LogUniform(low, high)` | `[low, high]`, `low > 0` | yes |
| `Normal(loc, scale)` | the real line | yes |
| `LogNormal(loc, scale)` | `(0, inf)` | yes |
| `HalfNormal(scale)` | `[0, inf)` | yes |
| `Rayleigh(scale)` | `(0, inf)` | yes |
| `Beta(concentration1, concentration0)` | `(0, 1)` | yes |
| `TruncatedDistribution(base_dist, low, high)` | the cut of a scalar distribution | yes |
| `MultivariateNormal(loc, scale_tril)` | vectors | yes |
| `Mixture(mixing_logits, component_distribution)` | union of the components | no |

Every record implements five methods: `support`, `log_prob`, `sample`,
`cdf` and `icdf`. Three free functions work on any distribution:
`to_real`, `from_real` and `real_log_prob`, the default change of variable to
an unconstrained coordinate, chosen from the static `bounds` of the distribution.

Names follow NumPyro: `sample(key, sample_shape)`, `log_prob`, `cdf`, `icdf(q)`,
`Beta(concentration1, concentration0)`, `MultivariateNormal(loc, scale_tril)`,
`TruncatedDistribution(base_dist, low, high)`. The differences are deliberate:
`support()` is a method returning the bounds as arrays rather than a
constraint object, `bounds` names the bounded sides statically, `has_icdf`
says before tracing whether `icdf` exists, and `Mixture(mixing_logits,
component_distribution)` takes its weights as logits rather than a
categorical distribution.

## Conventions

- **Shapes.** A value has shape `sample + batch + event`, where `batch` is
  the broadcast of the parameter shapes. A distribution is elementwise or has one
  event axis. The components of a `Mixture` are one distribution whose batch has a
  last axis of length `K`.
- **Precision.** The output dtype is the promotion of the argument and the
  parameters. A distribution built from Python numbers follows its argument, so it
  serves float32 and float64 pipelines alike. Nothing is built at import:
  enable x64 before constructing a distribution, not before importing the package.
- **Off the support.** `log_prob` is exactly `-inf` outside the support,
  with a zero gradient, and NaN for a NaN value.
- **Invalid parameters.** A distribution built from concrete parameters outside their
  domain raises `ValueError`. Under a transformation the parameters are
  traced and cannot be inspected, so `log_prob` returns `-inf` and the other
  methods return NaN: a sampler that proposes an invalid parameter rejects
  it. Set `PRIORITIES_DEBUG_CHECKS=1` before import to raise at run time
  instead.
- **Truncation.** Which sides are cut is static: `None` leaves a side open.
  The mass of the cut is computed from the base distribution's `cdf`, so a cut
  far in a tail loses relative precision.
- **Beta shapes.** The Beta density is differentiable in its shapes. Its
  distribution function and quantile are not, and differentiating them in
  `a` or `b` raises instead of returning a wrong number.
- **Mixture.** A scalar mixture has a distribution function (`cdf`) but
  no quantile: `has_icdf` is false and `icdf` raises.

## Adding a distribution

Subclass `AbstractDistribution` when the mathematics is new, and implement the five
methods. A named prior that only fixes the parameters of an existing record
is a function returning that record, not a class:

```python
import priorities as pr


def kipping_eccentricity():
    """Short-period eccentricity distribution of Kipping (2013)."""
    return pr.Beta(0.867, 3.03)


assert isinstance(kipping_eccentricity(), pr.Beta)
```

`priorities.testing.check_distribution(distribution, key)` checks that the draws, the inverse CDF
and the support of a distribution agree with its density without restating any
formula, so it applies unchanged to a new subclass. It needs no reference
code; pin the absolute scale of `log_prob` against one separately.

## Scope

The package stays small on purpose. It holds distributions and the three functions
above. Samplers, inference, parameter trees, bijector objects and
domain-specific priors belong to the packages that use it. A test holds the
source to its line budget and to `jax` and `equinox` as the only runtime
dependencies.

## Installation

```bash
pip install priorities
```
