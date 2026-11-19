"""Probability distributions as Equinox records.

A distribution gives a log density against ``d(value)``, draws, and, where it has one,
a map to and from the unit cube. The free functions ``to_real``, ``from_real``
and ``real_log_prob`` add the default transform to unconstrained coordinates.
"""

try:
    from priorities._version import __version__
except ImportError:  # pragma: no cover - source tree without a generated version file
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("priorities")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from priorities._composite import Mixture, MultivariateNormal, TruncatedDistribution
from priorities._distribution import AbstractDistribution
from priorities._records import (
    Beta,
    HalfNormal,
    LogNormal,
    LogUniform,
    Normal,
    Rayleigh,
    Uniform,
)
from priorities._transform import from_real, real_log_prob, to_real

__all__ = [
    "AbstractDistribution",
    "Beta",
    "HalfNormal",
    "LogNormal",
    "LogUniform",
    "Mixture",
    "MultivariateNormal",
    "Normal",
    "Rayleigh",
    "TruncatedDistribution",
    "Uniform",
    "__version__",
    "from_real",
    "real_log_prob",
    "to_real",
]
