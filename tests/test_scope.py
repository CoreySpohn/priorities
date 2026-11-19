"""The scope and budget rules of the package, enforced mechanically."""

import ast
import dataclasses
import inspect
import io
import sys
import tokenize
import tomllib
from pathlib import Path

import jax

import priorities as pr

ROOT = Path(__file__).parent.parent
SOURCES = sorted(
    path
    for path in (ROOT / "src" / "priorities").glob("*.py")
    if path.name != "_version.py"
)
# Lines of code: physical lines holding a token other than a comment or a
# docstring. Crossing the budget calls for a design review, not a larger number.
LINE_BUDGET = 600
RECORDS = {
    "Uniform",
    "LogUniform",
    "Normal",
    "LogNormal",
    "HalfNormal",
    "Beta",
    "Rayleigh",
    "TruncatedDistribution",
    "MultivariateNormal",
    "Mixture",
}
FUNCTIONS = {"to_real", "from_real", "real_log_prob"}


def _code_lines(path):
    source = path.read_text()
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.update(range(value.lineno, value.end_lineno + 1))
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT}
    skip |= {tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in skip:
            lines.update(range(token.start[0], token.end[0] + 1))
    return len(lines - docstrings)


def test_runtime_dependencies_are_jax_and_equinox():
    """The package declares and imports jax and equinox, and nothing else.

    Fails if a third runtime dependency is declared or imported.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {
        spec.split(">")[0].split("=")[0].strip() for spec in project["dependencies"]
    }
    assert declared == {"jax", "equinox"}
    imported = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"priorities"}
    assert third_party == {"jax", "equinox"}


def test_source_stays_inside_the_line_budget():
    """The source holds at most 600 lines of code.

    Fails if the package grows past its budget.
    """
    assert len(SOURCES) >= 5
    assert sum(_code_lines(path) for path in SOURCES) <= LINE_BUDGET


def test_public_surface_is_records_and_three_functions():
    """The exports are the base, ten records and three free functions.

    Fails if a record, a transform object or a function over per-distribution
    mathematics is added without a decision.
    """
    public = set(pr.__all__) - {"__version__", "AbstractDistribution"}
    assert public == RECORDS | FUNCTIONS
    for name in RECORDS:
        assert issubclass(getattr(pr, name), pr.AbstractDistribution)
    for name in FUNCTIONS:
        assert inspect.isfunction(getattr(pr, name))


def test_records_hold_arrays_and_dists_only(dist):
    """Every field of a record is an array, a distribution or an unset cut.

    Fails if a record gains a callable field, a transform object or a Python
    scalar parameter.
    """
    fields = dataclasses.fields(dist)
    assert fields
    for field in fields:
        item = getattr(dist, field.name)
        assert item is None or isinstance(item, jax.Array | pr.AbstractDistribution)
        assert not callable(item)
