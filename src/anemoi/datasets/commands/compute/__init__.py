# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``anemoi-datasets compute`` command.

Recompute statistics, tendencies or residual statistics for a dataset on the fly,
using a simple single-process chunked loop with no parallelism and no on-disk
caching. The computation is deliberately standalone (it does not reuse the
creation-time statistics code) and always accumulates in ``float64``.

Usage
-----
::

    anemoi-datasets compute <dataset> [key=value ...] \\
        [--statistics] [--statistics-tendencies 6h] \\
        [--statistics-residual <dataset-2> [key=value ...]] \\
        [--chunk-size N] [--compare] [--output FILE.json] \\
        [--checkpoint PATH] [--resume] [--parallel N]

``<dataset>`` is either a name/path followed by ``key=value`` ``open_dataset``
options (e.g. ``start=2020-01-01 end=2020-12-31``), or a single JSON literal that
is a complete ``open_dataset`` config (e.g. ``'{"dataset": "x", "start": ...}'``).
The JSON is passed straight to ``open_dataset``; it must NOT contain compute
options, which are always given as the CLI flags above. ``--statistics-residual``
introduces a second dataset described the same way (name + ``key=value`` or a JSON
config). ``--statistics-tendencies`` takes a single delta. NaNs are ignored
per-variable by default.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Any

import numpy as np

from .. import Command
from .engine import Task
from .engine import run as run_engine
from .statistics import DEFAULT_CHUNK_SIZE
from .statistics import STATISTICS

LOG = logging.getLogger(__name__)


def _coerce(value: str) -> Any:
    """Coerce a ``key=value`` string value to int, float, bool, None or str.

    Dates such as ``2020-01-01`` are left as strings (they are not valid ints).

    Parameters
    ----------
    value : str
        The raw value from the command line.
    """
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _is_json(token: str) -> bool:
    """Return ``True`` if ``token`` looks like a JSON object/array literal."""
    return token.lstrip().startswith(("{", "["))


def _parse_dataset_segment(tokens: list[str]) -> tuple[list[Any], dict[str, Any], str]:
    """Parse a dataset specification into ``open_dataset`` (args, kwargs, label).

    A segment is either a single JSON literal (a complete ``open_dataset`` config,
    passed through untouched) or a ``name key=value ...`` form. The two cannot be
    mixed: when a JSON config is given, all options must live inside it.

    Parameters
    ----------
    tokens : list of str
        The tokens describing one dataset.
    """
    if not tokens:
        raise ValueError("Missing dataset name")

    if _is_json(tokens[0]):
        if len(tokens) > 1:
            raise ValueError(
                f"A JSON dataset config must be a single argument; put options inside the JSON, got extra: {tokens[1:]}"
            )
        config = json.loads(tokens[0])
        return [config], {}, _short(config)

    name = tokens[0]
    kwargs: dict[str, Any] = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            raise ValueError(f"Expected key=value, got '{tok}'")
        key, _, val = tok.partition("=")
        kwargs[key] = _coerce(val)
    return [name], kwargs, name


class _Parsed:
    """Container for the parsed ``compute`` command line.

    The dataset(s) are stored as ``open_dataset`` call descriptions (positional
    ``open_args`` plus ``open_kwargs``) so that both the ``key=value`` form
    (``name`` + kwargs) and the JSON form (a single config dict) share one path.
    """

    def __init__(self) -> None:
        # First dataset.
        self.open_args: list[Any] = []
        self.open_kwargs: dict[str, Any] = {}
        self.label: str = ""
        # Actions.
        self.do_statistics = False
        self.tendency: str | None = None
        self.chunk_size = DEFAULT_CHUNK_SIZE
        # Residual (second) dataset.
        self.has_residual = False
        self.residual_open_args: list[Any] = []
        self.residual_open_kwargs: dict[str, Any] = {}
        self.residual_label: str = ""
        # Behaviour.
        self.allow_nans = True  # NaNs are ignored per-variable by default
        self.compare = False
        self.output: str | None = None
        self.checkpoint: str | None = None
        self.resume = False
        self.parallel = 0
        self.sample_dates: float | None = None

    def finalise(self) -> None:
        """Apply default actions once parsing is complete."""
        if not self.do_statistics and self.tendency is None:
            self.do_statistics = True


def _short(obj: Any) -> str:
    """Return a short, human-readable label for a dataset config or name."""
    if isinstance(obj, dict):
        obj = obj.get("dataset", obj)
    text = obj if isinstance(obj, str) else str(obj)
    return text if len(text) <= 60 else text[:57] + "..."


def _parse(tokens: list[str]) -> _Parsed:
    """Parse the raw ``compute`` argument tokens into a :class:`_Parsed`.

    The first token is the dataset, either a name/path (optionally followed by
    ``key=value`` ``open_dataset`` options) or a single JSON literal that is a
    complete ``open_dataset`` config. Compute options are always CLI flags and are
    never part of the JSON. ``--statistics-residual`` introduces a second dataset
    described the same way; it consumes only that dataset's spec, so compute flags
    may appear before or after it.

    Parameters
    ----------
    tokens : list of str
        The raw remainder tokens following ``compute`` on the command line.
    """
    parsed = _Parsed()

    if not tokens:
        raise ValueError("Missing dataset")

    # The first token is the dataset (JSON config or name); the rest are
    # key=value open_dataset options (name form only) and compute flags.
    json_main = _is_json(tokens[0])
    if json_main:
        parsed.open_args = [json.loads(tokens[0])]
        parsed.label = _short(parsed.open_args[0])
    else:
        parsed.open_args = [tokens[0]]
        parsed.label = tokens[0]

    kwargs: dict[str, Any] = {}
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--statistics":
            parsed.do_statistics = True
            i += 1
        elif tok in ("--statistics-tendencies", "--statistics-tendency"):
            i += 1
            if i >= len(tokens) or tokens[i].startswith("--") or "=" in tokens[i]:
                raise ValueError("--statistics-tendencies requires a delta (e.g. 6h)")
            parsed.tendency = tokens[i]
            i += 1
        elif tok == "--statistics-residual":
            i += 1
            if i >= len(tokens):
                raise ValueError("--statistics-residual requires a dataset")
            if _is_json(tokens[i]):
                seg = [tokens[i]]
                i += 1
            else:
                start = i
                i += 1
                while i < len(tokens) and not tokens[i].startswith("--") and "=" in tokens[i]:
                    i += 1
                seg = tokens[start:i]
            parsed.has_residual = True
            (
                parsed.residual_open_args,
                parsed.residual_open_kwargs,
                parsed.residual_label,
            ) = _parse_dataset_segment(seg)
        elif tok in ("--chunk-size", "--chunk_size"):
            i += 1
            if i >= len(tokens):
                raise ValueError("--chunk-size requires a value")
            parsed.chunk_size = int(tokens[i])
            i += 1
        elif tok == "--compare":
            parsed.compare = True
            i += 1
        elif tok == "--resume":
            parsed.resume = True
            i += 1
        elif tok == "--output":
            i += 1
            if i >= len(tokens):
                raise ValueError("--output requires a path")
            parsed.output = tokens[i]
            i += 1
        elif tok == "--checkpoint":
            i += 1
            if i >= len(tokens):
                raise ValueError("--checkpoint requires a path")
            parsed.checkpoint = tokens[i]
            i += 1
        elif tok == "--parallel":
            i += 1
            if i >= len(tokens):
                raise ValueError("--parallel requires a number of workers")
            parsed.parallel = int(tokens[i])
            i += 1
        elif tok in ("--sample-dates", "--sample_dates"):
            i += 1
            if i >= len(tokens):
                raise ValueError("--sample-dates requires a fraction (e.g. 0.1)")
            parsed.sample_dates = float(tokens[i])
            i += 1
        elif tok.startswith("--"):
            raise ValueError(f"Unknown option '{tok}'")
        elif "=" in tok:
            if json_main:
                raise ValueError(
                    f"key=value options are not allowed when the dataset is a JSON config; "
                    f"put '{tok}' inside the JSON instead"
                )
            key, _, val = tok.partition("=")
            kwargs[key] = _coerce(val)
            i += 1
        else:
            raise ValueError(f"Unexpected token '{tok}' (expected key=value or an option)")

    parsed.open_kwargs = kwargs

    parsed.finalise()
    return parsed


def _print_statistics(title: str, variables: list[str], stats: dict[str, Any]) -> None:
    """Pretty-print a statistics dict as a per-variable table.

    Parameters
    ----------
    title : str
        Heading printed above the table.
    variables : list of str
        Variable names, indexing the statistics arrays.
    stats : dict
        Mapping with at least the keys in :data:`STATISTICS`.
    """
    width = max((len(v) for v in variables), default=8)
    width = max(width, 8)
    print()
    print(title)
    print("-" * len(title))
    header = f"{'variable':<{width}}  " + "  ".join(f"{k:>16}" for k in STATISTICS)
    print(header)
    for i, name in enumerate(variables):
        row = f"{name:<{width}}  " + "  ".join(f"{float(stats[k][i]):>16.8g}" for k in STATISTICS)
        print(row)


def _args_sha(parsed: "_Parsed") -> str:
    """Return a short SHA-1 over the arguments that affect the computation result.

    Excludes presentation/runtime options (output, checkpoint path, parallelism,
    resume) so that the same logical computation maps to the same checkpoint.

    Parameters
    ----------
    parsed : _Parsed
        The parsed command line.
    """
    canonical = {
        "open_args": parsed.open_args,
        "open_kwargs": parsed.open_kwargs,
        "do_statistics": parsed.do_statistics,
        "tendency": parsed.tendency,
        "chunk_size": parsed.chunk_size,
        "allow_nans": parsed.allow_nans,
        "has_residual": parsed.has_residual,
        "residual_open_args": parsed.residual_open_args,
        "residual_open_kwargs": parsed.residual_open_kwargs,
        "sample_dates": parsed.sample_dates,
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _compare_block(title: str, variables: list[str], recomputed: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    """Print and return a per-variable comparison between recomputed and stored stats.

    Parameters
    ----------
    title : str
        Heading for the comparison table.
    variables : list of str
        Variable names.
    recomputed : dict
        The recomputed statistics.
    stored : dict
        The dataset's stored statistics.

    Returns
    -------
    dict
        A JSON-serialisable structure of the comparison.
    """
    width = max((len(v) for v in variables), default=8)
    width = max(width, 8)
    print()
    print(title)
    print("-" * len(title))
    print(f"{'variable':<{width}}  {'stat':>8}  {'recomputed':>16}  {'stored':>16}  {'abs diff':>14}  {'rel diff':>12}")

    block: dict[str, Any] = {}
    for i, name in enumerate(variables):
        block[name] = {}
        for key in STATISTICS:
            r = float(recomputed[key][i])
            s = float(stored[key][i])
            abs_diff = abs(r - s)
            denom = max(abs(s), 1e-30)
            rel_diff = abs_diff / denom
            block[name][key] = {"recomputed": r, "stored": s, "abs_diff": abs_diff, "rel_diff": rel_diff}
            print(f"{name:<{width}}  {key:>8}  {r:>16.8g}  {s:>16.8g}  {abs_diff:>14.6g}  {rel_diff:>12.4g}")
    return block


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to plain Python for JSON output."""
    if isinstance(obj, np.ndarray):
        return [None if (isinstance(x, float) and np.isnan(x)) else x for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


class Compute(Command):
    """Recompute statistics, tendencies or residuals for a dataset."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add command-line arguments to the parser.

        Parameters
        ----------
        command_parser : Any
            The argument parser instance.
        """
        # Everything is captured verbatim and parsed by hand: the grammar (mixed
        # key=value tokens and a nested --statistics-residual dataset with its own options or
        # JSON config) does not map cleanly onto argparse.
        command_parser.add_argument(
            "rest",
            nargs=argparse.REMAINDER,
            help=(
                "<dataset> [key=value ...] [--statistics] [--statistics-tendencies 6h] "
                "[--statistics-residual <dataset-2> [key=value ...]] [--chunk-size N] "
                "[--sample-dates FRACTION] "
                "[--compare] [--output FILE.json] [--checkpoint PATH] [--resume] [--parallel N]. "
                "<dataset> may be a name/path with key=value open_dataset options, or a single "
                "JSON literal that is a complete open_dataset config (compute options stay as flags). "
                "NaNs are ignored by default."
            ),
        )

    def run(self, args: Any) -> None:
        """Execute the compute command.

        Parameters
        ----------
        args : Any
            The command-line arguments.
        """
        parsed = _parse(list(args.rest))

        sha = _args_sha(parsed)
        checkpoint = parsed.checkpoint or os.path.join(os.getcwd(), f"compute-checkpoint-{sha}.pkl")

        task = Task(
            open_args=parsed.open_args,
            open_kwargs=parsed.open_kwargs,
            label=parsed.label,
            do_statistics=parsed.do_statistics,
            tendency=parsed.tendency,
            chunk_size=parsed.chunk_size,
            allow_nans=parsed.allow_nans,
            has_residual=parsed.has_residual,
            residual_open_args=parsed.residual_open_args,
            residual_open_kwargs=parsed.residual_open_kwargs,
            residual_label=parsed.residual_label,
            parallel=parsed.parallel,
            checkpoint_path=checkpoint,
            resume=parsed.resume,
            args_sha=sha,
            sample_dates=parsed.sample_dates,
            live=sys.stdout.isatty(),
        )

        variables, results = run_engine(task)

        # Build a JSON-serialisable document while printing the tables.
        document: dict[str, Any] = {
            "dataset": parsed.label,
            "residual": parsed.residual_label if parsed.has_residual else None,
            "variables": variables,
            "tendency": parsed.tendency,
            "statistics": None,
            "tendency_statistics": None,
            "compare": {},
        }

        stats_title = (
            f"Residual statistics ({parsed.label} - {parsed.residual_label})"
            if parsed.has_residual
            else f"Statistics ({parsed.label})"
        )

        if results["statistics"] is not None:
            _print_statistics(stats_title, variables, results["statistics"])
            document["statistics"] = results["statistics"]

        if results["tendency"] is not None:
            t_title = (
                f"Residual tendency statistics (delta={parsed.tendency})"
                if parsed.has_residual
                else f"Tendency statistics (delta={parsed.tendency})"
            )
            _print_statistics(t_title, variables, results["tendency"])
            document["tendency_statistics"] = results["tendency"]

        if parsed.compare:
            self._compare(parsed, variables, results, document)

        if parsed.output:
            with open(parsed.output, "w") as f:
                json.dump(_jsonable(document), f, indent=2)
            LOG.info("Results written to %s", parsed.output)

    def _compare(self, parsed: "_Parsed", variables: list[str], results: dict[str, Any], document: dict[str, Any]) -> None:
        """Compare recomputed statistics with the dataset's stored statistics.

        Parameters
        ----------
        parsed : _Parsed
            The parsed command line.
        variables : list of str
            Variable names.
        results : dict
            The recomputed results from the engine.
        document : dict
            The output document to augment with the comparison.
        """
        if parsed.has_residual:
            LOG.warning("--compare is not meaningful for residuals (no stored stats); skipping.")
            return

        from anemoi.datasets import open_dataset

        ds = open_dataset(*parsed.open_args, **parsed.open_kwargs)

        if results["statistics"] is not None:
            document["compare"]["statistics"] = _compare_block(
                f"Compare statistics vs stored ({parsed.label})",
                variables,
                results["statistics"],
                ds.statistics,
            )

        if results["tendency"] is not None:
            label = parsed.tendency
            try:
                stored = ds.statistics_tendencies(label)
            except Exception as e:  # noqa: BLE001 - dataset may not store this delta
                LOG.warning("Could not read stored tendencies for delta=%s: %s", label, e)
                return
            document["compare"]["tendency"] = _compare_block(
                f"Compare tendency statistics vs stored (delta={label})",
                variables,
                results["tendency"],
                stored,
            )


command = Compute
