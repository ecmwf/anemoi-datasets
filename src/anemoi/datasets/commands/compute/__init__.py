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
        [--start DATE] [--end DATE] [--frequency FREQ] \\
        [--statistics] [--statistics-tendencies 6h] \\
        [--minus <dataset-2> [key=value ...]] \\
        [--grid o96] [--grid-method nearest] \\
        [--output PATH.zarr] \\
        [--chunk-size N] [--compare] [--output-statistics FILE.npz] [--overwrite] \\
        [--checkpoint PATH] [--resume] [--parallel N]

The statistics are always written to a compressed numpy archive (``.npz``), which
holds them as arrays and everything else -- the header, the variable names, the
provenance -- as a JSON string under ``metadata``. Without ``--output-statistics``
the default is ``<dataset-name>.statistics.npz`` in the current directory. The
command fails if the output file already exists unless ``--overwrite`` is given.
The document carries a ``kind``/``version``/``datasets`` header (see
:mod:`anemoi.datasets.misc.residual_statistics`), so a residual document can be
fed back to ``open_dataset(..., residual_statistics=...)`` in either format -- an
experimental option that may be removed or renamed in a future release.

``--grid`` interpolates the dataset -- and the one given to ``--minus`` -- onto
the requested grid before anything else, so that two datasets at different
resolutions can be subtracted. The interpolation is done by anemoi-transform; see
:mod:`anemoi.datasets.commands.compute.interpolation`.

``--start``, ``--end`` and ``--frequency`` are ``open_dataset`` options applied to
every dataset of the command, so that the dataset and the one given to ``--minus``
do not have to repeat them. Giving one of them both as a flag and as a
``key=value`` (or inside a JSON config) is an error.

``--output`` additionally writes the values read by the loop -- the dataset
as opened, or the residual ``<dataset> - <dataset-2>`` -- to a new Zarr dataset of
the same layout (gridded or trajectories), in the same single pass, always as
``float32``. The recomputed statistics become the statistics
of that dataset, and its metadata is derived from the opened dataset(s); a
``derived_from`` metadata entry records the ``open_dataset`` calls the values came
from. See :mod:`anemoi.datasets.commands.compute.output_dataset`.

``<dataset>`` is either a name/path followed by ``key=value`` ``open_dataset``
options (e.g. ``start=2020-01-01 end=2020-12-31``), or a single JSON literal that
is a complete ``open_dataset`` config (e.g. ``'{"dataset": "x", "start": ...}'``).
The JSON is passed straight to ``open_dataset``; it must NOT contain compute
options, which are always given as the CLI flags above. ``--minus``
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

from anemoi.datasets.misc.residual_statistics import NPZ_SUFFIX
from anemoi.datasets.misc.residual_statistics import RESIDUAL_KIND
from anemoi.datasets.misc.residual_statistics import STATISTICS_KIND
from anemoi.datasets.misc.residual_statistics import check_path
from anemoi.datasets.misc.residual_statistics import header
from anemoi.datasets.misc.residual_statistics import save

from .. import Command
from .engine import Task
from .engine import run as run_engine
from .interpolation import DEFAULT_GRID_METHOD
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

    Returns
    -------
    Any
        The coerced value.
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

    Returns
    -------
    tuple of (list, dict, str)
        The ``open_dataset`` positional args, keyword args and a short label.
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
        self.output_statistics: str | None = None
        # open_dataset options applied to every dataset (both, with --minus).
        self.start: Any = None
        self.end: Any = None
        self.frequency: Any = None
        # Interpolation.
        self.grid: str | None = None
        self.grid_method: str = DEFAULT_GRID_METHOD
        self.output_dataset: str | None = None
        self.overwrite = False
        self.checkpoint: str | None = None
        self.resume = False
        self.parallel = 0
        self.sample_dates: float | None = None

    def finalise(self) -> None:
        """Apply default actions once parsing is complete, and check the combinations."""
        if not self.do_statistics and self.tendency is None:
            self.do_statistics = True

        for key in ("start", "end", "frequency"):
            value = getattr(self, key)
            if value is not None:
                self._apply_to_every_dataset(key, value)

        if self.grid is None and self.grid_method != DEFAULT_GRID_METHOD:
            raise ValueError("--grid-method requires --grid")

        if self.output_statistics is not None:
            check_path(self.output_statistics)

        if self.output_dataset is not None:
            # A generated dataset always carries its statistics, so they are
            # computed even when only tendencies were asked for.
            self.do_statistics = True

            if not self.output_dataset.endswith(".zarr"):
                raise ValueError(f"--output must end with '.zarr', got '{self.output_dataset}'")

            if self.sample_dates is not None:
                raise ValueError(
                    "--output cannot be combined with --sample-dates: every date must be read to be written"
                )

    def _apply_to_every_dataset(self, key: str, value: Any) -> None:
        """Add an ``open_dataset`` option to every dataset of the command.

        ``--start``, ``--end`` and ``--frequency`` are shorthands for options that
        would otherwise have to be repeated on the dataset *and* on the one given
        to ``--minus``, which is the common case: the two are subtracted by
        position, so they have to be on the same dates anyway.

        Parameters
        ----------
        key : str
            The ``open_dataset`` option name.
        value : Any
            Its value.

        Raises
        ------
        ValueError
            If a dataset already sets that option, as ``key=value`` or inside its
            JSON config: the two would silently disagree.
        """
        for label, args, kwargs in (
            (self.label, self.open_args, self.open_kwargs),
            (self.residual_label, self.residual_open_args, self.residual_open_kwargs),
        ):
            if not args:  # no --minus
                continue
            if key in kwargs or (isinstance(args[0], dict) and key in args[0]):
                raise ValueError(
                    f"--{key} conflicts with the '{key}' already given for '{label}'. "
                    f"Set it either as the global --{key} flag or per dataset, not both."
                )
            kwargs[key] = value


def _short(obj: Any) -> str:
    """Return a short, human-readable label for a dataset config or name."""
    if isinstance(obj, dict):
        obj = obj.get("dataset", obj)
    text = obj if isinstance(obj, str) else str(obj)
    return text if len(text) <= 60 else text[:57] + "..."


def _default_statistics_output(parsed: "_Parsed") -> str:
    """Derive the default ``--output-statistics`` path, ``<dataset-name>.statistics.npz``.

    The name is the basename of the (first) dataset with any ``.zarr``/``.zip``
    extension stripped; for a JSON config the short label is used instead.
    """
    name = os.path.basename(parsed.label.rstrip("/")) or parsed.label
    for ext in (".zarr", ".zip"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return f"{name}.statistics{NPZ_SUFFIX}"


def _parse(tokens: list[str]) -> _Parsed:
    """Parse the raw ``compute`` argument tokens into a :class:`_Parsed`.

    The first token is the dataset, either a name/path (optionally followed by
    ``key=value`` ``open_dataset`` options) or a single JSON literal that is a
    complete ``open_dataset`` config. Compute options are always CLI flags and are
    never part of the JSON. ``--minus`` introduces a second dataset
    described the same way; it consumes only that dataset's spec, so compute flags
    may appear before or after it.

    Parameters
    ----------
    tokens : list of str
        The raw remainder tokens following ``compute`` on the command line.

    Returns
    -------
    _Parsed
        The parsed command line.
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
        elif tok == "--minus":
            i += 1
            if i >= len(tokens):
                raise ValueError("--minus requires a dataset")
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
        elif tok in ("--output-statistics", "--output_statistics"):
            i += 1
            if i >= len(tokens):
                raise ValueError("--output-statistics requires a path")
            parsed.output_statistics = tokens[i]
            i += 1
        elif tok == "--start":
            i += 1
            if i >= len(tokens):
                raise ValueError("--start requires a date")
            parsed.start = _coerce(tokens[i])
            i += 1
        elif tok == "--end":
            i += 1
            if i >= len(tokens):
                raise ValueError("--end requires a date")
            parsed.end = _coerce(tokens[i])
            i += 1
        elif tok == "--frequency":
            i += 1
            if i >= len(tokens):
                raise ValueError("--frequency requires a frequency (e.g. 6h)")
            parsed.frequency = _coerce(tokens[i])
            i += 1
        elif tok == "--grid":
            i += 1
            if i >= len(tokens):
                raise ValueError("--grid requires a grid (e.g. o96, 0.25, or a path to a .npz grid file)")
            parsed.grid = tokens[i]
            i += 1
        elif tok in ("--grid-method", "--grid_method"):
            i += 1
            if i >= len(tokens):
                raise ValueError("--grid-method requires a method (e.g. nearest)")
            parsed.grid_method = tokens[i]
            i += 1
        elif tok in ("--output", "--output-dataset", "--output_dataset"):
            if tok != "--output":
                LOG.warning("%s is deprecated, use --output instead.", tok)
            i += 1
            if i >= len(tokens):
                raise ValueError(f"{tok} requires a path")
            parsed.output_dataset = tokens[i]
            i += 1
        elif tok == "--overwrite":
            parsed.overwrite = True
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
            raise ValueError(
                f"Unexpected token '{tok}': expected key=value or an option. "
                "Only the first token is the dataset, and every option takes a single value."
            )

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

    Returns
    -------
    str
        A 16-character hexadecimal digest.
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
        "grid": parsed.grid,
        "grid_method": parsed.grid_method,
        # The generated dataset is part of the result: a resumed run must not
        # skip blocks that were written to a different store.
        "output_dataset": parsed.output_dataset,
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _compare_block(
    title: str, variables: list[str], recomputed: dict[str, Any], stored: dict[str, Any]
) -> dict[str, Any]:
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


#: The ``description`` of the ``compute`` sub-parser: what the command does, how a
#: dataset is spelled, and every option. The options are not declared to argparse
#: (see :meth:`Compute.add_arguments`), so this text is the only place ``-h`` gets
#: them from; keep it in step with :func:`_parse` and with ``docs/cli/compute.rst``.
HELP = """\
Recompute the statistics of a dataset -- or of its tendencies, or of its
difference with another dataset -- on the fly from the dataset as opened, without
rewriting its Zarr store. Optionally write the values read to a new dataset.

The results are always written to a compressed numpy archive (.npz), and printed
as a table.


THE DATASET

  <dataset> (and the dataset after --minus) is given in one of two ways:

    a name or path, optionally followed by key=value tokens that are forwarded
    to open_dataset:

        my-dataset start=2020-01-01 end=2020-12-31 select=2t

    or a single JSON literal that is a complete open_dataset config, which is
    handy for nested ones:

        '{"dataset": "my-dataset", "start": "2020-01-01", "select": ["2t"]}'

  The compute options below are always CLI flags: they never go inside the JSON,
  and key=value tokens cannot be mixed with a JSON config.


WHAT TO COMPUTE (--statistics is used when none of these is given)

  --statistics                mean, minimum, maximum and stdev, in float64.

  --statistics-tendencies DELTA
                              the same, for value(t) - value(t - DELTA), e.g. 6h.
                              DELTA must be a multiple of the dataset frequency.

  --minus <dataset-2> [key=value ...]
                              compute on <dataset> - <dataset-2> instead. The
                              subtraction is by position, so both datasets must
                              end up with the same dates, the same variables in
                              the same order and the same shape. Their grids may
                              differ only if --grid is given: without it only the
                              shapes are checked, and two different grids with the
                              same number of points would be subtracted point by
                              point, which is meaningless.

  --compare                   also print the difference with the statistics stored
                              in the dataset. Not applicable to --minus.


INTERPOLATION

  --grid GRID                 bring the dataset -- and the one given to --minus --
                              onto GRID before anything else, so that datasets at
                              different resolutions can be subtracted. GRID is a
                              named grid (o96, n320, ...), a resolution (0.25,
                              0.25x0.25) or the path of an .npz file holding
                              'latitudes' and 'longitudes' arrays. It is a grid,
                              not a pre-computed interpolation matrix.

  --grid-method METHOD        how to interpolate (default: nearest). 'nearest' is
                              a KD-tree over the source points and works for any
                              pair of grids; any other method (linear, ...) is
                              passed to earthkit-regrid, which needs both grids to
                              be known to it and the dataset to have a resolution.
                              Requires --grid.


OUTPUT

  --output-statistics FILE.npz
                              where to write the statistics. Defaults to
                              <dataset-name>.statistics.npz in the current
                              directory. The archive holds one array per
                              statistic, plus the rest of the document as a JSON
                              string under 'metadata'.

  --overwrite                 replace the output file, and the generated dataset,
                              if they exist. Without it the command stops straight
                              away rather than at the end of the computation.

  --output PATH.zarr          also write the values read -- the dataset as opened,
                              or the residual -- to a new Zarr dataset, in the same
                              pass. Gridded and trajectories datasets are both
                              supported, and the generated one has the layout of
                              the one it came from. The recomputed statistics become
                              its statistics. The data array is float32, as in
                              'anemoi-datasets create'; the statistics stay float64.


RUNNING

  --chunk-size N              number of time steps read at a time (default: 1).

  --parallel N                use N worker processes. The dates are split into
                              segments and merged; the result is identical to the
                              sequential one.

  --sample-dates FRACTION     compute on a random fraction of the dates (0.1 is
                              10%), for a quick estimate. Deterministic. Not
                              compatible with --statistics-tendencies (which needs
                              adjacent dates) nor with --parallel.

  --checkpoint PATH           checkpoint file (default:
                              ./compute-checkpoint-<sha1 of the arguments>.pkl).
                              Written about every minute, removed on success.

  --resume                    restart from the checkpoint. The arguments must be
                              the ones it was created with.

NaNs are ignored, per variable. Missing dates cannot be read: they are skipped,
and no tendency is computed across the gap they leave. A generated dataset keeps
them, as NaN and in its own 'missing_dates'. Open the dataset with
fill_missing_dates to fill and include them instead.
"""

#: The ``epilog`` of the ``compute`` sub-parser: worked command lines, printed
#: after the options.
EXAMPLES = """\
EXAMPLES

  Recompute the statistics of a dataset as opened, over a sub-period:

      anemoi-datasets compute my-dataset start=2020-01-01 end=2020-12-31

  Check the statistics stored in a dataset:

      anemoi-datasets compute my-dataset --compare

  Statistics of the 6-hour tendencies, on 8 processes, to a named file:

      anemoi-datasets compute my-dataset --statistics-tendencies 6h \\
          --parallel 8 --output-statistics tendencies.npz

  A quick estimate from 10% of the dates:

      anemoi-datasets compute my-dataset --sample-dates 0.1

  Write the results as a numpy archive rather than JSON:

      anemoi-datasets compute my-dataset --output-statistics my-dataset.statistics.npz

  Statistics of the difference between two datasets at different resolutions,
  both brought onto o96 first:

      anemoi-datasets compute hres-dataset --minus lres-dataset --grid o96

  ... interpolating properly rather than by nearest neighbour:

      anemoi-datasets compute hres-dataset --minus lres-dataset \\
          --grid o96 --grid-method linear

  Give the dataset as a JSON open_dataset config (compute options stay flags):

      anemoi-datasets compute '{"dataset": "my-dataset", "select": ["2t", "10u"]}' \\
          --statistics --parallel 8

  Materialise a view as a dataset of its own, with the recomputed statistics:

      anemoi-datasets compute my-dataset select=2t --output 2t.zarr \\
          --output-statistics 2t.statistics.npz

  Write the difference between two datasets as a dataset, and resume it if it
  is interrupted:

      anemoi-datasets compute a --minus b --grid o96 --output a-b.zarr
      anemoi-datasets compute a --minus b --grid o96 --output a-b.zarr --resume
"""


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
        # The options are documented by hand in HELP/EXAMPLES rather than declared
        # to argparse, because everything after the command name is captured
        # verbatim and parsed by hand: the grammar (mixed key=value tokens and a
        # nested --minus dataset with its own options or JSON config) does not map
        # cleanly onto argparse. RawDescriptionHelpFormatter keeps the layout of
        # the two texts, which the default formatter would re-wrap.
        command_parser.formatter_class = argparse.RawDescriptionHelpFormatter
        command_parser.usage = "%(prog)s <dataset> [key=value ...] [options]"
        command_parser.description = HELP
        command_parser.epilog = EXAMPLES
        # argparse only carries the tokens; the positional itself is not worth
        # showing, the usage line and HELP describe it.
        command_parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    def run(self, args: Any) -> None:
        """Execute the compute command.

        Parameters
        ----------
        args : Any
            The command-line arguments.
        """
        parsed = _parse(list(args.rest))

        # Resolve the statistics path and fail fast (before any computation) if it
        # already exists and --overwrite was not given.
        output_statistics = parsed.output_statistics or _default_statistics_output(parsed)
        if os.path.exists(output_statistics) and not parsed.overwrite:
            raise ValueError(f"Statistics file already exists: {output_statistics} (use --overwrite to replace it)")

        if parsed.output_dataset and os.path.exists(parsed.output_dataset) and not (parsed.overwrite or parsed.resume):
            raise ValueError(
                f"Output dataset already exists: {parsed.output_dataset} "
                "(use --overwrite to replace it, or --resume to continue writing it)"
            )

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
            grid=parsed.grid,
            grid_method=parsed.grid_method,
            output_dataset=parsed.output_dataset,
            output_overwrite=parsed.overwrite,
        )

        variables, results = run_engine(task)

        # Build a JSON-serialisable document while printing the tables.
        # The `kind`/`version`/`datasets` header identifies the file: a residual
        # document can be fed back to `open_dataset(..., residual_statistics=...)`,
        # which refuses anything that is not marked as a residual.
        document: dict[str, Any] = {
            **header(
                RESIDUAL_KIND if parsed.has_residual else STATISTICS_KIND,
                [parsed.label, parsed.residual_label] if parsed.has_residual else [parsed.label],
            ),
            "dataset": parsed.label,
            "residual": parsed.residual_label if parsed.has_residual else None,
            # Where the values came from and what produced them: the
            # open_dataset call(s), the arithmetic, the anemoi-datasets version
            # and the command line. The same entry a generated dataset carries.
            "derived_from": results["derived_from"],
            "variables": variables,
            "tendency": parsed.tendency,
            "grid": parsed.grid,
            "output_dataset": parsed.output_dataset,
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

        save(output_statistics, document)
        LOG.info("Statistics written to %s", output_statistics)

    def _compare(
        self, parsed: "_Parsed", variables: list[str], results: dict[str, Any], document: dict[str, Any]
    ) -> None:
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
