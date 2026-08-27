# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The residual-statistics JSON file: shared format, writer header and reader.

Warnings
--------

Experimental: the file format, the ``residual_statistics`` option and the
``residual_statistics`` attribute may be removed or renamed in a future
release.

*Residual statistics* are the statistics of the difference ``dsA - dsB`` between
two datasets. They are produced by::

    anemoi-datasets compute <dsA> --statistics-residual <dsB> --output residual.json

and consumed by::

    open_dataset(..., residual_statistics="residual.json").residual_statistics

Both sides use this module so that the format cannot drift apart. The file is a
JSON object; the keys that identify and describe it are:

``kind``
    :data:`RESIDUAL_KIND` for a residual-statistics file. Plain (non-residual)
    statistics written by the same command carry :data:`STATISTICS_KIND`
    instead, so a file of the wrong kind is rejected rather than silently used.
``version``
    Format version, see :data:`VERSION`. A reader refuses a version it does not
    know about.
``datasets``
    The two dataset labels the residual was computed from, in the order they
    were subtracted: the residual is ``datasets[0] - datasets[1]``.
``variables``
    The variable names, indexing the statistics arrays.
``statistics``
    Mapping of each key in :data:`STATISTICS_KEYS` to a list of one value per
    variable, in the order given by ``variables``. ``null`` stands for NaN.

Other keys (``tendency``, ``tendency_statistics``, ``compare``, ...) may be
present; they are ignored by the reader but kept by the writer.
"""

import json
import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

#: ``kind`` marker of a residual-statistics file.
RESIDUAL_KIND = "residual-statistics"

#: ``kind`` marker of a plain (non-residual) statistics file.
STATISTICS_KIND = "statistics"

#: Current format version.
VERSION = 1

#: The statistics the file is expected to carry, one value per variable.
STATISTICS_KEYS = ("mean", "stdev", "minimum", "maximum")


class ResidualStatisticsNotAvailable(AttributeError):
    """Raised when a dataset has no residual statistics attached.

    Subclasses :class:`AttributeError` so that ``getattr(ds,
    "residual_statistics", None)`` and friends behave as expected.
    """


def header(kind: str, datasets: list[str]) -> dict[str, Any]:
    """Build the identifying header of a statistics document.

    Parameters
    ----------
    kind : str
        Either :data:`RESIDUAL_KIND` or :data:`STATISTICS_KIND`.
    datasets : list of str
        The dataset labels the statistics were computed from: for a residual,
        the two datasets in the order they were subtracted (the residual is
        ``datasets[0] - datasets[1]``); otherwise the single dataset.

    Returns
    -------
    dict
        The ``kind``, ``version`` and ``datasets`` keys to merge into the
        document.
    """
    assert kind in (RESIDUAL_KIND, STATISTICS_KIND), kind
    if kind == RESIDUAL_KIND and len(datasets) != 2:
        raise ValueError(f"A residual needs exactly two datasets, got {datasets}")
    return dict(kind=kind, version=VERSION, datasets=list(datasets))


class ResidualStatisticsFile:
    """A validated residual-statistics JSON file.

    Attributes
    ----------
    path : str
        Where the file was read from.
    kind : str
        Always :data:`RESIDUAL_KIND` (a file of another kind is refused at load
        time).
    version : int
        The format version of the file.
    datasets : list of str
        The two dataset labels the residual was computed from.
    variables : list of str
        The variable names indexing the statistics.
    """

    def __init__(self, path: str, document: dict[str, Any]) -> None:
        """Validate ``document`` as a residual-statistics file.

        Parameters
        ----------
        path : str
            The path the document was read from, used in error messages and
            metadata.
        document : dict
            The parsed JSON document.

        Raises
        ------
        ValueError
            If the document is not a valid residual-statistics file.
        """
        self.path = str(path)

        if not isinstance(document, dict):
            raise ValueError(f"{self.path}: expected a JSON object, got {type(document).__name__}")

        self.kind = document.get("kind")
        if self.kind != RESIDUAL_KIND:
            if self.kind == STATISTICS_KIND:
                raise ValueError(
                    f"{self.path}: this file holds plain statistics, not residual statistics. "
                    f"Recompute it with `anemoi-datasets compute <dataset-a> "
                    f"--statistics-residual <dataset-b>`."
                )
            raise ValueError(
                f"{self.path}: not a residual-statistics file (expected kind={RESIDUAL_KIND!r}, got kind={self.kind!r})"
            )

        self.version = document.get("version")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ValueError(f"{self.path}: missing or invalid 'version' ({self.version!r})")
        if self.version > VERSION:
            raise ValueError(
                f"{self.path}: format version {self.version} is newer than the supported "
                f"version {VERSION}; please upgrade anemoi-datasets."
            )

        datasets = document.get("datasets")
        if not isinstance(datasets, (list, tuple)) or len(datasets) != 2:
            raise ValueError(f"{self.path}: 'datasets' must be a list of the two dataset names, got {datasets!r}")
        if not all(isinstance(d, str) and d for d in datasets):
            raise ValueError(f"{self.path}: 'datasets' must be a list of two non-empty names, got {datasets!r}")
        self.datasets = [str(d) for d in datasets]

        variables = document.get("variables")
        if not isinstance(variables, (list, tuple)) or not variables:
            raise ValueError(f"{self.path}: 'variables' must be a non-empty list, got {variables!r}")
        if not all(isinstance(v, str) for v in variables):
            raise ValueError(f"{self.path}: 'variables' must be a list of names, got {variables!r}")
        self.variables = [str(v) for v in variables]

        duplicates = sorted({v for v in self.variables if self.variables.count(v) > 1})
        if duplicates:
            raise ValueError(f"{self.path}: duplicated variables {duplicates}")

        self._index = {name: i for i, name in enumerate(self.variables)}
        self._arrays = self._parse_statistics(document.get("statistics"))

    def _parse_statistics(self, statistics: Any) -> dict[str, NDArray[np.float64]]:
        """Validate and convert the ``statistics`` block to float64 arrays."""
        if not isinstance(statistics, dict):
            raise ValueError(f"{self.path}: 'statistics' must be an object, got {statistics!r}")

        missing = [k for k in STATISTICS_KEYS if k not in statistics]
        if missing:
            raise ValueError(f"{self.path}: 'statistics' is missing {missing}")

        result = {}
        for key in STATISTICS_KEYS:
            values = statistics[key]
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{self.path}: statistics['{key}'] must be a list, got {values!r}")
            if len(values) != len(self.variables):
                raise ValueError(
                    f"{self.path}: statistics['{key}'] has {len(values)} values "
                    f"but there are {len(self.variables)} variables"
                )
            # The writer maps NaN to null so that the document is valid JSON.
            result[key] = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        return result

    @classmethod
    def load(cls, path: "str | os.PathLike[str]") -> "ResidualStatisticsFile":
        """Read and validate a residual-statistics file.

        Parameters
        ----------
        path : str or path-like
            Path to the JSON file.

        Returns
        -------
        ResidualStatisticsFile
            The validated file.
        """
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such residual-statistics file: {path}")
        with open(path) as f:
            try:
                document = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: not a valid JSON file ({e})") from e
        return cls(path, document)

    def select(self, variables: list[str]) -> dict[str, NDArray[np.float64]]:
        """Return the statistics for ``variables``, in that order.

        Parameters
        ----------
        variables : list of str
            The variables to extract, typically a dataset's ``variables``. The
            file may hold more variables than requested, but not fewer.

        Returns
        -------
        dict
            Mapping of :data:`STATISTICS_KEYS` to arrays indexed like
            ``variables``.
        """
        unknown = [v for v in variables if v not in self._index]
        if unknown:
            raise ValueError(
                f"{self.path}: no residual statistics for variable(s) {unknown}; the file has {self.variables}"
            )
        index = np.array([self._index[v] for v in variables], dtype=int)
        return {k: v[index] for k, v in self._arrays.items()}

    def metadata(self) -> dict[str, Any]:
        """Return the provenance of this file, for the dataset metadata."""
        return dict(
            path=self.path,
            kind=self.kind,
            version=self.version,
            datasets=list(self.datasets),
            variables=list(self.variables),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path}, {' - '.join(self.datasets)})"
