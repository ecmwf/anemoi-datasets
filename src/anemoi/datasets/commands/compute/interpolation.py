# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Grid interpolation for the ``compute`` command.

``--grid`` brings the dataset (and, with ``--minus``, the dataset subtracted from
it) onto a common target grid before anything else happens, so that two datasets
at different resolutions can be differenced, and so that the generated dataset is
on the requested grid.

The interpolation is not implemented here: it is delegated to anemoi-transform,
the same code its ``regrid`` filter uses.

- ``nearest`` (the default) uses :func:`anemoi.transform.spatial.nearest_grid_points`,
  a KD-tree over the source points. It only needs the latitudes and longitudes of
  the two grids, so it works for any pair of grids, named or not.
- Any other method is passed to ``earthkit.regrid.interpolate``, which needs both
  grids to be known to earthkit-regrid and a pre-generated matrix to exist for the
  ``(in_grid, out_grid, method)`` triple. The input grid is taken from the
  dataset's ``resolution``.

The target grid itself is resolved by :func:`anemoi.transform.grids.named.lookup`,
which accepts a named grid (``o96``, ``n320``, ...), a resolution (``0.25``,
``0.25x0.25``) or the path of an ``.npz`` file holding ``latitudes`` and
``longitudes`` arrays. Named grids are downloaded once and cached.
"""

import logging
import os
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

#: Interpolation method used when ``--grid-method`` is not given. It is the only
#: method that works for arbitrary grids.
DEFAULT_GRID_METHOD = "nearest"

#: Cache of the nearest-neighbour index arrays, so that a worker process computes
#: the KD-tree once rather than once per segment.
_NEAREST_CACHE: dict[Any, NDArray[np.int64]] = {}


def _lookup_key(spec: str) -> str | list[float]:
    """Convert a ``--grid`` specification to a key for the anemoi-transform lookup.

    Parameters
    ----------
    spec : str
        A named grid (``o96``), a resolution (``0.25``, ``0.25x0.25``,
        ``0.25/0.25``) or the path of an ``.npz`` grid file.

    Returns
    -------
    str or list of float
        The key to pass to :func:`anemoi.transform.grids.named.lookup`.
    """
    if spec.endswith(".npz"):
        return spec

    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[x/]\s*(-?\d+(?:\.\d+)?)\s*", spec)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    if re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", spec):
        return [float(spec), float(spec)]

    return spec


class TargetGrid:
    """The grid the datasets are interpolated to.

    Parameters
    ----------
    spec : str
        The ``--grid`` specification: a named grid, a resolution, or the path of
        an ``.npz`` file holding ``latitudes`` and ``longitudes``.
    """

    def __init__(self, spec: str) -> None:
        from anemoi.transform.grids.named import lookup

        self.spec = spec
        data = lookup(_lookup_key(spec))

        for key in ("latitudes", "longitudes"):
            if key not in data:
                raise ValueError(f"Grid '{spec}': no '{key}' in the grid definition (found {sorted(data)})")

        self.latitudes = np.asarray(data["latitudes"]).reshape(-1)
        self.longitudes = np.asarray(data["longitudes"]).reshape(-1)

        if self.latitudes.shape != self.longitudes.shape:
            raise ValueError(
                f"Grid '{spec}': latitudes and longitudes have different shapes "
                f"({self.latitudes.shape} vs {self.longitudes.shape})"
            )

        LOG.info("Target grid '%s' has %d points", spec, len(self))

    def __len__(self) -> int:
        """Return the number of points of the grid."""
        return len(self.latitudes)

    @property
    def name(self) -> str:
        """A short name for the grid, used as the resolution of a generated dataset."""
        if self.spec.endswith(".npz"):
            name = os.path.basename(self.spec)[: -len(".npz")]
            return name[len("grid-") :] if name.startswith("grid-") else name
        return self.spec

    def __repr__(self) -> str:
        """Return the string representation of the grid."""
        return f"TargetGrid({self.spec}, {len(self)} points)"


class Interpolator:
    """Interpolates the fields of one dataset onto a target grid.

    Parameters
    ----------
    latitudes : ndarray
        Latitudes of the source grid.
    longitudes : ndarray
        Longitudes of the source grid.
    target : TargetGrid
        The grid to interpolate to.
    method : str, optional
        The interpolation method (default: ``nearest``).
    source_grid : str, optional
        The source grid as known to earthkit-regrid (the dataset's
        ``resolution``). Only used by the non-nearest methods.
    """

    def __init__(
        self,
        latitudes: NDArray[Any],
        longitudes: NDArray[Any],
        target: TargetGrid,
        method: str = DEFAULT_GRID_METHOD,
        source_grid: str | None = None,
    ) -> None:
        self.latitudes = np.asarray(latitudes).reshape(-1)
        self.longitudes = np.asarray(longitudes).reshape(-1)
        self.target = target
        self.method = method
        self.source_grid = source_grid
        self._indices: NDArray[np.int64] | None = None

    @classmethod
    def build(
        cls,
        dataset: Any,
        target: TargetGrid | None,
        method: str = DEFAULT_GRID_METHOD,
    ) -> "Interpolator | None":
        """Return the interpolator for a dataset, or ``None`` when there is nothing to do.

        Parameters
        ----------
        dataset : Dataset
            The opened dataset to interpolate.
        target : TargetGrid or None
            The grid to interpolate to; ``None`` when ``--grid`` was not given.
        method : str, optional
            The interpolation method.

        Returns
        -------
        Interpolator or None
            ``None`` when no grid was requested, or when the dataset is already on
            the target grid.
        """
        if target is None:
            return None

        latitudes = np.asarray(dataset.latitudes).reshape(-1)
        longitudes = np.asarray(dataset.longitudes).reshape(-1)

        if len(latitudes) == len(target) and np.array_equal(latitudes, target.latitudes):
            if np.array_equal(longitudes, target.longitudes):
                LOG.info("Dataset is already on grid '%s'; no interpolation needed", target.spec)
                return None

        try:
            source_grid = dataset.resolution
        except KeyError:
            source_grid = None

        return cls(latitudes, longitudes, target, method=method, source_grid=source_grid)

    @property
    def indices(self) -> NDArray[np.int64]:
        """Indices of the source point nearest to each target point."""
        if self._indices is None:
            key = (
                self.target.spec,
                len(self.target),
                self.latitudes.tobytes(),
                self.longitudes.tobytes(),
            )
            if key not in _NEAREST_CACHE:
                from anemoi.transform.spatial import nearest_grid_points

                LOG.info(
                    "Computing the nearest neighbours of %d target points among %d source points",
                    len(self.target),
                    len(self.latitudes),
                )
                _NEAREST_CACHE[key] = np.asarray(
                    nearest_grid_points(
                        self.latitudes,
                        self.longitudes,
                        self.target.latitudes,
                        self.target.longitudes,
                    )
                )
            self._indices = _NEAREST_CACHE[key]
        return self._indices

    def __call__(self, data: NDArray[Any]) -> NDArray[Any]:
        """Interpolate a block of data onto the target grid.

        Parameters
        ----------
        data : ndarray
            The values, whose last axis is the grid.

        Returns
        -------
        ndarray
            The same array with its last axis on the target grid.
        """
        data = np.asarray(data)

        if data.shape[-1] != len(self.latitudes):
            raise ValueError(
                f"Data has {data.shape[-1]} points on its grid axis, "
                f"but the source grid has {len(self.latitudes)} points"
            )

        if self.method == "nearest":
            return data[..., self.indices]

        return self._earthkit(data)

    def _earthkit(self, data: NDArray[Any]) -> NDArray[Any]:
        """Interpolate field by field with earthkit-regrid."""
        from earthkit.regrid import interpolate

        if self.source_grid is None:
            raise ValueError(
                f"Cannot interpolate with method '{self.method}': the dataset has no 'resolution', so its grid "
                "is unknown to earthkit-regrid. Use the default 'nearest' method instead."
            )

        in_grid = {"grid": self.source_grid}
        out_grid = {"grid": self.target.spec}

        flat = data.reshape(-1, data.shape[-1])
        result = np.empty((flat.shape[0], len(self.target)), dtype=np.float64)
        for i, field in enumerate(flat):
            values = interpolate(field, in_grid=in_grid, out_grid=out_grid, method=self.method)
            values = np.asarray(values).reshape(-1)
            if len(values) != len(self.target):
                raise ValueError(
                    f"Interpolation to '{self.target.spec}' returned {len(values)} points, "
                    f"expected {len(self.target)}"
                )
            result[i] = values

        return result.reshape(data.shape[:-1] + (len(self.target),))
