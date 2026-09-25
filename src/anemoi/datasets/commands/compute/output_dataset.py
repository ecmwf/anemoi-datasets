# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Optional dataset output for the ``compute`` command.

The ``compute`` command already reads every value of the dataset *as opened* (or,
with ``--minus``, of the difference between two datasets, both optionally
interpolated onto the ``--grid`` grid). This module lets that single pass also
write those values to a new Zarr dataset, so that a view -- a subset, a
selection, a join, a difference or a regridding -- can be materialised as a
dataset in its own right.

The store is written with :class:`anemoi.datasets.create.dataset.Dataset`, the
same writer the ``create`` command uses, so the layout (``data``, the time
axis, ``latitudes``, ``longitudes``, the statistics arrays and the
``_ARRAY_DIMENSIONS`` attributes) is identical to that of a created dataset. The
metadata is derived from the opened view rather than from a recipe: there is no
recipe, so instead a ``derived_from`` entry records the ``open_dataset`` calls
the data came from, together with the metadata of those datasets.

Both grid-shaped layouts are supported, told apart by the rank of the view:

``gridded``
    ``(time, variable, ensemble, cell)``, with a ``dates`` array.
``trajectories``
    ``(base date, variable, ensemble, step, cell)``, with ``base_dates`` and
    ``steps`` arrays and the two frequencies that go with them.

Everything between the two is shared: the values are read, interpolated and
differenced the same way, and the statistics are indexed by variable, which is
axis 1 of both.
"""

import datetime
import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

from .statistics import STATISTICS

LOG = logging.getLogger(__name__)

#: Version of the ``derived_from`` metadata entry.
DERIVED_FROM_VERSION = 1

#: Chunking of the output ``data`` array, matching the default of
#: ``anemoi-datasets create``: one date and one ensemble member per chunk (and,
#: for trajectories, one step), all variables and all grid points. A date chunk
#: of 1 also means that any two workers writing different time ranges never
#: touch the same chunk.
DATES_PER_CHUNK = 1
ENSEMBLES_PER_CHUNK = 1
STEPS_PER_CHUNK = 1

#: The layouts that can be generated, by the rank of the view they come from.
LAYOUTS = {4: "gridded", 5: "trajectories"}

#: The dimension names of the ``data`` array of each layout.
DIMENSIONS = {
    "gridded": ("time", "variable", "ensemble", "cell"),
    "trajectories": ("time", "variable", "ensemble", "step", "cell"),
}

#: The ``dimensions`` metadata entry of each layout. Unlike :data:`DIMENSIONS`,
#: which names the axes of the array, this names the coordinates of the dataset.
METADATA_DIMENSIONS = {
    "gridded": ["dates", "variables", "ensembles", "values"],
    "trajectories": ["base_dates", "variables", "ensembles", "steps", "values"],
}

#: ``dtype`` of the output ``data`` array. Always ``float32``, as in
#: ``anemoi-datasets create``; the statistics stay ``float64``.
DTYPE = "float32"

#: Attributes copied from the source dataset(s) when they are present and all
#: sources agree. They describe the grid, the naming of the variables or the
#: provenance of the values, and are not otherwise reachable from the ``Dataset``
#: API.
PROPAGATED = ("licence", "attribution", "data_request", "origins", "proj_string", "variable_naming")

#: The subset of :data:`PROPAGATED` copied when the values are a residual: the
#: grid and the variable names are unchanged by a difference, but the lineage of
#: the values is not that of either dataset alone.
PROPAGATED_FOR_RESIDUAL = ("proj_string", "variable_naming")

#: Attributes that describe the source grid, and are therefore dropped when the
#: values have been interpolated onto another one.
GRID_ATTRIBUTES = ("proj_string", "data_request")

#: Marker for an attribute that is not propagated (``None`` is a value that can
#: legitimately be propagated).
MISSING = object()


def _digest(array: NDArray[Any]) -> str:
    """Return a hash of the bytes of ``array``."""
    return hashlib.blake2b(np.ascontiguousarray(array).tobytes(), digest_size=16).hexdigest()


class ConstantsTracker:
    """Tracks which variables are constant in time.

    A variable is constant when every time step holds exactly the same field.
    Within a chunk this is checked by comparing the rows to the chunk's first row
    (NaNs comparing equal to NaNs); across chunks -- and across the segments of a
    parallel run, which are merged rather than compared -- it is checked by
    comparing a hash of the bytes of the first row of each chunk. Only hashes are
    kept, so the state stays small enough to be checkpointed.

    Parameters
    ----------
    variables : list of str
        Variable names, indexing axis 1 of the data.
    """

    def __init__(self, variables: list[str]) -> None:
        self.variables = list(variables)
        n = len(self.variables)
        self.constant = [True] * n
        self.digest: list[str | None] = [None] * n

    def update(self, data: NDArray[Any]) -> None:
        """Feed a chunk of data, shaped ``(time, variable, ...)``."""
        if len(data) == 0:
            return

        for i in range(len(self.variables)):
            if not self.constant[i]:
                continue

            column = np.asarray(data[:, i])
            first = column[0]
            digest = _digest(first)

            if self.digest[i] is None:
                self.digest[i] = digest
            elif digest != self.digest[i]:
                self.constant[i] = False
                continue

            nans = np.isnan(first)
            if not np.all((column == first) | (np.isnan(column) & nans)):
                self.constant[i] = False

    def merge(self, other: "ConstantsTracker") -> "ConstantsTracker":
        """Merge another tracker into a new one (used by the parallel path)."""
        if self.variables != other.variables:
            raise ValueError("Cannot merge constants trackers with different variables")

        result = ConstantsTracker(self.variables)
        for i in range(len(self.variables)):
            if self.digest[i] is None:  # this tracker has seen no data
                result.constant[i] = other.constant[i]
                result.digest[i] = other.digest[i]
            elif other.digest[i] is None:  # the other tracker has seen no data
                result.constant[i] = self.constant[i]
                result.digest[i] = self.digest[i]
            else:
                result.constant[i] = self.constant[i] and other.constant[i] and self.digest[i] == other.digest[i]
                result.digest[i] = self.digest[i]
        return result

    def constant_fields(self) -> list[str]:
        """Return the sorted names of the variables that are constant in time."""
        return sorted(name for i, name in enumerate(self.variables) if self.constant[i] and self.digest[i] is not None)


def _source_attributes(dataset: Any) -> list[dict[str, Any]]:
    """Return the Zarr attributes of every store the opened dataset is built on."""
    from anemoi.datasets.usage.store import ZarrStore

    result: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node.dataset, ZarrStore):
            result.append(dict(node.dataset.store.attrs))
        for kid in node.kids:
            walk(kid)

    try:
        walk(dataset.tree())
    except Exception as e:  # noqa: BLE001 - metadata propagation must never be fatal
        LOG.warning("Could not collect the attributes of the source dataset(s): %s", e)

    return result


def _unanimous(attributes: list[dict[str, Any]], key: str) -> Any:
    """Return the value of ``key`` if all the sources that have it agree, else :data:`MISSING`."""
    values = [a[key] for a in attributes if key in a]
    if not values:
        return MISSING
    first = json.dumps(values[0], sort_keys=True, default=str)
    for value in values[1:]:
        if json.dumps(value, sort_keys=True, default=str) != first:
            LOG.debug("Not propagating '%s': the source datasets disagree", key)
            return MISSING
    return values[0]


class OutputDataset:
    """Writer of the dataset generated by the ``compute`` command.

    The lifecycle is split so that it can be driven from several processes:
    :meth:`create` is called once, in the main process, before any data is read;
    :meth:`open_for_write` is called in every process that writes blocks; and
    :meth:`finalise` is called once, at the end, to add the statistics and the
    remaining metadata.

    Parameters
    ----------
    path : str
        Path of the Zarr store to write.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.dtype = DTYPE
        self._dataset: Any = None
        self._array: Any = None

    # ----------------------------------------------------------------- #
    # Creation
    # ----------------------------------------------------------------- #

    @staticmethod
    def layout(dataset: Any) -> str:
        """Return the layout to generate from ``dataset``, from the rank of its shape."""
        return LAYOUTS[len(dataset.shape)]

    @staticmethod
    def time_axis(dataset: Any) -> NDArray[np.datetime64]:
        """Return the dates along axis 0: ``base_dates`` for trajectories, ``dates`` otherwise."""
        return dataset.base_dates if OutputDataset.layout(dataset) == "trajectories" else dataset.dates

    @staticmethod
    def check_dataset(dataset: Any, label: str) -> None:
        """Check that a dataset generated from this view can be written.

        Missing dates are not a reason to refuse: they cannot be read, so they are
        left as NaN in the generated store and recorded in its ``missing_dates``,
        which makes it a dataset with the same gaps as the one it came from.

        Parameters
        ----------
        dataset : Dataset
            The opened dataset (or the first of the two, for a residual).
        label : str
            The dataset label, for error messages.

        Raises
        ------
        ValueError
            If the view is neither gridded nor trajectories, or if it is empty.
        """
        shape = tuple(dataset.shape)
        if len(shape) not in LAYOUTS:
            raise ValueError(
                f"Cannot generate a dataset from '{label}': expected a gridded dataset "
                f"(time, variable, ensemble, cell) or a trajectories one "
                f"(base date, variable, ensemble, step, cell), got shape {shape}"
            )

        if shape[0] == 0:
            raise ValueError(f"Cannot generate a dataset from '{label}': it has no dates")

    def create(
        self,
        dataset: Any,
        *,
        derived_from: dict[str, Any],
        residual: bool,
        allow_nans: bool,
        overwrite: bool = False,
        grid: Any = None,
        missing: frozenset[int] = frozenset(),
    ) -> None:
        """Create the store, its arrays and its metadata.

        No data is written: the ``data`` array is created full of NaNs and filled
        in later by :meth:`write`. The missing dates are never written, so they
        keep that NaN and are recorded in the store's ``missing_dates``.

        Parameters
        ----------
        dataset : Dataset
            The opened dataset the values come from (the first one, for a residual).
        derived_from : dict
            Description of the ``open_dataset`` call(s) the data comes from, stored
            in the metadata.
        residual : bool
            Whether the values written are a difference between two datasets.
        allow_nans : bool
            The NaN policy of the computation, stored as the dataset's ``allow_nans``.
        overwrite : bool, optional
            Whether to replace an existing store.
        grid : TargetGrid, optional
            The grid the values are interpolated onto, when ``--grid`` was given.
            The generated dataset is then on that grid rather than on the
            dataset's own.
        missing : frozenset of int, optional
            The time indices that cannot be read, and are therefore left as NaN
            and recorded as the generated dataset's own missing dates.
        """
        from anemoi.datasets.create.dataset import Dataset as ZarrWriter

        self.check_dataset(dataset, derived_from["datasets"][0]["label"])

        layout = self.layout(dataset)
        latitudes = np.asarray(dataset.latitudes if grid is None else grid.latitudes)
        longitudes = np.asarray(dataset.longitudes if grid is None else grid.longitudes)

        shape = tuple(int(_) for _ in dataset.shape[:-1]) + (len(latitudes),)
        chunks = self._chunks(layout, shape)

        writer = ZarrWriter(self.path, overwrite=overwrite, create=True)
        writer.update_metadata(
            self._metadata(dataset, derived_from, residual=residual, allow_nans=allow_nans, grid=grid, missing=missing)
        )

        writer.add_array(
            name="data",
            chunks=chunks,
            dtype=self.dtype,
            shape=shape,
            dimensions=DIMENSIONS[layout],
            fill_value=np.nan,
        )

        if layout == "trajectories":
            # Two time axes: the base dates along axis 0, the forecast steps
            # along axis -2.
            writer.add_array(name="base_dates", data=np.array(dataset.base_dates, "<M8[s]"), dimensions=("time",))
            writer.add_array(name="steps", data=np.asarray(dataset.steps), dimensions=("step",))
        else:
            writer.add_array(name="dates", data=np.array(dataset.dates, "<M8[s]"), dimensions=("time",))

        writer.add_array(name="latitudes", data=latitudes, dimensions=("cell",))
        writer.add_array(name="longitudes", data=longitudes, dimensions=("cell",))

        writer.add_provenance(name="provenance_compute")
        writer.touch()

        LOG.info("Created %s dataset %s with shape %s and chunks %s", layout, self.path, shape, chunks)
        self._check_name(dataset, grid)

    @staticmethod
    def _chunks(layout: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the chunking of the ``data`` array of a generated dataset.

        One date and one ensemble member per chunk -- and, for trajectories, one
        forecast step -- with all the variables and all the grid points, which is
        what ``anemoi-datasets create`` defaults to for both layouts.

        Parameters
        ----------
        layout : str
            The layout being generated.
        shape : tuple of int
            The shape of the ``data`` array.

        Returns
        -------
        tuple of int
            The chunk sizes, in array order.
        """
        chunks = [
            min(DATES_PER_CHUNK, shape[0]) or 1,
            shape[1],
            min(ENSEMBLES_PER_CHUNK, shape[2]) or 1,
        ]
        if layout == "trajectories":
            chunks.append(min(STEPS_PER_CHUNK, shape[3]) or 1)
        chunks.append(shape[-1])
        return tuple(chunks)

    @staticmethod
    def _date_range(dataset: Any) -> tuple[str, str]:
        """Return the ``(start_date, end_date)`` envelope of the valid times, as ISO strings.

        For a gridded dataset those are the first and last dates. For a
        trajectories one they are the first base date plus the first step and
        the last base date plus the last step, which is what the store's own
        ``start_date`` / ``end_date`` return.

        Parameters
        ----------
        dataset : Dataset
            The opened dataset.

        Returns
        -------
        tuple of str
            The two ISO-formatted dates.
        """
        if OutputDataset.layout(dataset) == "trajectories":
            return (
                dataset.start_date.astype(object).isoformat(),
                dataset.end_date.astype(object).isoformat(),
            )
        dates = dataset.dates
        return dates[0].astype(object).isoformat(), dates[-1].astype(object).isoformat()

    @staticmethod
    def _resolution(dataset: Any) -> str | None:
        """Return the resolution of the dataset, or ``None`` when it has none."""
        try:
            return dataset.resolution
        except KeyError:
            LOG.warning("The source dataset has no 'resolution'; the generated dataset will not have one either.")
            return None

    def _check_name(self, dataset: Any, grid: Any = None) -> None:
        """Warn if the output name does not follow the naming conventions.

        The trajectories convention carries a second frequency (the forecast
        step one), so the layout decides which frequencies are checked.
        """
        from pathlib import Path

        from anemoi.datasets.create.naming import check_dataset_name

        layout = self.layout(dataset)
        dates = self.time_axis(dataset)
        trajectories = layout == "trajectories"

        for message in check_dataset_name(
            Path(self.path).stem,
            resolution=self._resolution(dataset) if grid is None else grid.name,
            start_date=dates[0].astype(object),
            end_date=dates[-1].astype(object),
            frequency=dataset.base_frequency if trajectories else dataset.frequency,
            step_frequency=dataset.step_frequency if trajectories else None,
            layout=layout,
        ):
            LOG.warning("Dataset name warning: %s", message)

    def _metadata(
        self,
        dataset: Any,
        derived_from: dict[str, Any],
        *,
        residual: bool,
        allow_nans: bool,
        grid: Any = None,
        missing: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        """Build the metadata of the generated dataset from the opened view.

        Parameters
        ----------
        dataset : Dataset
            The opened dataset the values come from.
        derived_from : dict
            Description of the ``open_dataset`` call(s) the data comes from.
        residual : bool
            Whether the values written are a difference between two datasets.
        allow_nans : bool
            The NaN policy of the computation.
        grid : TargetGrid, optional
            The grid the values are interpolated onto, if any.
        missing : frozenset of int, optional
            The time indices that are missing.

        Returns
        -------
        dict
            The metadata to write as the store's attributes.
        """
        import uuid

        from anemoi.datasets.create.creator import VERSION

        layout = self.layout(dataset)
        dates = self.time_axis(dataset)
        label = derived_from["datasets"][0]["label"]

        # The missing dates are read from neither dataset, so they stay at the
        # array's NaN fill value; recording them here makes the generated store a
        # dataset with the same gaps as the one(s) it came from.
        missing_dates = [dates[i].astype(object).isoformat() for i in sorted(missing)]
        if missing_dates:
            LOG.warning(
                "%d date(s) of %s are missing: they are written as NaN and recorded " "in the 'missing_dates' of %s.",
                len(missing_dates),
                label,
                self.path,
            )

        # 'start_date' and 'end_date' are the envelope of the valid times. For
        # trajectories those are the base dates shifted by the first and last
        # forecast step, not the base dates themselves.
        start_date, end_date = self._date_range(dataset)

        metadata: dict[str, Any] = {
            "version": VERSION,
            "uuid": str(uuid.uuid4()),
            "layout": layout,
            "dimensions": list(METADATA_DIMENSIONS[layout]),
            "variables": list(dataset.variables),
            "variables_metadata": dataset.variables_metadata,
            "frequency": frequency_to_string(dataset.base_frequency if layout == "trajectories" else dataset.frequency),
            "start_date": start_date,
            "end_date": end_date,
            "missing_dates": missing_dates,
            "resolution": self._resolution(dataset) if grid is None else grid.name,
            # An interpolated dataset is a list of points, not a structured field.
            "field_shape": list(dataset.field_shape) if grid is None else [len(grid)],
            "dtype": self.dtype,
            "allow_nans": bool(allow_nans),
            # Every readable date is used, so the statistics span the whole dataset.
            "statistics_start_date": start_date,
            "statistics_end_date": end_date,
            "description": (
                (
                    f"Residual {label} - {derived_from['datasets'][1]['label']},"
                    " generated by 'anemoi-datasets compute'"
                    if residual
                    else f"Generated by 'anemoi-datasets compute' from {label}"
                )
                + ("" if grid is None else f", interpolated to {grid.spec}")
            ),
            "derived_from": derived_from,
        }

        if layout == "trajectories":
            # The base-date axis, which 'start_date'/'end_date' do not describe.
            metadata["start_base_date"] = dates[0].astype(object).isoformat()
            metadata["end_base_date"] = dates[-1].astype(object).isoformat()

        attributes = _source_attributes(dataset)
        for key in PROPAGATED_FOR_RESIDUAL if residual else PROPAGATED:
            # Attributes describing the source grid no longer hold once the
            # values have been interpolated onto another one.
            if grid is not None and key in GRID_ATTRIBUTES:
                continue
            value = _unanimous(attributes, key)
            if value is not MISSING:
                metadata[key] = value

        if residual:
            # The values are differences: the units are unchanged, but a property
            # such as 'is_accumulation' no longer describes them.
            LOG.warning(
                "The variables metadata of %s is copied from %s: it describes the values of that dataset, "
                "not the differences stored here.",
                self.path,
                label,
            )

        return metadata

    # ----------------------------------------------------------------- #
    # Writing
    # ----------------------------------------------------------------- #

    def open_for_write(self) -> "OutputDataset":
        """Open the existing store for writing, and return ``self``."""
        from anemoi.datasets.create.dataset import Dataset as ZarrWriter

        self._dataset = ZarrWriter(self.path, update=True)
        self._array = self._dataset.data
        return self

    def write(self, block: slice, data: NDArray[Any]) -> None:
        """Write one block of data.

        Parameters
        ----------
        block : slice
            The time range the data belongs to.
        data : ndarray
            The values, shaped ``(time, variable, ensemble, cell)``.
        """
        assert isinstance(block, slice), f"Expected a slice, got {block!r}"
        self._array[block] = np.asarray(data).astype(self.dtype, copy=False)

    # ----------------------------------------------------------------- #
    # Finalisation
    # ----------------------------------------------------------------- #

    def finalise(
        self,
        variables: list[str],
        results: dict[str, Any],
        tendency: str | None = None,
    ) -> None:
        """Add the statistics and the final metadata to the store.

        Parameters
        ----------
        variables : list of str
            Variable names, in the order of the statistics arrays.
        results : dict
            The engine results: ``statistics``, ``tendency`` and ``constants``.
        tendency : str, optional
            The tendency delta, used to name the tendency statistics arrays.
        """
        from anemoi.datasets.create.dataset import Dataset as ZarrWriter

        writer = ZarrWriter(self.path, update=True)

        statistics = results.get("statistics")
        if statistics is None:
            raise ValueError("Cannot generate a dataset without statistics")

        for key in STATISTICS:
            writer.add_array(
                name=key,
                data=np.asarray(statistics[key], dtype=np.float64),
                dimensions=("variable",),
                overwrite=True,
            )

        if results.get("tendency") is not None and tendency is not None:
            from anemoi.utils.dates import frequency_to_timedelta

            delta = frequency_to_string(frequency_to_timedelta(tendency))
            for key in STATISTICS:
                writer.add_array(
                    name=f"statistics_tendencies_{delta}_{key}",
                    data=np.asarray(results["tendency"][key], dtype=np.float64),
                    dimensions=("variable",),
                    overwrite=True,
                )

        constants = results.get("constants")
        metadata: dict[str, Any] = {
            "chunks": writer.data.chunks,
            "shape": writer.data.shape,
        }

        if constants is not None:
            constant_fields = constants.constant_fields()
            metadata["constant_fields"] = constant_fields

            # Mirror what 'create' does: flag the constant variables in their metadata.
            variables_metadata = writer.get_metadata("variables_metadata", {}) or {}
            variables_metadata = json.loads(json.dumps(variables_metadata))
            for name in constant_fields:
                if name in variables_metadata:
                    variables_metadata[name]["constant_in_time"] = True
            metadata["variables_metadata"] = variables_metadata

        writer.update_metadata(metadata)
        writer.touch()
        writer.remove_lock_file()

        LOG.info("Dataset written to %s", self.path)


def _full_label(label: str, open_args: list[Any]) -> str:
    """Return the dataset name behind an ``open_dataset`` spec, untruncated.

    The labels used for the printed tables are shortened to 60 characters, which
    is not what the metadata of a generated dataset should carry.

    Parameters
    ----------
    label : str
        The shortened label, used as a fallback.
    open_args : list
        The positional arguments of the ``open_dataset`` call.

    Returns
    -------
    str
        The dataset name, or ``label`` when the spec has no single name.
    """
    if len(open_args) == 1:
        first = open_args[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and isinstance(first.get("dataset"), str):
            return first["dataset"]
    return label


def derived_from(
    label: str,
    open_args: list[Any],
    open_kwargs: dict[str, Any],
    dataset: Any,
    *,
    residual_label: str | None = None,
    residual_open_args: list[Any] | None = None,
    residual_open_kwargs: dict[str, Any] | None = None,
    residual_dataset: Any = None,
    tendency: str | None = None,
    chunk_size: int | None = None,
    allow_nans: bool = True,
    grid: str | None = None,
    grid_method: str | None = None,
) -> dict[str, Any]:
    """Describe the ``open_dataset`` call(s) a generated dataset comes from.

    The generated dataset has no recipe, so this is what records how it was made:
    the arguments of each ``open_dataset`` call, the metadata of the datasets they
    returned, and the arithmetic applied to their values.

    Parameters
    ----------
    label : str
        Label of the first dataset.
    open_args : list
        Positional arguments of its ``open_dataset`` call.
    open_kwargs : dict
        Keyword arguments of its ``open_dataset`` call.
    dataset : Dataset
        The dataset itself, whose metadata is recorded.
    residual_label : str, optional
        Label of the subtracted dataset, if any.
    residual_open_args : list, optional
        Positional arguments of its ``open_dataset`` call.
    residual_open_kwargs : dict, optional
        Keyword arguments of its ``open_dataset`` call.
    residual_dataset : Dataset, optional
        The subtracted dataset itself.
    tendency : str, optional
        The tendency delta the statistics were computed for.
    chunk_size : int, optional
        Number of time steps read per chunk.
    allow_nans : bool, optional
        Whether NaNs were ignored per-variable.
    grid : str, optional
        The grid both datasets were interpolated to, if any.
    grid_method : str, optional
        The interpolation method used to reach that grid.

    Returns
    -------
    dict
        The ``derived_from`` metadata entry.
    """
    import anemoi.datasets

    def _metadata(ds: Any) -> dict[str, Any] | None:
        try:
            return ds.metadata()
        except Exception as e:  # noqa: BLE001 - provenance must never be fatal
            LOG.warning("Could not collect the metadata of the source dataset: %s", e)
            return None

    datasets = [
        {
            "label": _full_label(label, open_args),
            "open_args": open_args,
            "open_kwargs": open_kwargs,
            "metadata": _metadata(dataset),
        }
    ]

    if residual_dataset is not None:
        datasets.append(
            {
                "label": _full_label(residual_label, residual_open_args or []),
                "open_args": residual_open_args,
                "open_kwargs": residual_open_kwargs,
                "metadata": _metadata(residual_dataset),
            }
        )

    return {
        "version": DERIVED_FROM_VERSION,
        "command": "anemoi-datasets compute",
        "anemoi_datasets_version": anemoi.datasets.__version__,
        "created": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(),
        # State the arithmetic rather than naming it: the values are the first
        # dataset, or the first minus the second.
        "arithmetic": "datasets[0] - datasets[1]" if residual_dataset is not None else "datasets[0]",
        "datasets": datasets,
        "computation": {
            "tendency": tendency,
            "chunk_size": chunk_size,
            "allow_nans": bool(allow_nans),
            "grid": grid,
            "grid_method": grid_method,
            "command_line": _command_line(),
        },
    }


def _command_line() -> str:
    """Return the command line that produced the dataset."""
    import sys

    return " ".join([os.path.basename(sys.argv[0])] + list(sys.argv[1:]))
