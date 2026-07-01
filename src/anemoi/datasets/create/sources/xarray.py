# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import TYPE_CHECKING
from typing import Any

import earthkit.data as ekd
from earthkit.data.readers.xarray.fieldlist import XArrayFieldList

from anemoi.datasets.create.sources.patterns import iterate_patterns
from anemoi.datasets.create.types import DateList

from ..source import Source
from . import source_registry
from .legacy import LegacySource

# Backwards-compatibility alias
XarrayFieldList = XArrayFieldList

__all__ = ["load_many", "load_one", "XArrayFieldList", "XarrayFieldList"]

LOG = logging.getLogger(__name__)

# Mapping from legacy earthkit 0.x sel keys to earthkit 1.0 component paths.
# Used to translate recipe kwargs into component-path keys for sel().
# Unlike grib.py which passes remapping= to sel(), xarray sources convert
# kwargs directly to component-path keys.  This avoids a strict int/float
# comparison issue in earthkit's remapping-based sel — xarray fields expose
# levels as float64 while recipes typically specify integer values.
_SEL_KEY_MAP = {
    "valid_datetime": "time.valid_datetime",
    "param": "parameter.variable",
    "shortName": "parameter.variable",
    "level": "vertical.level",
    "levelist": "vertical.level",
    "levtype": "metadata.levtype",
}

if TYPE_CHECKING:
    pass


def load_one(
    emoji: str,
    context: Any,
    dates: list[str],
    dataset: Any,
    *,
    options: dict[str, Any] | None = None,
    flavour: str | None = None,
    patch: Any | None = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Loads a single dataset.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    dataset : Union[str, xr.Dataset]
        The dataset to load.
    options : Dict[str, Any], optional
        Additional options for loading the dataset.
    flavour : Optional[str], optional
        Flavour of the dataset.
    patch : Optional[Any], optional
        Patch for the dataset.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The loaded dataset.
    """

    # Loading xarray may be long, so import it here to avoid slowing down module imports
    import xarray as xr

    if options is None:
        options = {}

    context.trace(emoji, dataset, options, kwargs)

    if isinstance(dataset, str) and (dataset.startswith("ec:") or dataset.startswith("ectmp:")):
        from anemoi.datasets.create.ecfs import get_ecfs_file

        dataset = get_ecfs_file(dataset)

    if isinstance(dataset, str) and dataset.endswith(".zarr"):
        # If the dataset is a zarr store, we need to use the zarr engine
        options["engine"] = "zarr"

    if isinstance(dataset, xr.Dataset):
        data = dataset
    else:
        print(f"Opening dataset {dataset} with options {options}")
        data = xr.open_dataset(dataset, **options)

    # Pre-filter the xarray dataset before creating the FieldList.
    # XArrayFieldList.sel() doesn't scale well on large FieldLists
    # (hundreds of thousands of fields), and remapping-based sel has
    # strict int/float comparison issues.  Filtering at the xarray level
    # is fast and avoids both problems.  If a pre-filter can't be applied
    # (e.g. param name doesn't match xarray variable names), the kwargs
    # fall through to FieldList sel.
    _xr_param_keys = {"param", "shortName"}
    _xr_coord_keys = {"level", "levelist"}

    remaining_kwargs: dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in _xr_param_keys:
            xr_vars = v if isinstance(v, list) else [v]
            available = [name for name in xr_vars if name in data.data_vars]
            if available:
                data = data[available]
            else:
                # Variable names don't match xarray — defer to FieldList sel
                remaining_kwargs[k] = v
        elif k in _xr_coord_keys:
            try:
                data = data.sel(level=v)
            except (KeyError, ValueError):
                # Coordinate not in dataset — defer to FieldList sel
                remaining_kwargs[k] = v
        else:
            remaining_kwargs[k] = v

    # Pre-filter time dimension to keep the FieldList small.
    # Datasets like CONUS404 span decades (15k+ time steps × levels =
    # hundreds of thousands of fields); XArrayFieldList.sel() doesn't
    # scale on those sizes.
    if dates and "time" in data.dims:
        try:
            data = data.sel(time=dates)
        except KeyError:
            pass  # requested dates not in dataset — FieldList sel will handle

    fs = XArrayFieldList.from_xarray(data, flavour=flavour, patch=patch)

    # Translate any remaining kwargs to component-path keys for sel().
    sel_kwargs: dict[str, Any] = {}
    for k, v in remaining_kwargs.items():
        if ":" in k:
            sel_kwargs[f"metadata.{k}"] = v
        else:
            sel_kwargs[_SEL_KEY_MAP.get(k, k)] = v

    if len(dates) == 0:
        if sel_kwargs:
            result = fs.sel(**sel_kwargs)
        else:
            result = fs
    else:
        if sel_kwargs:
            result = ekd.concat(*[fs.sel(**{"time.valid_datetime": date, **sel_kwargs}) for date in dates])
        else:
            result = ekd.concat(*[fs.sel(**{"time.valid_datetime": date}) for date in dates])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and dates {dates} and {kwargs}")
        LOG.warning(f"Options: {options}")

        for i, k in enumerate(fs):
            a = ["valid_datetime", k.get("time.valid_datetime", default=None)]
            for n in kwargs.keys():
                a.extend([n, k.get(f"parameter.{n}", default=k.get(n, default=None))])
            LOG.warning(f"{[str(x) for x in a]}")

            if i > 16:
                break

    return result


def load_many(emoji: str, context: Any, dates: list[datetime.datetime], pattern: str, **kwargs: Any) -> ekd.FieldList:
    """Loads multiple datasets.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    pattern : str
        Pattern for loading datasets.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The loaded datasets.
    """
    result = []

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        result.append(load_one(emoji, context, dates, path, **kwargs))

    return ekd.concat(*result) if result else ekd.create_fieldlist()


class XarraySourceBase(Source):
    """An Xarray base data source, intended to be subclassed."""

    emoji = "✖️"  # For tracing

    options: dict[str, Any] | None = None
    flavour: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None

    path_or_url: str | None = None

    def __init__(self, context: Any, path: str = None, url: str = None, *args: Any, **kwargs: Any):
        """Initialise the source.

        Parameters
        ----------
        context : Any
            The context for the data source.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)

        if path is not None and url is not None:
            raise ValueError("Cannot specify both path and url")

        if path is not None:
            self.path_or_url = path
        else:
            self.path_or_url = url

        self.args = args
        self.kwargs = kwargs

    def execute_valid_dates(self, dates: DateList) -> FieldList:
        """Execute the data loading process for the given dates.

        Parameters
        ----------
        dates : DateList
            List of dates for which data needs to be loaded.

        Returns
        -------
        FieldList
            The loaded data fields.
        """

        # For now, just a simple wrapper around load_many
        # TODO: move the implementation here

        return load_many(
            self.emoji,
            self.context,
            dates,
            pattern=self.path_or_url,
            options=self.options,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )


class XarraySource(XarraySourceBase):
    pass


@source_registry.register("xarray")
class LegacyXarraySource(LegacySource):
    """Legacy xarray source registered under the 'xarray' source type."""

    name = "xarray"

    @staticmethod
    def _execute(context: Any, dates: list[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
        """Execute the loading of datasets.

        Parameters
        ----------
        context : Any
            Context object.
        dates : List[str]
            List of dates.
        url : str
            URL pattern for loading datasets.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ekd.FieldList
            The loaded datasets.
        """
        return load_many("🌐", context, dates, url, *args, **kwargs)
