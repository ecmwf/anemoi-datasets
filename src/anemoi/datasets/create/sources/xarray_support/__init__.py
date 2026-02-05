# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any

import earthkit.data as ekd
import xarray as xr
from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.sources.patterns import iterate_patterns

from .. import source_registry
from ..legacy import LegacySource
from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)


def check(what: str, ds: xr.Dataset, paths: list[str], **kwargs: Any) -> None:
    """Checks if the dataset has the expected number of fields.

    Parameters
    ----------
    what : str
        Description of what is being checked.
    ds : xr.Dataset
        The dataset to check.
    paths : List[str]
        List of paths.
    **kwargs : Any
        Additional keyword arguments.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


def load_one(
    emoji: str,
    context: Any,
    dates: list[str],
    dataset: str | xr.Dataset,
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
    MultiFieldList
        The loaded dataset.
    """

    if options is None:
        options = {}

    context.trace(emoji, dataset, options, kwargs)

    if isinstance(dataset, str) and dataset.endswith(".zarr"):
        # If the dataset is a zarr store, we need to use the zarr engine
        options["engine"] = "zarr"

    if isinstance(dataset, xr.Dataset):
        data = dataset
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, flavour=flavour, patch=patch)

    if len(dates) == 0:
        result = fs.sel(**kwargs)
    else:
        result = MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and dates {dates} and {kwargs}")
        LOG.warning(f"Options: {options}")

        for i, k in enumerate(fs):
            a = ["valid_datetime", k.metadata("valid_datetime", default=None)]
            for n in kwargs.keys():
                a.extend([n, k.metadata(n, default=None)])
            LOG.warning(f"{[str(x) for x in a]}")

            if i > 16:
                break

        # LOG.warning(data)

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
    MultiFieldList
        The loaded datasets.
    """
    result = []

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        result.append(load_one(emoji, context, dates, path, **kwargs))

    return MultiFieldList(result)


@source_registry.register("xarray")
class LegacyXarraySource(LegacySource):
    name = "xarray"

    @staticmethod
    def _execute(context: Any, dates: list[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
        """Executes the loading of datasets.

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
