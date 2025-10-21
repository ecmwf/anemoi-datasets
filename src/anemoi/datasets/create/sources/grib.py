# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import glob
import logging
from typing import Any

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry
from earthkit.data import from_source
from earthkit.data.utils.patterns import Pattern

from . import source_registry
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


def check(ds: Any, paths: list[str], **kwargs: Any) -> None:
    """Check if the dataset matches the expected number of fields.

    Parameters
    ----------
    ds : Any
        The dataset to check.
    paths : list of str
        List of paths to the GRIB files.
    **kwargs : Any
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If the number of fields does not match the expected count.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    # in the case of static data (e.g repeated dates) dates might be empty
    if len(ds) != count and kwargs.get("dates", []) == []:
        LOG.warning(
            f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})"
            f" Received empty dates - assuming this is static data."
        )
        return

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})")


def _expand(paths: list[str]) -> Any:
    """Expand the given paths using glob.

    Parameters
    ----------
    paths : list of str
        List of paths to expand.

    Returns
    -------
    Any
        The expanded paths.
    """
    for path in paths:
        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


@source_registry.register("grib")
class GribSource(LegacySource):

    @staticmethod
    def _execute(
        context: Any,
        dates: list[Any],
        path: str | list[str],
        flavour: str | dict[str, Any] | None = None,
        grid_definition: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ekd.FieldList:
        """Executes the function to load data from GRIB files.

        Parameters
        ----------
        context : Any
            The context in which the function is executed.
        dates : list of Any
            List of dates.
        path : str or list of str
            Path or list of paths to the GRIB files.
        flavour : str or dict of str to Any, optional
            Flavour information, by default None.
        grid_definition : dict of str to Any, optional
            Grid definition configuration to create a Grid object, by default None.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The loaded dataset.
        """
        given_paths = path if isinstance(path, list) else [path]
        if flavour is not None:
            flavour = RuleBasedFlavour(flavour)

        if grid_definition is not None:
            grid = grid_registry.from_config(grid_definition)
        else:
            grid = None

        ds = from_source("empty")
        dates = [d.isoformat() for d in dates]

        for path in given_paths:

            # do not substitute if not needed
            if "{" not in path:
                paths = [path]
            else:
                paths = Pattern(path).substitute(*args, date=dates, allow_extra=True, **kwargs)

            for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
                if name in kwargs:
                    raise ValueError(f"MARS interpolation parameter '{name}' not supported")

            for path in _expand(paths):
                context.trace("📁", "PATH", path)
                s = from_source("file", path)
                if flavour is not None:
                    s = flavour.map(s)
                sel_kwargs = kwargs.copy()
                if dates != []:
                    sel_kwargs["valid_datetime"] = dates
                s = s.sel(**sel_kwargs)
                ds = ds + s

        if kwargs and not context.partial_ok:
            check(ds, given_paths, valid_datetime=dates, **kwargs)

        if grid is not None:
            ds = new_fieldlist_from_list([new_field_from_grid(f, grid) for f in ds])

        if len(ds) == 0:
            LOG.warning(f"No fields found for {dates} in {given_paths} (kwargs={kwargs})")

        return ds
