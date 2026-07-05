# (C) Copyright 2024-2026 Anemoi contributors.
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

from anemoi.transform import Field
from anemoi.transform import FieldList
from anemoi.transform.fields import metadata_key
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry
from earthkit.data.utils.patterns import Pattern

from anemoi.datasets.create.arguments import ValidDates

from ..source import Source
from . import source_registry

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
class GribSource(Source):

    def __init__(
        self,
        context: Any,
        path: str | list[str],
        flavour: str | dict[str, Any] | None = None,
        grid_definition: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the GRIB source.

        Parameters
        ----------
        context : Any
            The context in which the source is created.
        path : str or list of str
            Path or list of paths to the GRIB files.
        flavour : str or dict of str to Any, optional
            Flavour information, by default None.
        grid_definition : dict of str to Any, optional
            Grid definition configuration to create a Grid object, by default None.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments forwarded to ``.sel()``.
        """
        super().__init__(context)
        self.path = path
        self.flavour = RuleBasedFlavour(flavour) if flavour is not None else None
        self.grid = grid_registry.from_config(grid_definition) if grid_definition is not None else None
        self.args = args
        self.kwargs = kwargs

    def execute_valid_dates(self, dates: ValidDates) -> FieldList:
        """Load data from the GRIB files for the given dates.

        Parameters
        ----------
        dates : ValidDates
            The validity-time argument from the pipeline.

        Returns
        -------
        FieldList
            The loaded dataset.
        """
        given_paths = self.path if isinstance(self.path, list) else [self.path]

        all_fields: list[Any] = []
        dates = [d.isoformat() for d in dates]

        # Build sel kwargs, remapping legacy key names to earthkit 1.0 paths.
        # Keys with eccodes type qualifiers (e.g. "level:d") are passed
        # through as "metadata.key:type" — component paths do not support
        # eccodes qualifiers.  ``levtype`` deliberately targets the raw
        # ``metadata.levtype`` (recipes use MARS values such as "sfc"/"pl",
        # which ``vertical.level_type`` would not match).
        sel_kwargs: dict[str, Any] = {}
        sel_remapping: dict[str, str] = {}
        for k, v in self.kwargs.items():
            if ":" in k:
                sel_kwargs[f"metadata.{k}"] = v
                continue
            target = "metadata.levtype" if k == "levtype" else metadata_key(k, default=f"metadata.{k}")
            sel_remapping[k] = "{" + target + "}"
            sel_kwargs[k] = v
        if dates:
            sel_kwargs["valid_datetime"] = dates
            sel_remapping["valid_datetime"] = "{time.valid_datetime}"

        for path in given_paths:

            # do not substitute if not needed
            if "{" not in path:
                paths = [path]
            else:
                paths = Pattern(path).substitute(*self.args, date=dates, allow_extra=True, **self.kwargs)

            for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
                if name in self.kwargs:
                    raise ValueError(f"MARS interpolation parameter '{name}' not supported")

            for path in _expand(paths):
                self.context.trace("📁", "PATH", path)

                if isinstance(path, str) and (path.startswith("ec:") or path.startswith("ectmp:")):
                    from anemoi.datasets.create.ecfs import get_ecfs_file

                    path = get_ecfs_file(path)

                s = FieldList.from_source("file", path)
                if self.flavour is not None:
                    s = self.flavour.map(s)

                s = s.sel(**sel_kwargs, remapping=sel_remapping)
                all_fields.extend(list(s))

        ds = FieldList.from_fields(all_fields)

        # if kwargs and not context.partial_ok:
        # BACK    check(ds, given_paths, valid_datetime=dates, **kwargs)

        if self.grid is not None:
            ds = FieldList.from_fields([Field.from_latitudes_longitudes(f, *self.grid.latlon()) for f in ds])

        if len(ds) == 0:
            LOG.warning(f"No fields found for {dates} in {given_paths} (kwargs={self.kwargs})")

        return ds
