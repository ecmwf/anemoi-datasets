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

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
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

    def _forbid_mars_interpolation_keys(self) -> None:
        """Raise if a MARS-only interpolation keyword was passed to this source."""
        for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
            if name in self.kwargs:
                raise ValueError(f"MARS interpolation parameter '{name}' not supported")

    @staticmethod
    def _step(valid_time: Any, basetime: Any) -> int | float:
        """The ``step`` path keyword: the lead time in hours.

        Whole hours are returned as an ``int``, so that ``{step:int(%03d)}``
        renders them. A sub-hourly lead time is returned as a ``float``, which
        that same format rejects — truncating it to the hour instead would
        silently resolve to a neighbouring file.  An archive whose files are
        named by a sub-hourly lead time is addressed with ``step_minutes`` or
        ``step_seconds`` instead.
        """
        hours = (valid_time - basetime).total_seconds() / 3600
        return int(hours) if hours.is_integer() else hours

    @staticmethod
    def _step_keywords(valid_time: Any, basetime: Any) -> dict[str, Any]:
        """The lead-time path keywords: ``step`` (hours), ``step_minutes``, ``step_seconds``.

        ``step`` keeps its whole-hour-only contract (see :meth:`_step`);
        the other two always render, so a minute-resolution archive can be
        addressed with e.g. ``{step_minutes:int(%04d)}``.
        """
        offset = valid_time - basetime
        return {
            "step": GribSource._step(valid_time, basetime),
            "step_minutes": int(offset.total_seconds() // 60),
            "step_seconds": int(offset.total_seconds()),
        }

    def _sel_kwargs(self, valid_datetimes: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
        """Build ``.sel()`` kwargs, remapping legacy key names to earthkit 1.0 paths.

        Keys with eccodes type qualifiers (e.g. "level:d") are passed
        through as "metadata.key:type" — component paths do not support
        eccodes qualifiers.  ``levtype`` deliberately targets the raw
        ``metadata.levtype`` (recipes use MARS values such as "sfc"/"pl",
        which ``vertical.level_type`` would not match).
        """
        sel_kwargs: dict[str, Any] = {}
        sel_remapping: dict[str, str] = {}
        for k, v in self.kwargs.items():
            if ":" in k:
                sel_kwargs[f"metadata.{k}"] = v
                continue
            target = "metadata.levtype" if k == "levtype" else metadata_key(k, default=f"metadata.{k}")
            sel_remapping[k] = "{" + target + "}"
            sel_kwargs[k] = v
        if valid_datetimes:
            sel_kwargs["valid_datetime"] = valid_datetimes
            sel_remapping["valid_datetime"] = "{time.valid_datetime}"
        return sel_kwargs, sel_remapping

    def _substitute_paths(self, path: str, **template_kwargs: Any) -> list[str]:
        """Resolve one ``path`` template against the given placeholder values.

        In addition to the recipe's own selectors (``self.kwargs``, e.g.
        ``param``/``level``), a path template may reference ``date`` (the
        validity time), ``base_date``/``step`` (forecast requests), or
        ``start_date``/``end_date``/``middle_date`` (accumulation intervals).

        A selector of the same name as one of those keywords (e.g. an
        explicit ``step:`` in the recipe) wins, so that a recipe can always
        pin the value substituted into its own path; merging the two dicts
        rather than splatting both also avoids a duplicate-keyword TypeError.
        """
        if "{" not in path:
            return [path]
        params = {**template_kwargs, **self.kwargs}
        paths = Pattern(path).substitute(*self.args, allow_extra=True, **params)
        # ``substitute`` returns a bare string unless one of the parameters is
        # a list. The forecast and interval paths substitute scalars, so
        # normalise here — a string would otherwise be iterated character by
        # character downstream.
        return [paths] if isinstance(paths, str) else list(paths)

    def _read_fields(self, paths: list[str], sel_kwargs: dict[str, Any], sel_remapping: dict[str, str]) -> list[Any]:
        """Open each resolved path and select the fields matching ``sel_kwargs``."""
        fields: list[Any] = []
        for path in _expand(paths):
            self.context.trace("📁", "PATH", path)

            if isinstance(path, str) and (path.startswith("ec:") or path.startswith("ectmp:")):
                from anemoi.datasets.create.ecfs import get_ecfs_file

                path = get_ecfs_file(path)

            s = FieldList.from_source("file", path)
            if self.flavour is not None:
                s = self.flavour.map(s)

            s = s.sel(**sel_kwargs, remapping=sel_remapping)
            fields.extend(list(s))
        return fields

    def _finalise(self, all_fields: list[Any], request_desc: Any, given_paths: list[str]) -> FieldList:
        """Wrap the collected fields into a ``FieldList``, regridding and warning as needed."""
        ds = FieldList.from_fields(all_fields)

        # if kwargs and not context.partial_ok:
        # BACK    check(ds, given_paths, valid_datetime=dates, **kwargs)

        if self.grid is not None:
            ds = FieldList.from_fields([Field.from_latitudes_longitudes(f, *self.grid.latlon()) for f in ds])

        if len(ds) == 0:
            LOG.warning(f"No fields found for {request_desc} in {given_paths} (kwargs={self.kwargs})")

        return ds

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
        self._forbid_mars_interpolation_keys()
        given_paths = self.path if isinstance(self.path, list) else [self.path]
        dates_iso = [d.isoformat() for d in dates]
        sel_kwargs, sel_remapping = self._sel_kwargs(dates_iso)

        all_fields: list[Any] = []
        for path in given_paths:
            paths = self._substitute_paths(path, date=dates_iso)
            all_fields.extend(self._read_fields(paths, sel_kwargs, sel_remapping))

        return self._finalise(all_fields, dates_iso, given_paths)

    def execute_forecast_dates(self, dates: ForecastDates) -> FieldList:
        """Load data from the GRIB files for the given (valid_time, basetime) pairs.

        Each item is resolved on its own (rather than batching every
        ``valid_time``/``basetime`` into one templated glob) so that the
        ``base_date`` and ``step`` keywords are substituted per pair — batching
        them as parallel lists would let ``Pattern.substitute`` recombine them
        as an (incorrect) cartesian product across unrelated trajectory rows.

        Parameters
        ----------
        dates : ForecastDates
            The ``(valid_time, basetime)`` pairs from the pipeline.

        Returns
        -------
        FieldList
            The loaded dataset.
        """
        self._forbid_mars_interpolation_keys()
        given_paths = self.path if isinstance(self.path, list) else [self.path]

        all_fields: list[Any] = []
        for valid_time, basetime in dates.items:
            step_kwargs = self._step_keywords(valid_time, basetime)
            sel_kwargs, sel_remapping = self._sel_kwargs([valid_time.isoformat()])
            for path in given_paths:
                paths = self._substitute_paths(path, date=valid_time, base_date=basetime, **step_kwargs)
                all_fields.extend(self._read_fields(paths, sel_kwargs, sel_remapping))

        return self._finalise(all_fields, dates.items, given_paths)

    def execute_intervals(self, dates: Intervals) -> FieldList:
        """Load data from the GRIB files for the given archive intervals.

        Used when this source is the ``source:`` of a base-less
        ``accumulate:`` block. Exposes ``start_date``/``end_date``/
        ``middle_date`` template keywords for each interval, in addition to
        ``date`` (the interval's end, i.e. the archived field's own validity
        time).

        Parameters
        ----------
        dates : Intervals
            The archive-resolved accumulation windows from the pipeline.

        Returns
        -------
        FieldList
            The loaded dataset.
        """
        return self._execute_by_interval(dates.intervals)

    def execute_forecast_intervals(self, dates: ForecastIntervals) -> FieldList:
        """Load data from the GRIB files for the given forecast accumulation windows.

        As :meth:`execute_intervals`, plus ``base_date``/``step`` template
        keywords for intervals that carry a model-run time.

        Parameters
        ----------
        dates : ForecastIntervals
            The forecast accumulation windows from the pipeline.

        Returns
        -------
        FieldList
            The loaded dataset.
        """
        return self._execute_by_interval(dates.intervals)

    def _execute_by_interval(self, intervals: list) -> FieldList:
        """Shared implementation for :meth:`execute_intervals` and :meth:`execute_forecast_intervals`.

        Each archive field is read once. Fields are identified by the
        *normalised* ``(min, max, base)`` triple rather than by the
        ``SignedInterval`` itself: the sign carries how the accumulator will
        combine the field, not which field it is, so a covering that both adds
        and subtracts the same archive interval (``from-zero`` does exactly
        that — ``+a(base→step) − a(base→step−period)``) must still fetch it
        only once. This is the same identity ``Accumulator.compute`` matches
        on; returning a second copy would leave it without a consumer and
        raise "Field not used for any accumulation".
        """
        self._forbid_mars_interpolation_keys()
        given_paths = self.path if isinstance(self.path, list) else [self.path]

        all_fields: list[Any] = []
        seen: set = set()
        for interval in intervals:
            key = (interval.min, interval.max, interval.base)
            if key in seen:
                continue
            seen.add(key)

            start, end = interval.min, interval.max
            middle = start + (end - start) / 2
            template_kwargs: dict[str, Any] = {
                "date": end,
                "start_date": start,
                "end_date": end,
                "middle_date": middle,
            }
            if interval.base is not None:
                template_kwargs["base_date"] = interval.base
                template_kwargs.update(self._step_keywords(end, interval.base))

            sel_kwargs, sel_remapping = self._sel_kwargs([end.isoformat()])
            for path in given_paths:
                paths = self._substitute_paths(path, **template_kwargs)
                all_fields.extend(self._read_fields(paths, sel_kwargs, sel_remapping))

        return self._finalise(all_fields, intervals, given_paths)
