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
from typing import TYPE_CHECKING
from typing import Any

import earthkit.data as ekd
from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.sources.patterns import iterate_patterns

from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import xarray as xr


def check(what: str, ds: "xr.Dataset", paths: list[str], **kwargs: Any) -> None:
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
    MultiFieldList
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


def load_many_forecast(
    emoji: str,
    context: Any,
    forecast_dates: Any,
    path: str,
    **kwargs: Any,
) -> ekd.FieldList:
    """Load forecast fields for a list of ``(valid_time, basetime)`` pairs.

    Used by the trajectory layout, where each file holds one forecast run.  The
    file is located by substituting the forecast basetime into the
    ``{base_date}`` placeholder of *path* (distinct from ``{date}``, which is the
    validity time used by the analysis layout), and the requested validity times
    are selected within that file.  Each returned field's ``date``/``time``/
    ``step`` metadata is rewritten to describe the forecast (base date and time,
    step in hours), so the trajectory creator can place it on the
    ``(basetime, step)`` grid.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    forecast_dates : ForecastDates
        Iterable of ``(valid_time, basetime)`` pairs.
    path : str
        Path pattern; the ``{base_date}`` placeholder is filled with the
        forecast basetime, e.g. ``.../{base_date:strftime(%Y%m%dT%H)}Z.nc``.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`load_one` (e.g.
        ``param``, ``options``, ``flavour``, ``patch``).

    Returns
    -------
    ekd.FieldList
        The forecast fields, with forecast-aware ``date``/``time``/``step``.
    """
    from collections import defaultdict

    from anemoi.transform.fields import new_field_with_metadata
    from anemoi.transform.fields import new_fieldlist_from_list
    from anemoi.utils.dates import as_datetime

    # One file per basetime: group the requested validity times accordingly.
    by_basetime: dict[Any, list[Any]] = defaultdict(list)
    for valid_time, basetime in forecast_dates:
        by_basetime[as_datetime(basetime)].append(as_datetime(valid_time))

    fields: list[Any] = []
    for basetime, valid_times in by_basetime.items():
        valid_iso = [v.isoformat() for v in valid_times]
        # The file is keyed by the forecast basetime (``{base_date}``); the
        # validity times are what we select *inside* the file. ``dates`` is
        # left empty so no ``{date}`` (validity-time) substitution is implied.
        for resolved_path, _ in iterate_patterns(path, [], base_date=basetime, **kwargs):
            loaded = load_one(emoji, context, valid_iso, resolved_path, **kwargs)
            for field in loaded:
                valid_time = as_datetime(field.metadata("valid_datetime"))
                step_hours = round((valid_time - basetime).total_seconds() / 3600.0)
                fields.append(
                    new_field_with_metadata(
                        field,
                        date=int(basetime.strftime("%Y%m%d")),
                        time=int(basetime.strftime("%H%M")),
                        step=int(step_hours),
                    )
                )

    return new_fieldlist_from_list(fields)


def load_many_forecast_intervals(
    emoji: str,
    context: Any,
    intervals: Any,
    path: str,
    **kwargs: Any,
) -> ekd.FieldList:
    """Load the source fields covering a set of forecast accumulation intervals.

    Used by ``AccumulateSource`` (trajectory layout) when the inner source is
    NetCDF: each :class:`SignedInterval` describes one increment ``[start, end]``
    anchored on a basetime, and the field stored at validity time ``end`` carries
    that increment. The returned fields are tagged with the GRIB-style
    ``startStep``/``endStep``/``validityDate``/``validityTime`` (plus
    ``date``/``time`` = basetime) that ``FieldToInterval`` reads to recover the
    interval and match it to the accumulators.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    intervals : Iterable[SignedInterval]
        Covering intervals (each with ``base``, ``start``, ``end``).
    path : str
        Path pattern; ``{base_date}`` is filled with the basetime.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`load_one`.

    Returns
    -------
    ekd.FieldList
        The source increment fields, tagged with interval metadata.
    """
    from collections import defaultdict

    from anemoi.transform.fields import new_field_with_metadata
    from anemoi.transform.fields import new_fieldlist_from_list
    from anemoi.utils.dates import as_datetime

    # Deduplicate intervals (the same increment can be requested by several
    # targets) and group them by basetime (one file per run).
    unique: dict[tuple, Any] = {}
    for interval in intervals:
        base = as_datetime(interval.base)
        lo = as_datetime(interval.min)
        hi = as_datetime(interval.max)
        unique[(base, lo, hi)] = (base, lo, hi)

    by_base: dict[Any, list[tuple]] = defaultdict(list)
    for base, lo, hi in unique:
        by_base[base].append((lo, hi))

    fields: list[Any] = []
    for base, windows in by_base.items():
        valid_iso = [hi.isoformat() for _, hi in sorted(set((lo, hi) for lo, hi in windows))]
        for resolved_path, _ in iterate_patterns(path, [], base_date=base, **kwargs):
            loaded = load_one(emoji, context, valid_iso, resolved_path, **kwargs)

            # Index the loaded fields by their validity time (several fields per
            # time when multiple params/levels are selected).
            by_valid: dict[Any, list[Any]] = defaultdict(list)
            for field in loaded:
                by_valid[as_datetime(field.metadata("valid_datetime"))].append(field)

            for lo, hi in windows:
                start_step = round((lo - base).total_seconds() / 3600.0)
                end_step = round((hi - base).total_seconds() / 3600.0)
                for field in by_valid.get(hi, []):
                    fields.append(
                        new_field_with_metadata(
                            field,
                            date=int(base.strftime("%Y%m%d")),
                            time=int(base.strftime("%H%M")),
                            step=int(end_step),
                            startStep=int(start_step),
                            endStep=int(end_step),
                            validityDate=int(hi.strftime("%Y%m%d")),
                            validityTime=int(hi.strftime("%H%M")),
                        )
                    )

    return new_fieldlist_from_list(fields)
