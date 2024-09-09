# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import calendar
import datetime
import logging
from pathlib import PurePath

import numpy as np
import zarr
from anemoi.utils.config import load_config as load_settings

from .dataset import Dataset

LOG = logging.getLogger(__name__)


def load_config():
    return load_settings(defaults={"datasets": {"named": {}, "path": []}})


def add_named_dataset(name, path, **kwargs):
    config = load_config()
    if name["datasets"]["named"]:
        raise ValueError(f"Dataset {name} already exists")

    config["datasets"]["named"][name] = path


def add_dataset_path(path):
    config = load_config()

    if path not in config["datasets"]["path"]:
        config["datasets"]["path"].append(path)


def round_datetime(d, dates, up):
    """Round up (or down) a datetime to the nearest date in a list of dates"""
    if dates is None or len(dates) == 0:
        return d

    for i, date in enumerate(dates):
        if date == d:
            return date
        if date > d:
            if up:
                return date
            if i > 0:
                return dates[i - 1]
            return date
    return dates[-1]


def _as_date(d, dates, last):

    # WARNING,  datetime.datetime is a subclass of datetime.date
    # so we need to check for datetime.datetime first

    if isinstance(d, (np.datetime64, datetime.datetime)):
        d = round_datetime(d, dates, up=not last)
        return np.datetime64(d)

    if isinstance(d, datetime.date):
        d = d.year * 10_000 + d.month * 100 + d.day

    try:
        d = int(d)
    except (ValueError, TypeError):
        pass

    if isinstance(d, int):
        if len(str(d)) == 4:
            year = d
            if last:
                return _as_date(np.datetime64(f"{year:04}-12-31T23:59:59"), dates, last)
            else:
                return _as_date(np.datetime64(f"{year:04}-01-01T00:00:00"), dates, last)

        if len(str(d)) == 6:
            year = d // 100
            month = d % 100
            if last:
                _, last_day = calendar.monthrange(year, month)
                return _as_date(np.datetime64(f"{year:04}-{month:02}-{last_day:02}T23:59:59"), dates, last)
            else:
                return _as_date(np.datetime64(f"{year:04}-{month:02}-01T00:00:00"), dates, last)

        if len(str(d)) == 8:
            year = d // 10000
            month = (d % 10000) // 100
            day = d % 100
            if last:
                return _as_date(np.datetime64(f"{year:04}-{month:02}-{day:02}T23:59:59"), dates, last)
            else:
                return _as_date(np.datetime64(f"{year:04}-{month:02}-{day:02}T00:00:00"), dates, last)

    if isinstance(d, str):

        if "-" in d and ":" in d:
            date, time = d.replace(" ", "T").split("T")
            year, month, day = [int(_) for _ in date.split("-")]
            hour, minute, second = [int(_) for _ in time.split(":")]
            return _as_date(
                np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"),
                dates,
                last,
            )

        if "-" in d:
            assert ":" not in d
            bits = d.split("-")
            if len(bits) == 1:
                return _as_date(int(bits[0]), dates, last)

            if len(bits) == 2:
                return _as_date(int(bits[0]) * 100 + int(bits[1]), dates, last)

            if len(bits) == 3:
                return _as_date(int(bits[0]) * 10000 + int(bits[1]) * 100 + int(bits[2]), dates, last)

        if ":" in d:
            assert len(d) == 5
            hour, minute = d.split(":")
            assert minute == "00"
            assert not last
            first = dates[0].astype(object)
            year = first.year
            month = first.month
            day = first.day

            return _as_date(np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour}:00:00"), dates, last)

    raise NotImplementedError(f"Unsupported date: {d} ({type(d)})")


def as_first_date(d, dates):
    return _as_date(d, dates, last=False)


def as_last_date(d, dates):
    return _as_date(d, dates, last=True)


def _concat_or_join(datasets, kwargs):

    if "adjust" in kwargs:
        raise ValueError("Cannot use 'adjust' without specifying 'concat' or 'join'")
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    # Study the dates
    ranges = [(d.dates[0].astype(object), d.dates[-1].astype(object)) for d in datasets]

    if len(set(ranges)) == 1:
        from .join import Join

        return Join(datasets)._overlay(), kwargs

    from .concat import Concat

    Concat.check_dataset_compatibility(datasets)

    return Concat(datasets), kwargs


def _open(a):
    from .stores import Zarr
    from .stores import zarr_lookup

    if isinstance(a, Dataset):
        return a.mutate()

    if isinstance(a, zarr.hierarchy.Group):
        return Zarr(a).mutate()

    if isinstance(a, str):
        return Zarr(zarr_lookup(a)).mutate()

    if isinstance(a, PurePath):
        return _open(str(a)).mutate()

    if isinstance(a, dict):
        return _open_dataset(**a).mutate()

    if isinstance(a, (list, tuple)):
        return _open_dataset(*a).mutate()

    raise NotImplementedError(f"Unsupported argument: {type(a)}")


def _auto_adjust(datasets, kwargs):

    if "adjust" not in kwargs:
        return datasets, kwargs

    adjust_list = kwargs.pop("adjust")
    if not isinstance(adjust_list, (tuple, list)):
        adjust_list = [adjust_list]

    ALIASES = {
        "all": ["select", "frequency", "start", "end"],
        "dates": ["start", "end", "frequency"],
        "variables": ["select"],
    }

    adjust_set = set()

    for a in adjust_list:
        adjust_set.update(ALIASES.get(a, [a]))

    extra = set(adjust_set) - set(ALIASES["all"])
    if extra:
        raise ValueError(f"Invalid adjust keys: {extra}")

    subset_kwargs = [{} for _ in datasets]

    if "select" in adjust_set:
        assert "select" not in kwargs, "Cannot use 'select' in adjust and kwargs"

        variables = None

        for d in datasets:
            if variables is None:
                variables = set(d.variables)
            else:
                variables &= set(d.variables)

        if len(variables) == 0:
            raise ValueError("No common variables")

        for i, d in enumerate(datasets):
            if set(d.variables) != variables:
                subset_kwargs[i]["select"] = sorted(variables)

    if "start" in adjust_set:
        assert "start" not in kwargs, "Cannot use 'start' in adjust and kwargs"
        start = max(d.dates[0] for d in datasets).astype(object)
        for i, d in enumerate(datasets):
            if start != d.dates[0]:
                subset_kwargs[i]["start"] = start

    if "end" in adjust_set:
        assert "end" not in kwargs, "Cannot use 'end' in adjust and kwargs"
        end = min(d.dates[-1] for d in datasets).astype(object)
        for i, d in enumerate(datasets):
            if end != d.dates[-1]:
                subset_kwargs[i]["end"] = end

    if "frequency" in adjust_set:
        assert "frequency" not in kwargs, "Cannot use 'frequency' in adjust and kwargs"
        frequency = max(d.frequency for d in datasets)
        for i, d in enumerate(datasets):
            if d.frequency != frequency:
                subset_kwargs[i]["frequency"] = frequency

    datasets = [d._subset(**subset_kwargs[i]) for i, d in enumerate(datasets)]

    return datasets, kwargs


def _open_dataset(*args, **kwargs):
    sets = []
    for a in args:
        sets.append(_open(a))

    if "xy" in kwargs:
        from .xy import xy_factory

        assert not sets, sets
        return xy_factory(args, kwargs).mutate()

    if "x" in kwargs and "y" in kwargs:
        from .xy import xy_factory

        assert not sets, sets
        return xy_factory(args, kwargs).mutate()

    if "zip" in kwargs:
        from .xy import zip_factory

        assert not sets, sets
        return zip_factory(args, kwargs).mutate()

    if "chain" in kwargs:
        from .unchecked import chain_factory

        assert not sets, sets
        return chain_factory(args, kwargs).mutate()

    if "join" in kwargs:
        from .join import join_factory

        assert not sets, sets
        return join_factory(args, kwargs).mutate()

    if "concat" in kwargs:
        from .concat import concat_factory

        assert not sets, sets
        return concat_factory(args, kwargs).mutate()

    if "ensemble" in kwargs:
        from .ensemble import ensemble_factory

        assert not sets, sets
        return ensemble_factory(args, kwargs).mutate()

    if "grids" in kwargs:
        from .grids import grids_factory

        assert not sets, sets
        return grids_factory(args, kwargs).mutate()

    if "cutout" in kwargs:
        from .grids import cutout_factory

        assert not sets, sets
        return cutout_factory(args, kwargs).mutate()

    for name in ("datasets", "dataset"):
        if name in kwargs:
            datasets = kwargs.pop(name)
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            for a in datasets:
                sets.append(_open(a))

    assert len(sets) > 0, (args, kwargs)

    if len(sets) > 1:
        dataset, kwargs = _concat_or_join(sets, kwargs)
        return dataset._subset(**kwargs)

    return sets[0]._subset(**kwargs)
