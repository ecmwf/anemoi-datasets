# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source


def _date_to_datetime(d: Any) -> Any:
    """Converts a date string or a list/tuple of date strings to datetime objects.

    Parameters
    ----------
    d : Any
        A date string or a list/tuple of date strings.

    Returns
    -------
    Any
        A datetime object or a list/tuple of datetime objects.
    """
    if isinstance(d, (list, tuple)):
        return [_date_to_datetime(x) for x in d]
    return datetime.datetime.fromisoformat(d)


def normalise_time_delta(t: Any) -> datetime.timedelta:
    """Normalizes a time delta string to a datetime.timedelta object.

    Parameters
    ----------
    t : Any
        A time delta string ending with 'h' or a datetime.timedelta object.

    Returns
    -------
    datetime.timedelta
        A normalized datetime.timedelta object.
    """
    if isinstance(t, datetime.timedelta):
        assert t == datetime.timedelta(hours=t.hours), t

    assert t.endswith("h"), t

    t = int(t[:-1])
    t = datetime.timedelta(hours=t)
    return t


def group_by_field(ds: Any) -> Dict[Tuple, List[Any]]:
    """Groups fields by their metadata excluding 'date', 'time', and 'step'.

    Parameters
    ----------
    ds : Any
        A dataset object.

    Returns
    -------
    Dict[Tuple, List[Any]]
        A dictionary where keys are tuples of metadata items and values are lists of fields.
    """
    d = defaultdict(list)
    for field in ds.order_by("valid_datetime"):
        m = field.metadata(namespace="mars")
        for k in ("date", "time", "step"):
            m.pop(k, None)
        keys = tuple(m.items())
        d[keys].append(field)
    return d


@legacy_source(__file__)
def tendencies(dates: List[datetime.datetime], time_increment: Any, **kwargs: Any) -> Any:
    """Computes tendencies for the given dates and time increment.

    Parameters
    ----------
    dates : List[datetime.datetime]
        A list of datetime objects.
    time_increment : Any
        A time increment string ending with 'h' or a datetime.timedelta object.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        A dataset object with computed tendencies.
    """
    print("✅", kwargs)
    time_increment = normalise_time_delta(time_increment)

    shifted_dates = [d - time_increment for d in dates]
    all_dates = sorted(list(set(dates + shifted_dates)))

    # from .mars import execute as mars
    from anemoi.datasets.create.mars import execute as mars

    ds = mars(dates=all_dates, **kwargs)

    dates_in_data = ds.unique_values("valid_datetime", progress_bar=False)["valid_datetime"]
    for d in all_dates:
        assert d.isoformat() in dates_in_data, d

    ds1 = ds.sel(valid_datetime=[d.isoformat() for d in dates])
    ds2 = ds.sel(valid_datetime=[d.isoformat() for d in shifted_dates])

    assert len(ds1) == len(ds2), (len(ds1), len(ds2))

    group1 = group_by_field(ds1)
    group2 = group_by_field(ds2)

    assert group1.keys() == group2.keys(), (group1.keys(), group2.keys())

    # prepare output tmp file so we can read it back
    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    for k in group1:
        assert len(group1[k]) == len(group2[k]), k
        print()
        print("❌", k)

        for field, b_field in zip(group1[k], group2[k]):
            for k in ["param", "level", "number", "grid", "shape"]:
                assert field.metadata(k) == b_field.metadata(k), (
                    k,
                    field.metadata(k),
                    b_field.metadata(k),
                )

            c = field.to_numpy()
            b = b_field.to_numpy()
            assert c.shape == b.shape, (c.shape, b.shape)

            ################
            # Actual computation happens here
            x = c - b
            ################

            assert x.shape == c.shape, c.shape
            print(f"Computing data for {field.metadata('valid_datetime')}={field}-{b_field}")
            out.write(x, template=field)

    out.close()

    from earthkit.data import from_source

    ds = from_source("file", path)
    # save a reference to the tmp file so it is deleted
    # only when the dataset is not used anymore
    ds._tmp = tmp

    return ds


execute = tendencies

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """

    config:
      time_increment: 12h
      database: marser
      class: ea
      # date: computed automatically
      # time: computed automatically
      expver: "0001"
      grid: 20.0/20.0
      levtype: sfc
      param: [2t]
    """
    )["config"]

    dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in tendencies(dates, **config):
        print(f, f.to_numpy().mean())
