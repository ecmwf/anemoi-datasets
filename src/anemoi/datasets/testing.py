# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# A collection of functions to support pytest testing

import logging
from typing import Any
from typing import List
from typing import Optional

LOG = logging.getLogger(__name__)


def assert_field_list(
    fs: List[Any],
    size: Optional[int] = None,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    constant: bool = False,
    skip: Optional[Any] = None,
) -> None:
    """Asserts various properties of a list of fields.

    Parameters
    ----------
    fs : List[Any]
        List of fields to be checked.
    size : Optional[int], optional
        Expected size of the list. If None, the list must be non-empty.
    start : Optional[Any], optional
        Expected start metadata value. If None, no check is performed.
    end : Optional[Any], optional
        Expected end metadata value. If None, no check is performed.
    constant : bool, optional
        If True, checks that all fields are constant.
    skip : Optional[Any], optional
        Placeholder for future use.
    """
    import numpy as np

    if size is None:
        assert len(fs) > 0, fs
    else:
        assert len(fs) == size, (len(fs), size)

    first = fs[0]
    last = fs[-1]

    if constant:
        # TODO: add a check for constant fields
        pass
    else:
        assert start is None or first.metadata("valid_datetime") == start, (first.metadata("valid_datetime"), start)
        assert end is None or last.metadata("valid_datetime") == end, (last.metadata("valid_datetime"), end)
        print(first.datetime())

    print(last.metadata())

    first = first
    latitudes, longitudes = first.grid_points()

    assert len(latitudes.shape) == 1, latitudes.shape
    assert len(longitudes.shape) == 1, longitudes.shape

    assert len(latitudes) == len(longitudes), (len(latitudes), len(longitudes))
    data = first.to_numpy(flatten=True)

    assert len(data) == len(latitudes), (len(data), len(latitudes))

    north = np.max(latitudes)
    south = np.min(latitudes)
    east = np.max(longitudes)
    west = np.min(longitudes)

    assert north >= south, (north, south)
    assert east >= west, (east, west)
    assert north <= 90, north
    assert south >= -90, south
    assert east <= 360, east
    assert west >= -180, west
