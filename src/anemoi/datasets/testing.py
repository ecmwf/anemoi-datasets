# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# A collection of functions to support pytest testing

import logging

LOG = logging.getLogger(__name__)


def assert_field_list(fs, size=None, start=None, end=None, constant=False, skip=None):
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
