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


def assert_field_list(fs, size=None, start=None, end=None):
    assert size is None or len(fs) == size, (len(fs), size)

    first = fs[0]
    last = fs[-1]

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
