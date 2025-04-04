# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import xarray as xr
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_slow_tests

from anemoi.datasets.create.sources.xarray import XarrayFieldList
from anemoi.datasets.testing import assert_field_list


@skip_if_offline
@skip_slow_tests
def test_opendap() -> None:
    """Test loading and validating the opendap dataset."""
    ds = xr.open_dataset(
        "https://thredds.met.no/thredds/dodsC/meps25epsarchive/2023/01/01/meps_det_2_5km_20230101T00Z.nc",
    )

    fs = XarrayFieldList.from_xarray(ds)
    assert_field_list(fs, 79529, "2023-01-01T00:00:00", "2023-01-03T18:00:00")


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
