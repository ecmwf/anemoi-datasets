# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import xarray as xr
from multiurl import download

from anemoi.datasets.create.sources.xarray import XarrayFieldList

URLS = {
    "https://get.ecmwf.int/repository/test-data/earthkit-data/examples/efas.nc": dict(length=3),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/examples/era5-2m-temperature-dec-1993.nc": dict(length=1),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/examples/test.nc": dict(length=2),
    "https://get.ecmwf.int/repository/test-data/metview/gallery/era5_2000_aug.nc": dict(length=3),
    "https://get.ecmwf.int/repository/test-data/metview/gallery/era5_ozone_1999.nc": dict(length=4),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/fa_ta850.nc": dict(length=37),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/htessel_points.nc": dict(length=1),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/test_single.nc": dict(length=1),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/zgrid_rhgmet_metop_200701_R_2305_0010.nc": dict(
        length=1
    ),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/20210101-C3S-L2_GHG-GHG_PRODUCTS-TANSO2-GOSAT2-SRFP-DAILY-v2.0.0.nc": dict(
        length=1
    ),
    "https://get.ecmwf.int/repository/test-data/earthkit-data/test-data/20220401-C3S-L3S_FIRE-BA-OLCI-AREA_3-fv1.1.nc": dict(
        length=3
    ),
    "https://github.com/ecmwf/magics-test/raw/master/test/efas/tamir.nc": dict(length=1),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/C3S_OZONE-L4-TC-ASSIM_MSR-201608-fv0020.nc": dict(
        length=1
    ),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/avg_data.nc": dict(length=1),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/era5_2000_aug_1.nc": dict(length=3),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/missing.nc": dict(length=20),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/netcdf3_t_z.nc": dict(length=30),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/tos_O1_2001-2002.nc": dict(length=24),
    "https://github.com/ecmwf/magics-test/raw/master/test/gallery/z_500.nc": dict(length=1),
}


def skip_test_netcdf() -> None:
    """Test loading and validating various NetCDF datasets."""
    for url, checks in URLS.items():
        print(url)
        path = os.path.join(os.path.dirname(__file__), os.path.basename(url))
        if not os.path.exists(path):
            download(url, path)

        ds = xr.open_dataset(path)

        fs = XarrayFieldList.from_xarray(ds)

        assert len(fs) == checks["length"], (url, len(fs))

        print(fs[0].datetime())


if __name__ == "__main__":
    skip_test_netcdf()
