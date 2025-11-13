# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages

from anemoi.datasets import open_dataset

from .utils.create import create_dataset


@skip_if_offline
def test_grib(get_test_data: callable) -> None:
    """Test the creation of a dataset from GRIB files.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    """
    data1 = get_test_data("anemoi-datasets/create/grib-20100101.grib")
    data2 = get_test_data("anemoi-datasets/create/grib-20100102.grib")
    assert os.path.dirname(data1) == os.path.dirname(data2)

    path = os.path.dirname(data1)

    config = {
        "dates": {
            "start": "2010-01-01T00:00:00",
            "end": "2010-01-02T18:00:00",
            "frequency": "6h",
        },
        "input": {
            "grib": {
                "path": os.path.join(path, "grib-{date:strftime(%Y%m%d)}.grib"),
            },
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (8, 12, 1, 162)


@skip_if_offline
def test_grib_gridfile(get_test_data) -> None:
    """Test the creation of a dataset from GRIB files with an unstructured grid.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    This GRIB data is defined on an unstructured grid and therefore requires
    specifying a grid file.
    """
    data1 = get_test_data("anemoi-datasets/create/grib-iconch1-20250101.grib")
    data2 = get_test_data("anemoi-datasets/create/grib-iconch1-20250102.grib")
    gridfile = get_test_data("anemoi-datasets/create/icon_grid_0001_R19B08_mch.nc")
    assert os.path.dirname(data1) == os.path.dirname(data2)

    path = os.path.dirname(data1)

    config = {
        "dates": {
            "start": "2025-01-01T00:00:00",
            "end": "2025-01-02T18:00:00",
            "frequency": "6h",
        },
        "input": {
            "grib": {
                "path": os.path.join(path, "grib-iconch1-{date:strftime(%Y%m%d)}.grib"),
                "grid_definition": {"icon": {"path": gridfile}},
                "flavour": [[{"levtype": "sfc"}, {"levelist": None}]],
            },
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (8, 1, 1, 1147980)
    assert ds.variables == ["2t"]


@skip_if_offline
@pytest.mark.parametrize(
    "refinement_level_c,shape",
    (
        (2, (2, 13, 1, 2880)),
        (7, (2, 13, 1, 2949120)),
    ),
)
def test_grib_gridfile_with_refinement_level(
    refinement_level_c: str, shape: tuple[int, int, int, int, int], get_test_data: callable
) -> None:
    """Test the creation of a dataset from GRIB files with an unstructured grid.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    This GRIB data is defined on an unstructured grid and therefore requires
    specifying a grid file. The `refinement_level_c` selection key and
    strftimedelta are used.
    """

    p = "anemoi-datasets/create/test_grib_gridfile_with_refinement_level/"
    data1 = get_test_data(p + "2023010103+fc_R03B07_rea_ml.2023010100")
    data2 = get_test_data(p + "2023010106+fc_R03B07_rea_ml.2023010103")
    gridfile = get_test_data("dwd/2024-12-11_00/icon_grid_0026_R03B07_subsetAICON.nc")
    assert os.path.dirname(data1) == os.path.dirname(data2)

    path = os.path.dirname(data1)

    param = ["pres", "t", "u", "v", "q"]
    level = [101, 119]
    forcings = ["cos_latitude", "sin_latitude", "cos_julian_day"]
    assert len(param) * len(level) + len(forcings) == shape[1]

    grib = {
        "path": os.path.join(path, "{date:strftimedelta(+3h;%Y%m%d%H)}+fc_R03B07_rea_ml.{date:strftime(%Y%m%d%H)}"),
        "grid_definition": {"icon": {"path": gridfile}},
        "param": param,
        "level": level,
    }
    refinement_filter = {"icon_refinement_level": {"grid": gridfile, "refinement_level_c": refinement_level_c}}

    config = {
        "dates": {
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T03:00:00",
            "frequency": "3h",
        },
        "input": {
            "pipe": [
                {
                    "join": [
                        {"grib": grib},
                        {"forcings": {"param": forcings, "template": "${input.pipe.0.join.0.grib}"}},
                    ]
                },
                refinement_filter,
            ]
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == shape
    assert np.all(ds.data[ds.to_index(date=0, variable="cos_julian_day", member=0)] == 1.0), "cos(julian_day = 0) == 1"
    assert np.all(ds.data[ds.to_index(date=0, variable="u_101", member=0)] == 42.0), "artificially constant data day 0"
    assert np.all(ds.data[ds.to_index(date=1, variable="v_119", member=0)] == 21.0), "artificially constant data day 1"
    assert ds.data[ds.to_index(date=0, variable="cos_latitude", member=0)].max() > 0.9
    assert ds.data[ds.to_index(date=0, variable="cos_latitude", member=0)].min() >= 0
    assert ds.data[ds.to_index(date=0, variable="sin_latitude", member=0)].max() > 0.9
    assert ds.data[ds.to_index(date=0, variable="sin_latitude", member=0)].min() < -0.9


def test_grib_gridfile_with_key_types(get_test_data: callable) -> None:
    """Test the creation of a dataset from GRIB files with an unstructured grid.

    This function tests eccodes key type formatters.
    """

    p = "anemoi-datasets/create/test_grib_gridfile_with_refinement_level/"
    data1 = get_test_data(p + "2023010103+fc_R03B07_rea_ml.2023010100")
    data2 = get_test_data(p + "2023010106+fc_R03B07_rea_ml.2023010103")
    gridfile = get_test_data("dwd/2024-12-11_00/icon_grid_0026_R03B07_subsetAICON.nc")
    assert os.path.dirname(data1) == os.path.dirname(data2)

    path = os.path.dirname(data1)

    config = {
        "dates": {
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T03:00:00",
            "frequency": "3h",
        },
        "input": {
            "grib": {
                "path": os.path.join(
                    path, "{date:strftimedelta(+3h;%Y%m%d%H)}+fc_R03B07_rea_ml.{date:strftime(%Y%m%d%H)}"
                ),
                "grid_definition": {"icon": {"path": gridfile}},
                "param": ["u"],
                "level:d": [101.0, 119.0],
            },
        },
        "build": {
            "variable_naming": "{param}_{level:d}",
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.to_index(date=0, variable="u_101.0", member=0) != ds.to_index(date=0, variable="u_119.0", member=0)

    with pytest.raises(ValueError):
        ds.to_index(date=0, variable="u_101", member=0)  # param does not exist
    with pytest.raises(ValueError):
        ds.to_index(date=0, variable="u_119", member=0)  # param does not exist


@skip_if_offline
def test_netcdf(get_test_data: callable) -> None:
    """Test for NetCDF files.

    This function tests the creation of a dataset from a NetCDF file.
    """
    data = get_test_data("anemoi-datasets/create/netcdf.nc")
    config = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "netcdf": {"path": data},
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@skip_missing_packages("fstd", "rpnpy.librmn")
def test_eccs_fstd(get_test_data: callable) -> None:
    """Test for 'fstd' files from ECCC."""
    # See https://github.com/neishm/fstd2nc

    data = get_test_data("anemoi-datasets/create/2025031000_000_TT.fstd", gzipped=True)
    config = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "eccc_fstd": {"path": data},
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@pytest.mark.slow
@skip_if_offline
@skip_missing_packages("kerchunk", "s3fs")
def test_kerchunk(get_test_data: callable) -> None:
    """Test for Kerchunk JSON files.

    This function tests the creation of a dataset from a Kerchunk JSON file.

    """
    # Note: last version of kerchunk compatible with zarr 2 is 0.2.7

    data = get_test_data("anemoi-datasets/create/kerchunck.json", gzipped=True)

    config = {
        "dates": {
            "start": "2024-03-01T00:00:00",
            "end": "2024-03-01T18:00:00",
            "frequency": "6h",
        },
        "input": {
            "xarray-kerchunk": {
                "json": data,
                "param": ["T"],
                "level": [1000],
            },
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (4, 1, 1, 1038240)


@skip_if_offline
@skip_missing_packages("planetary_computer", "adlfs")
def test_planetary_computer_conus404() -> None:
    """Test loading and validating the planetary_computer_conus404 dataset."""

    config = {
        "dates": {
            "start": "2022-01-01",
            "end": "2022-01-02",
            "frequency": "1d",
        },
        "input": {
            "planetary_computer": {
                "data_catalog_id": "conus404",
                "param": ["Z"],
                "level": [1],
                "patch": {
                    "coordinates": ["bottom_top_stag"],
                    "rename": {
                        "bottom_top_stag": "level",
                    },
                    "attributes": {
                        "lon": {"standard_name": "longitude", "long_name": "Longitude"},
                        "lat": {"standard_name": "latitude", "long_name": "Latitude"},
                    },
                },
            }
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 1, 1, 1387505), ds.shape


if __name__ == "__main__":
    test_planetary_computer_conus404()
    exit(0)
    from anemoi.utils.testing import run_tests

    run_tests(globals())
