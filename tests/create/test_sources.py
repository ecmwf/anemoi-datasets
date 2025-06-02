# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sys

import numpy as np
import pytest
import tqdm
from anemoi.utils.testing import get_test_data
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages
from anemoi.utils.testing import skip_slow_tests

from anemoi.datasets import open_dataset
from anemoi.datasets.create.testing import create_dataset


@skip_if_offline
def test_grib() -> None:
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
def test_accumulate_grib_index() -> None:
    """Test the creation of a accumulation from grib index.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    """

    filelist = [
        "2021-01-01_11h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_12h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_13h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_14h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_15h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_16h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_17h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_18h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_19h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_20h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_21h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_22h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-01_23h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-02_00h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-02_01h00/PAAROME_1S100_ECH1_SOL.grib",
        "2021-01-02_02h00/PAAROME_1S100_ECH1_SOL.grib",
    ]

    keys = [
        "class",
        "date",
        "expver",
        "level",
        "levelist",
        "levtype",
        "number",
        "paramId",
        "shortName",
        "step",
        "stream",
        "time",
        "type",
        "valid_datetime",
    ]

    data1 = []
    for file in filelist:
        data1.append(get_test_data(f"meteo-france/grib/{file}"))
        parent = os.path.dirname(os.path.dirname(data1[-1]))
    assert all([os.path.dirname(os.path.dirname(d)) == parent for d in data1])

    path_db = os.path.dirname(data1[-1])
    from anemoi.datasets.create.sources.grib_index import GribIndex

    # create a GribIndex database with grib files
    index = GribIndex(
        os.path.join(path_db, "grib-index-accumulate-tp.db"),
        keys=keys,
        update=True,
        overwrite=True,
        flavour=None,
    )

    paths = []
    for path in data1:
        if os.path.isfile(path):
            paths.append(path)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    full = os.path.join(root, file)
                    paths.append(full)

    for path in tqdm.tqdm(data1, leave=False):
        index.add_grib_file(path)

    reference_config = {
        "dates": {
            "start": "2021-01-01T12:00:00",
            "end": "2021-01-02T02:00:00",
            "frequency": "1h",
        },
        "input": {
            "pipe": [
                {
                    "grib-index": {
                        "indexdb": os.path.join(path_db, "grib-index-accumulate-tp.db"),
                        "levtype": "sfc",
                        "param": ["tp"],
                    }
                },
                {"remove-nans": {}},
            ]
        },
    }

    # get a reference daatset
    reference = create_dataset(config=reference_config, output=None)
    ds2 = open_dataset(reference)

    # creating configuration using the previously created grib-index
    config_grib_index = {
        "dates": {
            "start": "2021-01-01T18:00:00",
            "end": "2021-01-02T02:00:00",
            "frequency": "1h",
        },
        "input": {
            "pipe": [
                {
                    "accumulate": {
                        "source": {
                            "grib-index": {
                                "indexdb": os.path.join(path_db, "grib-index-accumulate-tp.db"),
                                "levtype": "sfc",
                                "param": ["tp"],
                                "accumulation_period": 6,
                            },
                        },
                    }
                },
                {"remove-nans": {}},  # needed because data has Nans due to projection
            ]
        },
    }

    created = create_dataset(config=config_grib_index, output=None)
    ds = open_dataset(created)

    # shapes should be offset by 'accumulation_period' since frequency is 1h
    assert ds.shape[0] == ds2.shape[0] - 6, (ds.shape, ds2.shape)
    
    assert (np.max(np.abs(ds[0] - np.sum(ds2[1:7], axis=(0,1,2))))<=1e-3), ("max of absolute difference, t=0", (np.max(np.abs(ds[0] - np.sum(ds2[1:7], axis=(0,1,2))))<=1e-3))
    assert (np.max(np.abs(ds[2] - np.sum(ds2[3:9], axis=(0,1,2))))<=1e-3), ("max of absolute difference, t=2", (np.max(np.abs(ds[2] - np.sum(ds2[3:9], axis=(0,1,2))))<=1e-3))
    assert (np.max(np.abs(ds[5] - np.sum(ds2[6:12], axis=(0,1,2))))<=1e-3), ("max of absolute difference, t=5", (np.max(np.abs(ds[5] - np.sum(ds2[6:12], axis=(0,1,2))))<=1e-3))

    # this construction should fail because dates are missing
    config_grib_index["input"]["pipe"][0]["accumulate"]["source"]["grib-index"]["accumulation_period"] = 24

    with pytest.raises(Exception) as e_info:
        created = create_dataset(config=config_grib_index, output=None)

    # this construction should fail because dates are missing
    config_grib_index["input"]["pipe"][0]["accumulate"]["source"]["grib-index"]["accumulation_period"] = 3
    config_grib_index["dates"]["frequency"] = 3

    created = create_dataset(config=config_grib_index, output=None)
    ds = open_dataset(created)

    assert ds.shape[0] == 3, ("shape mismatch", ds.shape, ds2.shape)

    assert np.allclose(np.max(ds[0]), np.max(np.sum(ds2[4:7], axis=(0,1,2))), rtol=1e-4), ("t=0",np.max(ds[0]), np.max(np.sum(ds2[4:7], axis=(0,1,2))))
    assert np.allclose(np.max(ds[1]), np.max(np.sum(ds2[7:10], axis=(0,1,2))), rtol=1e-4), ("t=1",np.max(ds[1]), np.max(np.sum(ds2[7:10], axis=(0,1,2))))
    assert np.allclose(np.max(ds[2]), np.max(np.sum(ds2[10:13], axis=(0,1,2))), rtol=1e-4), ("t=2",np.max(ds[2]), np.max(np.sum(ds2[10:13], axis=(0,1,2))))


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Type hints from anemoi-transform are not compatible with Python < 3.10"
)
@skip_if_offline
def test_grib_gridfile() -> None:
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


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Type hints from anemoi-transform are not compatible with Python < 3.10"
)
@skip_if_offline
@pytest.mark.parametrize(
    "refinement_level_c,shape",
    (
        (2, (2, 13, 1, 2880)),
        (7, (2, 13, 1, 2949120)),
    ),
)
def test_grib_gridfile_with_refinement_level(refinement_level_c: str, shape: tuple[int, int, int, int, int]) -> None:
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
        "grid_definition": {"icon": {"path": gridfile, "refinement_level_c": refinement_level_c}},
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


@skip_if_offline
def test_netcdf() -> None:
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
def test_eccs_fstd() -> None:
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


@skip_slow_tests
@skip_if_offline
@skip_missing_packages("kerchunk", "s3fs")
def test_kerchunk() -> None:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_kerchunk()
    exit()
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
