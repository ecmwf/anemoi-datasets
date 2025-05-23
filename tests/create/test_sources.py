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
        'class',
        'date',
        'expver',
        'level',
        'levelist',
        'levtype',
        'number',
        'paramId',
        'shortName',
        'step',
        'stream',
        'time',
        'type',
        'valid_datetime'
        ]

    data1 = []
    for file in filelist:
        data1.append(get_test_data(f"meteo-france/grib/{file}"))
        parent = os.path.dirname(os.path.dirname(data1[-1]))
    assert all([os.path.dirname(os.path.dirname(d))==parent for d in data1])
    
    path_db = os.path.dirname(data1[-1])
    from anemoi.datasets.create.sources.grib_index import GribIndex

    # create a database with grib files
    index = GribIndex(
        os.path.join(
            path_db,
            'grib-index-accumulate-tp.db'
            ),
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

    
    # creating configuration
    config_grib_index = {
        "dates": {
            "start": "2021-01-01T18:00:00",
            "end": "2021-01-02T02:00:00",
            "frequency": "1h",
        },
        "input": 
            {"pipe" : [{"accumulate": {"source": {
                        "grib-index": {
                            "indexdb" : os.path.join(
                                        path_db,
                                        'grib-index-accumulate-tp.db'
                                        ),
                            "levtype" : "sfc",
                            "param" : ["tp"],
                            "accumulation_period" : 6,
                            },
                    },
                }},
            {"remove-nans" : {}} # needed because data has Nans due to projection
            ]
            },
    }

    created = create_dataset(config=config_grib_index, output=None)
    ds = open_dataset(created)
    print(ds.shape)

    # get a reference zarr
    data2 = get_test_data("meteo-france/zarr/tp-test.zarr")
    ds2 = open_dataset(data2)
    print(ds2.shape)

    # shapes should be offset by 'accumulation_period' since frequency is 1h
    assert ds.shape[0] == ds2.shape[0] - 6, (ds.shape, ds2.shape)

    assert np.nansum(ds[0]) == np.nansum(ds2[:6]), (np.nansum(ds[0]), np.nansum(ds2[:6]))
    assert np.nansum(ds[2]) == np.nansum(ds2[2:8]), (np.nansum(ds[2]), np.nansum(ds2[2:8]))
    assert np.nansum(ds[5]) == np.nansum(ds2[5:11]), (np.nansum(ds[5]), np.nansum(ds2[5:11]))

    # this construction should fail because dates are missing
    config_grib_index['input']['pipe'][0]['source']['accumulation_period'] = 24

    try:
        created = create_dataset(config=config_grib_index, output=None)
    except AssertionError as e:
        print(f"Wrong dates correctly detected {e}")
        pass
    
    # this construction should fail because dates are missing
    config_grib_index['input']['pipe'][0]['source']['accumulation_period'] = 3
    config_grib_index['dates']['frequency'] = 3
    
    created = create_dataset(config=config_grib_index, output=None)
    ds = open_dataset(created)
    print(ds.shape)

    assert ds.shape == ds2.shape//3, (ds.shape, ds2.shape)

    assert np.nansum(ds[0]) == np.nansum(ds2[2:5]), (np.nansum(ds[0]), np.nansum(ds2[2:5]))
    assert np.nansum(ds[1]) == np.nansum(ds2[5:8]), (np.nansum(ds[1]), np.nansum(ds2[5:8]))
    assert np.nansum(ds[3]) == np.nansum(ds2[8:11]), (np.nansum(ds[3]), np.nansum(ds2[8:11]))


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
