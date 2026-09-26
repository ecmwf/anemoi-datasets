# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages

from anemoi.datasets import open_dataset

from .utils.create import create_dataset


def _run_one_group(recipe):
    from anemoi.datasets.create.input.builder import InputBuilder
    from anemoi.datasets.create.input.context import Context
    from anemoi.datasets.create.recipe import Recipe
    from anemoi.datasets.dates.groups import Groups

    class TestContext(Context):
        def create_result(self, argument: Any, data: Any) -> Any:
            return data

    recipe = Recipe(**recipe)
    builder = InputBuilder(recipe.input, recipe.data_sources)
    group = Groups(recipe.dates, recipe.build.group_by)
    return builder.select(TestContext(recipe), next(iter(group)))


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

    recipe = {
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

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.shape == (8, 12, 1, 162)


@skip_if_offline
def test_accumulate_grib_index(get_test_data: callable) -> None:
    """Test the creation of a accumulation from grib index.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    """

    filelist = [
        "2021-01-01_23h00/mod_precip.grib",
        "2021-01-02_00h00/mod_precip.grib",
        "2021-01-02_01h00/mod_precip.grib",
        "2021-01-02_02h00/mod_precip.grib",
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

    for path in paths:
        index.add_grib_file(path)

    reference_config = {
        "dates": {
            "start": "2021-01-02T00:00:00",
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

    # get a reference dataset
    reference = create_dataset(recipe=reference_config, output=None)
    ds2 = open_dataset(reference)
    # creating configuration using the previously created grib-index
    config_grib_index = {
        "dates": {
            "start": "2021-01-02T02:00:00",
            "end": "2021-01-02T02:00:00",
            "frequency": "1h",
        },
        "input": {
            "pipe": [
                {
                    "accumulate": {
                        "period": 3,  # requesting 3 hour accumulation
                        "availability": "1h",  # available data is accumulated every 1 hour
                        "source": {
                            "grib-index": {
                                "indexdb": os.path.join(path_db, "grib-index-accumulate-tp.db"),
                                "levtype": "sfc",
                                "param": ["tp"],
                            },
                        },
                    }
                },
                {"remove-nans": {}},  # needed because data has Nans due to projection
            ]
        },
    }

    created = create_dataset(recipe=config_grib_index, output=None)
    ds = open_dataset(created)

    # shapes should be divided by 'accumulation_period'
    assert ds.shape[0] == ds2.shape[0] // 3, (ds.shape, ds2.shape)

    assert np.max(np.abs(ds[0] - np.sum(ds2[:3], axis=(0, 1, 2)))) <= 1e-3, (
        "max of absolute difference, t=0",
        np.max(np.abs(ds[0] - np.sum(ds2[:3], axis=(0, 1, 2)))),
    )

    # this construction should fail because dates are missing
    config_grib_index["input"]["pipe"][0]["accumulate"]["source"]["accumulation_period"] = 24

    with pytest.raises(Exception):
        created = create_dataset(recipe=config_grib_index, output=None)


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

    recipe = {
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

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.shape == (8, 1, 1, 1147980)
    assert ds.variables == ["2t"]


@skip_if_offline
@pytest.mark.parametrize(
    "input_refinement_level_c,output_refinement_level_c,shape",
    (
        (7, 2, (2, 13, 1, 2880)),
        (7, 7, (2, 13, 1, 2949120)),
    ),
)
def test_grib_gridfile_with_refinement_level(
    input_refinement_level_c: str,
    output_refinement_level_c: str,
    shape: tuple[int, int, int, int, int],
    get_test_data: callable,
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
        "path": os.path.join(
            path,
            "{date:strftimedelta(+3h;%Y%m%d%H)}+fc_R03B07_rea_ml.{date:strftime(%Y%m%d%H)}",
        ),
        "grid_definition": {
            "icon": {
                "path": gridfile,
                "refinement_level_c": None,
            }
        },
        "param": param,
        "level": level,
    }
    refinement_filter = {
        "icon_refinement_level": {
            "grid": gridfile,
            "refinement_level_c": output_refinement_level_c,
        }
    }

    recipe = {
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
                        {
                            "forcings": {
                                "param": forcings,
                                "template": "${input.pipe.0.join.0.grib}",
                            }
                        },
                    ]
                },
                refinement_filter,
            ]
        },
    }

    created = create_dataset(recipe=recipe, output=None)
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

    recipe = {
        "dates": {
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T03:00:00",
            "frequency": "3h",
        },
        "input": {
            "grib": {
                "path": os.path.join(
                    path,
                    "{date:strftimedelta(+3h;%Y%m%d%H)}+fc_R03B07_rea_ml.{date:strftime(%Y%m%d%H)}",
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

    created = create_dataset(recipe=recipe, output=None)
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
    recipe = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "netcdf": {"path": data},
        },
    }

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@skip_missing_packages("fstd", "rpnpy.librmn")
def test_eccs_fstd(get_test_data: callable) -> None:
    """Test for 'fstd' files from ECCC."""
    # See https://github.com/neishm/fstd2nc

    data = get_test_data("anemoi-datasets/create/2025031000_000_TT.fstd", gzipped=True)
    recipe = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "eccc_fstd": {"path": data},
        },
    }

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@pytest.mark.slow
@skip_if_offline
@skip_missing_packages("kerchunk", "s3fs")
@pytest.mark.parametrize("input_kind", ["path", "string", "dict", "url"])
def test_kerchunk(get_test_data: Callable[[str, bool], str], input_kind: str) -> None:
    """Test for Kerchunk JSON files with different input kinds."""
    storage_opts = {"remote_protocol": "s3", "remote_options": {"anon": True}}
    if input_kind == "url":
        from anemoi.utils.testing import url_for_test_data

        data_path = url_for_test_data("anemoi-datasets/create/kerchunck.json.gz")
        json_opts = {"compression": "gzip"}
    else:
        data_path = get_test_data("anemoi-datasets/create/kerchunck.json", gzipped=True)
        json_opts = {}

    if input_kind in ["path", "url"]:
        json_arg = data_path
    elif input_kind == "string":
        json_arg = Path(data_path).read_text()
    else:
        with Path(data_path).open() as f:
            json_arg = json.load(f)

    recipe = {
        "dates": {"start": "2024-03-01T00:00:00", "end": "2024-03-01T18:00:00", "frequency": "6h"},
        "input": {
            "xarray-kerchunk": {
                "json": json_arg,
                "json_options": json_opts,
                "storage_options": storage_opts,
                "param": ["T"],
                "level": [1000],
            },
        },
    }
    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.shape == (4, 1, 1, 1038240)


@skip_if_offline
@skip_missing_packages("planetary_computer", "adlfs")
def test_planetary_computer_conus404() -> None:
    """Test loading and validating the planetary_computer_conus404 dataset."""
    import pystac_client

    recipe = {
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
    try:
        created = create_dataset(recipe=recipe, output=None)
    except pystac_client.exceptions.APIError:
        pytest.skip("Planetary Computer data catalog is not available")
    ds = open_dataset(created)
    assert ds.shape == (2, 1, 1, 1387505), ds.shape


@skip_if_offline
@skip_missing_packages("planetary_computer", "adlfs", "h5netcdf", "h5py")
@pytest.mark.parametrize(
    ("collection_id", "year", "param", "search_params", "shape"),
    [
        (
            "era5-pds",
            2020,
            ["surface_air_pressure"],
            {
                "filter": {"op": "=", "args": [{"property": "era5:kind"}, "an"]},
            },
            (2, 1, 1, 1038240),
        ),
        (
            "met-office-global-deterministic-near-surface",
            2026,
            ["rainfall_rate", "lwe_snowfall_rate"],
            {
                "variable_key_map": {
                    "lwe_snowfall_rate": "snowfall_rate",
                },
                "filter": {
                    "op": "and",
                    "args": [
                        {
                            "op": "=",
                            "args": [{"property": "forecast:horizon"}, "PT0000H00M"],
                        },
                        {
                            "op": "=",
                            "args": [
                                {"property": "met_office_deterministic:model"},
                                "global",
                            ],
                        },
                    ],
                },
            },
            (2, 2, 1, 4915200),
        ),
    ],
)
def test_planetary_computer_multipart(
    collection_id: str,
    year: int | str,
    param: list[str],
    search_params: dict,
    shape: tuple[int, int, int, int],
) -> None:
    """Test loading and validating Planetary Computer collection data.

    Parameterised to test two multipart collections:
    - era5-pds: multiple timesteps per ABFS URI Zarr store
    - met-office-global-deterministic-near-surface: one timestep per HTTPS URL NetCDF
    file with data variable name:STAC key inequality
    """
    import pystac_client

    search_params = {
        **search_params,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-01-01T06:00:00Z",
    }

    recipe = {
        "dates": {
            "start": f"{year}-01-01T00:00:00",
            "end": f"{year}-01-01T06:00:00",
            "frequency": "6h",
        },
        "input": {
            "planetary_computer": {
                "data_catalog_id": collection_id,
                "azure_log_level": "WARNING",
                "param": param,
                "search_params": search_params,
            },
        },
    }
    try:
        created = create_dataset(recipe=recipe, output=None)
    except pystac_client.exceptions.APIError:
        pytest.skip("Planetary Computer data catalog is not available")
    ds = open_dataset(created)
    assert ds.shape == shape, ds.shape


@skip_if_offline
def test_csv(get_test_data: callable) -> None:
    """Test for CSV source registration."""

    data = get_test_data("anemoi-datasets/obs/dribu.csv")
    recipe = {
        "dates": {
            "start": "2025-01-01T00:00:00",
            "end": "2025-12-21T23:59:59",
        },
        "input": {
            "csv": {
                "path": data,
                "flavour": {
                    "date": [
                        "typicalDate",
                        "typicalTime",
                    ]
                },
            }
        },
    }
    frame = _run_one_group(recipe)

    assert len(frame) == 2526

    assert "latitude" in frame.columns, frame.columns
    assert "longitude" in frame.columns, frame.columns
    assert "date" in frame.columns, frame.columns

    assert frame["latitude"].dtype == float or np.issubdtype(frame["latitude"].dtype, np.floating)
    assert frame["longitude"].dtype == float or np.issubdtype(frame["longitude"].dtype, np.floating)
    assert frame["date"].dtype == "datetime64[ns]" or np.issubdtype(frame["date"].dtype, np.datetime64)


# @pytest.mark.skip(reason="ODB source currently not functional")
@skip_if_offline
@pytest.mark.skipif(shutil.which("odc") is None, reason="odc command not accessible")
def test_odb(get_test_data: callable) -> None:

    data = get_test_data("anemoi-datasets/obs/dribu.odb")
    recipe = {
        "dates": {
            "start": "2025-01-01T00:00:00",
            "end": "2025-12-21T23:59:59",
        },
        "input": {"odb": {"path": data}},
    }
    frame = _run_one_group(recipe)

    assert len(frame) == 6838

    assert "latitude" in frame.columns, frame.columns
    assert "longitude" in frame.columns, frame.columns
    assert "date" in frame.columns, frame.columns

    assert frame["latitude"].dtype == float or np.issubdtype(frame["latitude"].dtype, np.floating)
    assert frame["longitude"].dtype == float or np.issubdtype(frame["longitude"].dtype, np.floating)
    assert frame["date"].dtype == "datetime64[ns]" or np.issubdtype(frame["date"].dtype, np.datetime64)


@pytest.mark.skip(reason="BUFR source currently not functional")
@skip_if_offline
def test_bufr(get_test_data: callable) -> None:

    from anemoi.datasets.create.recipe.dates import StartEndDates
    from anemoi.datasets.create.sources import create_source
    from anemoi.datasets.dates.groups import GroupOfDates

    data = get_test_data("anemoi-datasets/obs/dribu.bufr")

    source = create_source(context=None, config={"bufr": {"path": data}})
    provider = StartEndDates(
        start="2020-01-01T00:00:00",
        end="2020-01-02T23:59:59",
    )
    dates = GroupOfDates(provider.values, provider)

    source.execute(dates)
