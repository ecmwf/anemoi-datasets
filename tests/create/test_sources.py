# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import shutil
from typing import Any

import numpy as np
import pytest
import zarr
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
                                # a flavour masks the raw GRIB keys the accumulation reads
                                "flavour": [[{"levtype": "sfc"}, {"levelist": None}]],
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


def _write_grib(path, messages: list[dict]) -> str:
    """Write one GRIB2 message per dict of eccodes keys, with every value set to the dict's ``value``."""
    import eccodes

    with open(path, "wb") as f:
        for keys in messages:
            keys = dict(keys)
            value = keys.pop("value")
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            for k, v in keys.items():
                eccodes.codes_set(h, k, v)
            eccodes.codes_set_values(h, np.full(eccodes.codes_get(h, "numberOfValues"), float(value)))
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)
    return str(path)


def test_grib_ensemble_with_filter(tmp_path) -> None:
    """Ensemble members keep their order and count after a filter, and the MARS metadata follows the filter."""
    members = [1, 2, 10]
    path = _write_grib(
        tmp_path / "ens.grib",
        [
            dict(
                dataDate=20200101,
                dataTime=0,
                shortName="sp",
                productDefinitionTemplateNumber=1,
                perturbationNumber=number,
                numberOfForecastsInEnsemble=len(members),
                value=number,
            )
            for number in members
        ],
    )

    recipe = {
        "dates": {"start": "2020-01-01T00:00:00", "end": "2020-01-01T00:00:00", "frequency": "6h"},
        "input": {"pipe": [{"grib": {"path": path}}, {"rename": {"param": {"sp": "surface_pressure"}}}]},
    }

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.variables == ["surface_pressure"]
    assert ds.shape[2] == len(members)
    assert [float(ds[0, 0, m, 0]) for m in range(len(members))] == members

    mars = ds.variables_metadata["surface_pressure"]["mars"]
    assert (mars["param"], mars["levtype"]) == ("surface_pressure", "sfc")
    assert ds.variables_metadata["surface_pressure"]["units"] == "Pa"
    assert zarr.open(created, mode="r").attrs["data_request"]["param_level"] == {"sfc": ["surface_pressure"]}


def _write_icon_grid(path, npoints: int) -> tuple[str, np.ndarray, np.ndarray]:
    """Write a minimal ICON grid file (cell centres only) with ``npoints`` points."""
    import xarray as xr

    latitudes, longitudes = np.linspace(40.0, 50.0, npoints), np.linspace(0.0, 10.0, npoints)
    xr.Dataset({"clat": ("cell", np.deg2rad(latitudes)), "clon": ("cell", np.deg2rad(longitudes))}).to_netcdf(path)
    return str(path), latitudes, longitudes


def test_fdb(tmp_path, monkeypatch) -> None:
    """The fdb source returns the fields of its requests (FDB itself is mocked), flavoured, then on the grid."""
    import earthkit.data as ekd

    path = _write_grib(tmp_path / "fdb.grib", [dict(dataDate=20200101, dataTime=0, shortName="sp", value=1)])
    gridfile, latitudes, longitudes = _write_icon_grid(tmp_path / "grid.nc", 496)
    from_source = ekd.from_source

    def mock_from_source(name, *args, **kwargs):
        if name != "fdb":
            return from_source(name, *args, **kwargs)
        assert "read_all" not in kwargs  # rejected by the earthkit-data 1.x fdb source
        return from_source("stream", open(path, "rb"))

    monkeypatch.setattr(ekd, "from_source", mock_from_source)

    recipe = {
        "dates": {"start": "2020-01-01T00:00:00", "end": "2020-01-01T00:00:00", "frequency": "6h"},
        "input": {
            "fdb": {
                "fdb_request": {"class": "od", "stream": "oper", "type": "an", "param": ["sp"]},
                "param_id_map": {"sp": 134},
                "flavour": [[{"shortName": "sp"}, {"param": "surface_pressure"}]],
                "grid_definition": {"icon": {"path": gridfile}},
            }
        },
    }

    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.variables == ["surface_pressure"]
    assert float(ds[0, 0, 0, 0]) == 1
    assert np.allclose(ds.latitudes, latitudes) and np.allclose(ds.longitudes, longitudes)


def _grib_index(tmp_path, messages: list[dict]) -> str:
    """Index GRIB messages written with ``_write_grib`` in a new grib-index database."""
    from anemoi.datasets.create.sources.grib_index import GribIndex

    path = str(tmp_path / "index.db")
    GribIndex(path, update=True, overwrite=True).add_grib_file(_write_grib(tmp_path / "index.grib", messages))
    return path


def test_accumulate_with_grid_and_flavour(tmp_path) -> None:
    """Accumulated fields keep the grid and the flavour their source applied."""
    hourly = [
        dict(
            dataDate=20200101, dataTime=hour * 100, stepType="accum", shortName="tp", startStep=0, endStep=1, value=hour
        )
        for hour in range(3)
    ]
    gridfile, latitudes, longitudes = _write_icon_grid(tmp_path / "grid.nc", 496)
    recipe = {
        "dates": {"start": "2020-01-01T03:00:00", "end": "2020-01-01T03:00:00", "frequency": "3h"},
        "input": {
            "accumulate": {
                "period": "3h",
                "availability": "1h",
                "source": {
                    "grib-index": {
                        "indexdb": _grib_index(tmp_path, hourly),
                        "param": "tp",
                        "flavour": [[{"shortName": "tp"}, {"param": "precip"}]],
                        "grid_definition": {"icon": {"path": gridfile}},
                    }
                },
            }
        },
    }

    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.variables == ["precip"]
    assert np.all(ds[0, 0, 0] == 0 + 1 + 2)
    assert np.allclose(ds.latitudes, latitudes) and np.allclose(ds.longitudes, longitudes)
    assert ds.variables_metadata["precip"]["process"] == "accumulation"


def test_accumulate_grib_index_with_basetime(tmp_path) -> None:
    """A grib-index of forecasts (accumulated from their base time) serves accumulations over any window."""
    forecast = [
        dict(dataDate=20200101, dataTime=0, stepType="accum", shortName="tp", startStep=0, endStep=step, value=step)
        for step in range(1, 7)
    ]
    recipe = {
        "dates": {"start": "2020-01-01T06:00:00", "end": "2020-01-01T06:00:00", "frequency": "3h"},
        "input": {
            "accumulate": {
                "period": "3h",
                "covering": {"auto": {"accumulated-from-start": {"basetime": [0], "frequency": 1, "last_step": 6}}},
                "source": {"grib-index": {"indexdb": _grib_index(tmp_path, forecast), "param": "tp"}},
            }
        },
    }

    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert np.all(ds[0, 0, 0] == 6 - 3)


def test_forcings_metadata_from_accumulation_template(tmp_path) -> None:
    """Computed forcings do not take the GRIB metadata of their template."""
    path = _write_grib(
        tmp_path / "tp.grib",
        [dict(dataDate=20200101, dataTime=0, stepType="accum", shortName="tp", startStep=0, endStep=6, value=1)],
    )
    recipe = {
        "dates": {"start": "2020-01-01T06:00:00", "end": "2020-01-01T06:00:00", "frequency": "6h"},
        "input": {
            "join": [
                {"grib": {"path": path}},
                {"forcings": {"param": ["cos_latitude"], "template": "${input.join.0.grib}"}},
            ]
        },
    }

    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.variables_metadata["tp"]["process"] == "accumulation"
    assert not {"mars", "grib", "process", "period"} & set(ds.variables_metadata["cos_latitude"])


def test_check_variables_compatibility(tmp_path) -> None:
    """The recipe can relax the check that the variables of every part of the dataset are compatible."""
    path = _write_grib(
        tmp_path / "tp.grib",
        [
            dict(dataDate=20200101, dataTime=0, stepType="accum", shortName="tp", startStep=0, endStep=step, value=1)
            for step in (6, 12)
        ],
    )
    recipe = {
        "dates": {"start": "2020-01-01T06:00:00", "end": "2020-01-01T12:00:00", "frequency": "6h"},
        "build": {"group_by": 1},
        "input": {"grib": {"path": path}},
    }

    with pytest.raises(ValueError, match="not compatible"):
        create_dataset(recipe=recipe, output=None)

    recipe["build"]["check_variables_compatibility"] = {"ignore_processing_period": True}
    assert open_dataset(create_dataset(recipe=recipe, output=None)).shape[0] == 2


def test_selection_by_other_keys(tmp_path) -> None:
    """Recipe keys outside the legacy names still select fields in the grib and xarray sources."""
    import xarray as xr

    path = _write_grib(
        tmp_path / "levels.grib",
        [dict(dataDate=20200101, dataTime=0, shortName=name, value=1) for name in ("sp", "msl")],
    )
    dates = {"start": "2020-01-01T00:00:00", "end": "2020-01-01T00:00:00", "frequency": "6h"}

    recipe = {"dates": dates, "input": {"grib": {"path": path, "typeOfLevel": "surface"}}}
    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.variables == ["sp"]

    lat, lon = np.arange(50.0, 40.0, -1.0), np.arange(0.0, 10.0, 1.0)
    xr.Dataset(
        {"t2m": (("number", "time", "latitude", "longitude"), np.ones((3, 1, lat.size, lon.size)), {"units": "K"})},
        coords={
            "number": [0, 1, 2],
            "time": [np.datetime64("2020-01-01T00:00", "ns")],
            "latitude": ("latitude", lat, {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", lon, {"standard_name": "longitude", "units": "degrees_east"}),
        },
    ).to_netcdf(tmp_path / "ens.nc")

    recipe = {"dates": dates, "input": {"netcdf": {"path": str(tmp_path / "ens.nc"), "number": [1, 2]}}}
    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    assert ds.shape[2] == 2


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

    # unstructured grid: resolution "unknown", no MARS grid, the area is the bounding box
    data_request = zarr.open(created, mode="r").attrs["data_request"]
    assert ds.resolution == "unknown"
    assert data_request["grid"] is None
    assert data_request["area"] == pytest.approx(
        [ds.latitudes.max(), ds.longitudes.min(), ds.latitudes.min(), ds.longitudes.max()]
    )


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

    This function tests eccodes key type formatters.  The ``level:d`` source
    key selects fields by float-typed level.  The variable naming uses the
    plain ``{param}_{level}`` template because grid-wrapped fields (produced
    by ``grid_definition``) do not carry eccodes typed-key metadata
    (``metadata.level:d``); only the component path ``vertical.level`` (int)
    is available.
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
            "variable_naming": "{param}_{level}",
        },
    }

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
    assert ds.to_index(date=0, variable="u_101", member=0) != ds.to_index(date=0, variable="u_119", member=0)

    with pytest.raises(ValueError):
        ds.to_index(date=0, variable="u_101.0", member=0)  # float names not produced
    with pytest.raises(ValueError):
        ds.to_index(date=0, variable="u_119.0", member=0)  # float names not produced


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
def test_kerchunk(get_test_data: callable) -> None:
    """Test for Kerchunk JSON files.

    This function tests the creation of a dataset from a Kerchunk JSON file.

    """
    # Note: last version of kerchunk compatible with zarr 2 is 0.2.7

    data = get_test_data("anemoi-datasets/create/kerchunck.json", gzipped=True)

    recipe = {
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

    created = create_dataset(recipe=recipe, output=None)
    ds = open_dataset(created)
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
