# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Exercise every gridded ``open_dataset`` view/class, for every parameter form.

Each ``open_dataset`` option in ``usage/dataset.py::Dataset.__subset`` and each
top-level combining factory in ``usage/misc.py::_open_dataset`` builds a distinct
``Dataset`` subclass. This module opens one of each -- and every accepted form of
its parameters -- using ``synthetic`` datasets as the building blocks, so the
whole suite runs offline with no Zarr store.

``synthetic`` itself is not tested here (see ``tests/test_synthetic.py``); it is
the tool used to feed the other classes.

Trajectory- and tabular-only options (``step``, ``steps``, ``step_start``,
``step_end``, ``step_frequency``, ``base_start``, ``base_end``, ``base_frequency``,
``set_missing_base_dates`` and ``tensors``) build views over non-gridded layouts
that ``synthetic`` does not (yet) produce, so they are out of scope here.
"""

import numpy as np
import pytest

from anemoi.datasets import open_dataset

# --------------------------------------------------------------------------
# Synthetic building blocks
# --------------------------------------------------------------------------
# A 3x3 regular grid (field_shape (3, 3), 9 gridpoints) by default so that views
# needing a 2D field (e.g. TrimEdge) work; five 6-hourly dates by default.
_BBOX = [4.0, 0.0, 0.0, 4.0]
_RESOLUTION = 2.0
_START = "2020-01-01"
_END = "2020-01-02"
_FREQUENCY = "6h"


def synthetic(
    *,
    variables=("a", "b", "c"),
    bbox=_BBOX,
    resolution=_RESOLUTION,
    start=_START,
    end=_END,
    frequency=_FREQUENCY,
    ensembles=1,
    values=None,
) -> dict:
    """Return a ``{"synthetic": {...}}`` spec for a gridded dataset.

    Wrapping the spec in the ``synthetic`` key means it can be passed straight to
    ``open_dataset`` (``open_dataset(synthetic(), select=...)``) or dropped into a
    combining factory (``open_dataset(join=[synthetic(), synthetic()])``).
    """
    spec = {
        "geography": {"bbox": list(bbox), "resolution": resolution},
        "variables": list(variables),
        "dates": {"start": start, "end": end, "frequency": frequency},
        "layout": "gridded",
        "ensembles": ensembles,
    }
    spec["values"] = values if values is not None else {"constant": 1.0}
    return {"synthetic": spec}


def _test_dataset(ds, variables=None):
    assert len(ds) >= 0
    if variables is not None:
        assert ds.variables == variables, (
            set(ds.variables) - set(variables),
            set(variables) - set(ds.variables),
            ds.variables,
        )


# --------------------------------------------------------------------------
# Base store
# --------------------------------------------------------------------------
def test_class_gridded_synthetic_store():
    ds = open_dataset(synthetic())
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape == (5, 3, 1, 9)
    assert ds.field_shape == (3, 3)


# --------------------------------------------------------------------------
# Subset (start / end / frequency / shuffle)
# --------------------------------------------------------------------------
# A daily dataset with distinct dates so date subsetting has room to work.
def _daily():
    return synthetic(start="2020-01-01", end="2020-01-10", frequency="1d")


SUBSET_CASES = [
    ({"start": "2020-01-03"}, 8),
    ({"end": "2020-01-04"}, 4),
    ({"start": "2020-01-03", "end": "2020-01-06"}, 4),
    ({"frequency": "2d"}, 5),
    ({"shuffle": True}, 10),
]


@pytest.mark.parametrize("kwargs,expected_len", SUBSET_CASES)
def test_class_gridded_subset(kwargs, expected_len):
    ds = open_dataset(_daily(), **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert len(ds) == expected_len


# --------------------------------------------------------------------------
# Select (select / drop / reorder)
# --------------------------------------------------------------------------
SELECT_CASES = [
    ({"select": ["b", "a"]}, {"a", "b"}),
    ({"select": {"a", "b"}}, {"a", "b"}),
    ({"select": ("a", "c")}, {"a", "c"}),
    ({"select": "a"}, {"a"}),
    ({"drop": ["b"]}, {"a", "c"}),
    ({"drop": "b"}, {"a", "c"}),
    ({"drop": {"b", "c"}}, {"a"}),
]


@pytest.mark.parametrize("kwargs,expected", SELECT_CASES)
def test_class_gridded_select(kwargs, expected):
    ds = open_dataset(synthetic(), **kwargs)
    assert set(ds.variables) == expected


REORDER_CASES = [
    ({"reorder": ["c", "a", "b"]}, ["c", "a", "b"]),
    ({"reorder": {"c": 0, "a": 1, "b": 2}}, ["c", "a", "b"]),
    ({"reorder": "sort"}, ["a", "b", "c"]),
]


@pytest.mark.parametrize("kwargs,expected", REORDER_CASES)
def test_class_gridded_select_reorder(kwargs, expected):
    ds = open_dataset(synthetic(variables=["c", "a", "b"]), **kwargs)
    _test_dataset(ds, variables=expected)


# --------------------------------------------------------------------------
# Rename
# --------------------------------------------------------------------------
RENAME_CASES = [
    ({"a": "temperature"}, ["temperature", "b", "c"]),
    ({"a": "temperature", "b": "pressure"}, ["temperature", "pressure", "c"]),
]


@pytest.mark.parametrize("rename,expected", RENAME_CASES)
def test_class_gridded_rename(rename, expected):
    ds = open_dataset(synthetic(), rename=rename)
    _test_dataset(ds, variables=expected)


# --------------------------------------------------------------------------
# Rescale
# --------------------------------------------------------------------------
def test_class_gridded_rescale_scale_offset_tuple():
    ds = open_dataset(synthetic(values={"constant": 273.15}), rescale={"a": (1.0, -273.15)})
    np.testing.assert_allclose(ds[0][0], 0.0, atol=1e-4)


def test_class_gridded_rescale_scale_offset_dict():
    ds = open_dataset(
        synthetic(values={"constant": 273.15}),
        rescale={"a": {"scale": 1.0, "offset": -273.15}},
    )
    np.testing.assert_allclose(ds[0][0], 0.0, atol=1e-4)


def test_class_gridded_rescale_units():
    try:
        import cfunits  # noqa: F401
    except (ImportError, FileNotFoundError) as e:
        # cfunits absent, or present but the UDUNITS-2 C library is missing.
        pytest.skip(str(e))
    ds = open_dataset(synthetic(values={"constant": 273.15}), rescale={"a": ("K", "degC")})
    np.testing.assert_allclose(ds[0][0], 0.0, atol=1e-4)


# --------------------------------------------------------------------------
# Statistics (statistics / statistics_tendencies / both)
# --------------------------------------------------------------------------
def test_class_gridded_statistics():
    ds = open_dataset(
        synthetic(values={"constant": 1.0}),
        statistics=synthetic(values={"constant": 5.0}),
    )
    np.testing.assert_array_equal(ds.statistics["mean"], 5.0)


def test_class_gridded_statistics_tendencies():
    ds = open_dataset(
        synthetic(values={"constant": 1.0}),
        statistics_tendencies=synthetic(values={"constant": 2.0}),
    )
    _test_dataset(ds, variables=["a", "b", "c"])


def test_class_gridded_statistics_both():
    ds = open_dataset(
        synthetic(values={"constant": 1.0}),
        statistics=synthetic(values={"constant": 5.0}),
        statistics_tendencies=synthetic(values={"constant": 2.0}),
    )
    np.testing.assert_array_equal(ds.statistics["mean"], 5.0)


# --------------------------------------------------------------------------
# Masking (mask from a .npy file)
# --------------------------------------------------------------------------
def test_class_gridded_masking(tmp_path):
    mask = np.zeros(9, dtype=bool)
    mask[:4] = True
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)

    ds = open_dataset(synthetic(), mask=str(mask_file))
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape[-1] == 4


# --------------------------------------------------------------------------
# Cropping (area as bbox list/tuple, or as another dataset)
# --------------------------------------------------------------------------
CROPPING_CASES = [
    {"area": [4.0, 0.0, 2.0, 2.0]},
    {"area": (4.0, 0.0, 2.0, 2.0)},
    {"area": synthetic(bbox=[4.0, 0.0, 2.0, 2.0], resolution=2.0)},
]


@pytest.mark.parametrize("kwargs", CROPPING_CASES)
def test_class_gridded_cropping(kwargs):
    ds = open_dataset(synthetic(), **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape[-1] < 9


# --------------------------------------------------------------------------
# TrimEdge (edge as int, or as list of 4)
# --------------------------------------------------------------------------
TRIM_CASES = [
    (1, (1, 1)),
    ([1, 1, 0, 0], (1, 3)),
]


@pytest.mark.parametrize("edge,expected_field_shape", TRIM_CASES)
def test_class_gridded_trim_edge(edge, expected_field_shape):
    ds = open_dataset(synthetic(), trim_edge=edge)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.field_shape == expected_field_shape


# --------------------------------------------------------------------------
# Thinning (every method)
# --------------------------------------------------------------------------
THINNING_CASES = [
    (None, 2),  # default method -> every-nth
    ("every-nth", 2),
    ("distance-based", 300.0),  # km
    ("grid", 300.0),  # km
    ("random", 0.6),  # fraction
]


@pytest.mark.parametrize("method,thinning", THINNING_CASES)
def test_class_gridded_thinning(method, thinning):
    # A 5x5 grid gives the distance/grid/random methods room to work.
    kwargs = {"thinning": thinning}
    if method is not None:
        kwargs["method"] = method
    ds = open_dataset(synthetic(bbox=[8.0, 0.0, 0.0, 8.0], resolution=2.0), **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert 1 <= ds.shape[-1] <= 25


# --------------------------------------------------------------------------
# Number (ensemble member selection: number / numbers / member / members)
# --------------------------------------------------------------------------
NUMBER_CASES = [
    ({"member": 0}, 1),
    ({"members": [0, 2]}, 2),
    ({"number": 1}, 1),
    ({"numbers": [1, 3]}, 2),
]


@pytest.mark.parametrize("kwargs,expected_members", NUMBER_CASES)
def test_class_gridded_number(kwargs, expected_members):
    ds = open_dataset(synthetic(ensembles=4), **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape[2] == expected_members


# --------------------------------------------------------------------------
# MissingDates (set_missing_dates: int index, date string, mixed list)
# --------------------------------------------------------------------------
MISSING_CASES = [
    ([2], {2}),
    (["2020-01-01T12:00:00"], {2}),
    ([0, "2020-01-01T12:00:00"], {0, 2}),
]


@pytest.mark.parametrize("set_missing_dates,expected_missing", MISSING_CASES)
def test_class_gridded_set_missing_dates(set_missing_dates, expected_missing):
    ds = open_dataset(synthetic(), set_missing_dates=set_missing_dates)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.missing == expected_missing


# --------------------------------------------------------------------------
# SkipMissingDates (expected_access as int or slice)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("expected_access", [2, slice(0, 2)])
def test_class_gridded_skip_missing_dates(expected_access):
    base = open_dataset(synthetic(), set_missing_dates=[2])
    ds = open_dataset(base, skip_missing_dates=True, expected_access=expected_access)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert len(ds) < len(base)


# --------------------------------------------------------------------------
# fill_missing_dates (closest up/down, interpolate)
# --------------------------------------------------------------------------
FILL_CASES = [
    {"fill_missing_dates": "closest"},
    {"fill_missing_dates": "closest", "closest": "up"},
    {"fill_missing_dates": "closest", "closest": "down"},
    {"fill_missing_dates": "interpolate"},
]


@pytest.mark.parametrize("kwargs", FILL_CASES)
def test_class_gridded_fill_missing_dates(kwargs):
    base = open_dataset(synthetic(), set_missing_dates=[2])
    ds = open_dataset(base, **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.missing == set()


# --------------------------------------------------------------------------
# Interpolate frequency
# --------------------------------------------------------------------------
INTERPOLATE_FREQ_CASES = [
    ("3h", 9),  # ratio 2 -> (5-1)*2 + 1
    ("2h", 13),  # ratio 3
    ("1h", 25),  # ratio 6
]


@pytest.mark.parametrize("frequency,expected_len", INTERPOLATE_FREQ_CASES)
def test_class_gridded_interpolate_frequency(frequency, expected_len):
    ds = open_dataset(synthetic(), interpolate_frequency=frequency)
    _test_dataset(ds, variables=["a", "b", "c"])
    assert len(ds) == expected_len


# --------------------------------------------------------------------------
# Interpolate nearest (with and without max_distance)
# --------------------------------------------------------------------------
INTERPOLATE_NEAREST_CASES = [
    {"interpolate_variables": ["a"]},
    {"interpolate_variables": ["a", "b"], "max_distance": 1000000.0},
]


@pytest.mark.parametrize("kwargs", INTERPOLATE_NEAREST_CASES)
def test_class_gridded_interpolate_nearest(kwargs):
    ds = open_dataset(synthetic(), **kwargs)
    _test_dataset(ds, variables=["a", "b", "c"])


# --------------------------------------------------------------------------
# RollingAverage (freq / frequency, centred and one-sided windows)
# --------------------------------------------------------------------------
ROLLING_CASES = [
    (-1, 1, "freq"),
    (0, 2, "freq"),
    (-2, 0, "frequency"),
]


@pytest.mark.parametrize("window", ROLLING_CASES)
def test_class_gridded_rolling_average(window):
    ds = open_dataset(synthetic(), rolling_average=window)
    _test_dataset(ds, variables=["a", "b", "c"])


# --------------------------------------------------------------------------
# Join (join=, plain list, single dataset, disjoint variables)
# --------------------------------------------------------------------------
def test_class_gridded_join():
    ds = open_dataset(join=[synthetic(variables=["a", "b"]), synthetic(variables=["c", "d"])])
    _test_dataset(ds, variables=["a", "b", "c", "d"])


def test_class_gridded_join_via_list():
    ds = open_dataset([synthetic(variables=["a", "b"]), synthetic(variables=["c", "d"])])
    _test_dataset(ds, variables=["a", "b", "c", "d"])


def test_class_gridded_join_single():
    ds = open_dataset(join=[synthetic(variables=["a", "b"])])
    _test_dataset(ds, variables=["a", "b"])


# --------------------------------------------------------------------------
# Concat (concat=, plain list, with fill_missing_gaps)
# --------------------------------------------------------------------------
def _concat_parts(start1="2020-01-01", end1="2020-01-05", start2="2020-01-06", end2="2020-01-10"):
    return [
        synthetic(variables=["a", "b"], start=start1, end=end1, frequency="1d"),
        synthetic(variables=["a", "b"], start=start2, end=end2, frequency="1d"),
    ]


def test_class_gridded_concat():
    ds = open_dataset(concat=_concat_parts())
    _test_dataset(ds, variables=["a", "b"])
    assert len(ds) == 10


def test_class_gridded_concat_via_list():
    ds = open_dataset(_concat_parts())
    _test_dataset(ds, variables=["a", "b"])
    assert len(ds) == 10


def test_class_gridded_missing_dataset():
    # A gap between the two ranges, filled with a MissingDataset view.
    ds = open_dataset(
        concat=_concat_parts(end1="2020-01-03", start2="2020-01-06", end2="2020-01-08"),
        fill_missing_gaps=True,
    )
    _test_dataset(ds, variables=["a", "b"])
    assert len(ds) == 8  # 01..08 inclusive
    assert ds.missing  # the gap days are missing


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------
def test_class_gridded_ensemble():
    ds = open_dataset(ensemble=[synthetic(), synthetic()])
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape[2] == 2


# --------------------------------------------------------------------------
# Grids (with and without adjust)
# --------------------------------------------------------------------------
def test_class_gridded_grids():
    ds = open_dataset(
        grids=[
            synthetic(bbox=[4.0, 0.0, 0.0, 4.0], resolution=2.0),
            synthetic(bbox=[10.0, 10.0, 6.0, 14.0], resolution=2.0),
        ]
    )
    _test_dataset(ds, variables=["a", "b", "c"])
    assert ds.shape[-1] == 18  # 9 + 9


def test_class_gridded_grids_adjust():
    ds = open_dataset(
        grids=[
            synthetic(variables=["a", "b", "c"], bbox=[4.0, 0.0, 0.0, 4.0], resolution=2.0),
            synthetic(variables=["a", "b"], bbox=[10.0, 10.0, 6.0, 14.0], resolution=2.0),
        ],
        adjust="all",
    )
    _test_dataset(ds, variables=["a", "b"])


# --------------------------------------------------------------------------
# Cutout (default, and with extra parameters)
# --------------------------------------------------------------------------
def _cutout_parts():
    return [
        synthetic(bbox=[60.0, -10.0, 40.0, 10.0], resolution=2.0),  # LAM
        synthetic(bbox=[80.0, -40.0, 20.0, 40.0], resolution=10.0),  # global
    ]


def test_class_gridded_cutout():
    ds = open_dataset(cutout=_cutout_parts())
    _test_dataset(ds, variables=["a", "b", "c"])


def test_class_gridded_cutout_with_params():
    ds = open_dataset(cutout=_cutout_parts(), cropping_distance=3.0, neighbours=3, min_distance_km=0.0)
    _test_dataset(ds, variables=["a", "b", "c"])


# --------------------------------------------------------------------------
# Complement (none, nearest, nearest with k / max_distance)
# --------------------------------------------------------------------------
def test_class_gridded_complement_none():
    ds = open_dataset(
        complement=synthetic(variables=["a", "b"]),
        source=synthetic(variables=["a", "b", "c"]),
    )
    _test_dataset(ds, variables=["a", "b", "c"])


COMPLEMENT_NEAREST_CASES = [
    {"interpolation": "nearest"},
    {"interpolation": "nearest", "k": 2},
    {"interpolation": "nearest", "max_distance": 1000000.0},
]


@pytest.mark.parametrize("kwargs", COMPLEMENT_NEAREST_CASES)
def test_class_gridded_complement_nearest(kwargs):
    ds = open_dataset(
        complement=synthetic(variables=["a", "b"], bbox=[4.0, 0.0, 0.0, 4.0], resolution=2.0),
        source=synthetic(variables=["a", "b", "c"], bbox=[4.0, 0.0, 0.0, 4.0], resolution=1.0),
        **kwargs,
    )
    _test_dataset(ds, variables=["a", "b", "c"])


# --------------------------------------------------------------------------
# Merge (interleaved dates; with and without allow_gaps_in_dates)
# --------------------------------------------------------------------------
def _merge_parts():
    return [
        synthetic(
            variables=["a", "b"],
            start="2020-01-01T00",
            end="2020-01-02T00",
            frequency="12h",
        ),
        synthetic(
            variables=["a", "b"],
            start="2020-01-01T06",
            end="2020-01-01T18",
            frequency="12h",
        ),
    ]


def test_class_gridded_merge():
    ds = open_dataset(merge=_merge_parts())
    _test_dataset(ds, variables=["a", "b"])
    assert len(ds) == 5  # 00, 06, 12, 18, 00


def test_class_gridded_merge_allow_gaps():
    # union of dates is 00, 06, 18 -> inferred frequency 6h -> 12 is a gap
    ds = open_dataset(
        merge=[
            synthetic(
                variables=["a", "b"],
                start="2020-01-01T00",
                end="2020-01-01T06",
                frequency="6h",
            ),
            synthetic(
                variables=["a", "b"],
                start="2020-01-01T18",
                end="2020-01-01T18",
                frequency="6h",
            ),
        ],
        allow_gaps_in_dates=True,
    )
    _test_dataset(ds, variables=["a", "b"])
    assert ds.missing  # 12 is a gap


# --------------------------------------------------------------------------
# Chain (unchecked concat)
# --------------------------------------------------------------------------
def test_class_gridded_chain():
    ds = open_dataset(chain=_concat_parts())
    _test_dataset(ds)
    assert len(ds) == 10


# --------------------------------------------------------------------------
# Zip / XY (experimental; with and without compatibility checks)
# --------------------------------------------------------------------------
def test_class_gridded_zip():
    ds = open_dataset(zip=[synthetic(), synthetic()])
    assert len(ds) == 5
    assert isinstance(ds[0], tuple)


def test_class_gridded_zip_no_check():
    ds = open_dataset(zip=[synthetic(), synthetic()], check_compatibility=False)
    assert len(ds) == 5


def test_class_gridded_xy():
    ds = open_dataset(xy=[synthetic(), synthetic()])
    assert len(ds) == 5
    assert isinstance(ds[0], tuple)


def test_class_gridded_x_y():
    ds = open_dataset(x=synthetic(), y=synthetic())
    assert len(ds) == 5
    assert isinstance(ds[0], tuple)


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
