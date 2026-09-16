# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``clip:`` option of the accumulate source.

The option has the same configuration shape as the ``clip`` filter of
anemoi-transform, and delegates the arithmetic to it — so the unit tests
pin the *normalisation* (which spellings are accepted) and the *selection*
(which fields a spec applies to), and the end-to-end test pins that a
recipe carrying ``clip:`` builds a dataset whose accumulated values are
bounded.

The end-to-end part needs no network: the archive is written from an
eccodes sample, with 10-minute increments holding a *negative* value so
that the accumulation is negative and clipping is visible.
"""

import datetime
import os

import numpy as np
import pytest
from anemoi.transform import Field
from anemoi.transform import FieldList
from pydantic import ValidationError

from anemoi.datasets import open_dataset
from anemoi.datasets.create.sources.accumulate.clip import ClipSpec
from anemoi.datasets.create.sources.accumulate.clip import apply_clip
from anemoi.datasets.create.sources.accumulate.clip import normalise_clip
from anemoi.datasets.create.sources.accumulate.description import AccumulateSchema

from .utils.create import create_dataset

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_normalise_clip_absent():
    assert normalise_clip(None) is None
    assert normalise_clip(False) is None
    assert normalise_clip([]) is None


def test_normalise_clip_single_mapping():
    assert normalise_clip({"param": "tp", "minimum": 0}) == [ClipSpec(param="tp", minimum=0)]


def test_normalise_clip_list_of_mappings():
    specs = normalise_clip([{"param": "tp", "minimum": 0}, {"param": "cp", "maximum": 1}])
    assert specs == [ClipSpec(param="tp", minimum=0), ClipSpec(param="cp", maximum=1)]


def test_normalise_clip_param_is_optional():
    assert normalise_clip({"minimum": 0}) == [ClipSpec(minimum=0)]


def test_normalise_clip_accepts_already_built_specs():
    spec = ClipSpec(param="tp", minimum=0)
    assert normalise_clip(spec) == [spec]
    assert normalise_clip([spec]) == [spec]


def test_normalise_clip_rejects_true():
    # No bound is inferred, exactly as in the filter.
    with pytest.raises(ValueError, match="not a value"):
        normalise_clip(True)


def test_normalise_clip_rejects_bare_parameter_list():
    with pytest.raises(ValueError, match="must be a mapping"):
        normalise_clip(["tp", "cp"])


def test_normalise_clip_requires_a_bound():
    with pytest.raises(ValidationError, match="minimum"):
        normalise_clip({"param": "tp"})


def test_normalise_clip_rejects_unknown_key():
    with pytest.raises(ValidationError):
        normalise_clip({"param": "tp", "min": 0})


def test_normalise_clip_rejects_reversed_bounds():
    with pytest.raises(ValidationError, match="greater than maximum"):
        normalise_clip({"param": "tp", "minimum": 3, "maximum": 1})


def test_normalise_clip_rejects_duplicate_param():
    with pytest.raises(ValueError, match="more than one entry"):
        normalise_clip([{"param": "tp", "minimum": 0}, {"param": "tp", "maximum": 1}])


# ---------------------------------------------------------------------------
# recipe validation
# ---------------------------------------------------------------------------


def _schema(**kwargs):
    return AccumulateSchema(period="6h", source={"mars": {"param": ["tp"]}}, **kwargs)


def test_schema_default_is_no_clipping():
    assert _schema().clip is None
    assert _schema(clip=False).clip is None


def test_schema_normalises_clip():
    assert _schema(clip={"param": "tp", "minimum": 0}).clip == [ClipSpec(param="tp", minimum=0)]


def test_schema_clip_survives_a_dump_and_rebuild():
    schema = _schema(clip={"param": "tp", "minimum": 0})
    rebuilt = AccumulateSchema(**schema.model_dump(mode="json", by_alias=True))
    assert rebuilt.clip == schema.clip


def test_schema_rejects_a_clip_without_bounds():
    with pytest.raises(ValidationError, match="minimum"):
        _schema(clip={"param": "tp"})


# ---------------------------------------------------------------------------
# selection and arithmetic (delegated to the transform filter)
# ---------------------------------------------------------------------------


def _field(param, values):
    return Field.from_components(
        values=np.asarray(values, dtype=float),
        parameter={"variable": param},
        time={"valid_datetime": datetime.datetime(2021, 1, 1)},
        geography={"latitudes": np.arange(3.0), "longitudes": np.arange(3.0)},
    )


def _fields():
    return FieldList.from_fields([_field("tp", [-1, 0, 2]), _field("cp", [-3, 1, 5])])


def _values(fields):
    return {f.get("parameter.variable"): f.to_numpy().tolist() for f in fields}


def test_apply_clip_without_specs_is_a_no_op():
    fields = _fields()
    assert apply_clip(fields, None) is fields


def test_apply_clip_only_touches_the_named_param():
    clipped = apply_clip(_fields(), normalise_clip({"param": "tp", "minimum": 0}))
    assert _values(clipped) == {"tp": [0, 0, 2], "cp": [-3, 1, 5]}


def test_apply_clip_without_param_touches_every_param():
    clipped = apply_clip(_fields(), normalise_clip({"minimum": 0}))
    assert _values(clipped) == {"tp": [0, 0, 2], "cp": [0, 1, 5]}


def test_apply_clip_honours_both_bounds():
    clipped = apply_clip(_fields(), normalise_clip([{"param": "tp", "minimum": 0}, {"param": "cp", "maximum": 4}]))
    assert _values(clipped) == {"tp": [0, 0, 2], "cp": [-3, 1, 4]}


def test_apply_clip_rejects_a_param_that_was_not_accumulated():
    # The filter alone would silently pass every field through.
    with pytest.raises(ValueError, match="not among the accumulated parameters"):
        apply_clip(_fields(), normalise_clip({"param": "typo", "minimum": 0}))


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

eccodes = pytest.importorskip("eccodes")

BASE_TIME = datetime.datetime(2021, 1, 1, 0)
NI, NJ = 12, 6
NPOINTS = NI * NJ

# Every 10-minute increment holds this value, so each 30-minute accumulation
# is three times it — negative, which is what clipping is for.
INCREMENT = -1.0
ACCUMULATED = 3 * INCREMENT


def _sample_handle():
    """A regular lat/lon GRIB2 message, resized to a small test grid."""
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set(h, "Ni", NI)
    eccodes.codes_set(h, "Nj", NJ)
    eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", 60)
    eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", -60)
    eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0)
    eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", 330)
    eccodes.codes_set(h, "iDirectionIncrementInDegrees", 30)
    eccodes.codes_set(h, "jDirectionIncrementInDegrees", 24)
    return h


def _write_accumulation(path, base, start_minutes, end_minutes, value, param_id=228228):
    """An accumulated field of the *base* run covering ``[start, end]`` minutes."""
    end = base + datetime.timedelta(minutes=end_minutes)
    h = _sample_handle()
    try:
        eccodes.codes_set(h, "productDefinitionTemplateNumber", 8)
        eccodes.codes_set(h, "paramId", param_id)
        eccodes.codes_set(h, "typeOfStatisticalProcessing", 1)  # accumulation
        eccodes.codes_set(h, "dataDate", int(base.strftime("%Y%m%d")))
        eccodes.codes_set(h, "dataTime", int(base.strftime("%H%M")))
        eccodes.codes_set(h, "indicatorOfUnitOfTimeRange", 0)  # minutes
        eccodes.codes_set(h, "forecastTime", start_minutes)
        eccodes.codes_set(h, "indicatorOfUnitForTimeRange", 0)  # minutes
        eccodes.codes_set(h, "lengthOfTimeRange", end_minutes - start_minutes)
        for key, part in (
            ("year", end.year),
            ("month", end.month),
            ("day", end.day),
            ("hour", end.hour),
            ("minute", end.minute),
            ("second", 0),
        ):
            eccodes.codes_set(h, f"{key}OfEndOfOverallTimeInterval", part)
        eccodes.codes_set_values(h, np.full(NPOINTS, float(value)))
        with open(path, "wb") as f:
            eccodes.codes_write(h, f)
    finally:
        eccodes.codes_release(h)


def _archive(directory):
    """A base-less 10-minute increment archive, addressed by the end of its window."""
    os.makedirs(directory, exist_ok=True)
    times = [BASE_TIME + datetime.timedelta(minutes=10 * i) for i in range(1, 19)]  # 00:10..03:00
    for t in times:
        _write_accumulation(
            os.path.join(directory, f"inc{t:%Y%m%d%H%M}.grib"),
            t - datetime.timedelta(minutes=10),
            0,
            10,
            INCREMENT,
        )


def _recipe(directory, clip):
    recipe = {
        "dates": {"start": "2021-01-01 00:30:00", "end": "2021-01-01 03:00:00", "frequency": "30m"},
        "input": {
            "accumulate": {
                "period": "30m",
                "from": {"accumulation": "10m"},
                "source": {
                    "grib": {
                        "path": os.path.join(directory, "inc{end_date:strftime(%Y%m%d%H%M)}.grib"),
                        "param": ["tp"],
                    }
                },
            }
        },
        "build": {"group_by": 1},
        "statistics": {"end": 2021},
    }
    if clip is not None:
        recipe["input"]["accumulate"]["clip"] = clip
    return recipe


@pytest.mark.parametrize(
    "clip,expected",
    [
        (None, ACCUMULATED),
        ({"param": "tp", "minimum": 0}, 0.0),
        # The bound is the one given, not a hardcoded zero.
        ({"param": "tp", "minimum": -2}, -2.0),
        # `param` omitted — every accumulated parameter.
        ({"minimum": 0}, 0.0),
    ],
)
def test_build_with_clip(tmp_path, clip, expected):
    """A recipe carrying ``clip:`` builds, and the accumulated values are bounded."""
    tmp_path = str(tmp_path)
    directory = os.path.join(tmp_path, "inc")
    _archive(directory)

    path = os.path.join(tmp_path, "clip.zarr")
    create_dataset(recipe=_recipe(directory, clip), output=path)
    ds = open_dataset(path)

    assert ds.variables == ["tp"]
    assert np.allclose(ds[:], expected), (ds[:].min(), ds[:].max())
