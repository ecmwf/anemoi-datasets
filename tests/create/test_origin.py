# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anemoi.transform import Field
from anemoi.transform import FieldList

from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.input.origin import Filter
from anemoi.datasets.create.input.origin import Join
from anemoi.datasets.create.input.origin import Pipe
from anemoi.datasets.create.input.origin import Source
from anemoi.datasets.create.input.origin import _un_dotdict


def _field(param: str, **labels) -> Field:
    components = dict(
        values=np.zeros(4),
        parameter={"variable": param},
        time={"valid_datetime": datetime.datetime(2020, 1, 1)},
        geography={"latitudes": np.arange(4.0), "longitudes": np.arange(4.0)},
    )
    if labels:
        components["labels"] = labels
    return Field.from_components(**components)


class _StubAction:
    """A source/filter action reduced to what ``Context.origin`` needs."""

    def __init__(self, origin):
        self._origin = origin

    def origin(self):
        return self._origin


class _TestContext(Context):
    def create_result(self, argument: Any, data: Any) -> Any:
        return data


def _context() -> Context:
    recipe = SimpleNamespace(build=SimpleNamespace(use_grib_paramid=False, variable_naming="default"))
    return _TestContext(recipe)


# ---------------------------------------------------------------------------
# Origin classes
# ---------------------------------------------------------------------------


def test_origins_compare_by_identity():
    # Two sources with identical configurations are distinct origins:
    # "same origin" means "produced by the same action instance".
    a = Source("mars", {"param": "2t"})
    b = Source("mars", {"param": "2t"})
    assert a == a
    assert a != b
    assert len({a, b}) == 2


def test_source_combine_returns_self():
    source = Source("mars", {"param": "2t"})
    assert source.combine(None, None, None) is source


def test_source_combine_overrides_inherited_origin():
    # Fields built from a template (e.g. the forcings source) inherit the
    # template's origin; the producing source wins.
    template_origin = Source("mars", {"param": "2t"})
    forcings = Source("forcings", {"template": "${input.join.0.mars}"})
    assert forcings.combine(template_origin, None, None) is forcings


def test_filter_combine_builds_cached_pipe():
    source = Source("mars", {"param": "2t"})
    filter_ = Filter("rename", {"param": {"tp": "tp_6h"}})

    pipe = filter_.combine(source, None, None)
    assert isinstance(pipe, Pipe)
    assert pipe.steps == [source, filter_]

    # The pipe is cached per upstream origin, so all the fields flowing
    # through the filter share one origin object (identity grouping).
    assert filter_.combine(source, None, None) is pipe


def test_pipes_stay_flat():
    source = Source("mars", {})
    f1 = Filter("f1", {})
    f2 = Filter("f2", {})

    pipe = f2.combine(f1.combine(source, None, None), None, None)
    assert pipe.steps == [source, f1, f2]
    assert not any(isinstance(s, Pipe) for s in pipe.steps)


def test_filter_combine_recovers_origin_from_field_arguments():
    # A filter that returns fresh objects (dropping the origin tag) recovers
    # the origin from its input fields.
    source = Source("mars", {})
    filter_ = Filter("plugin", {})
    action = _StubAction(filter_)
    arguments = FieldList.from_fields([_field("2t", anemoi_origin=source)])

    pipe = filter_.combine(None, action, arguments)
    assert isinstance(pipe, Pipe)
    assert pipe.steps == [source, filter_]


def test_filter_combine_joins_mixed_field_arguments():
    s1 = Source("mars", {})
    s2 = Source("grib", {})
    filter_ = Filter("plugin", {})
    action = _StubAction(filter_)
    arguments = FieldList.from_fields([_field("2t", anemoi_origin=s1), _field("msl", anemoi_origin=s2)])

    pipe = filter_.combine(None, action, arguments)
    assert isinstance(pipe.steps[0], Join)
    assert set(pipe.steps[0].steps) == {s1, s2}


def test_filter_combine_recovers_origin_from_frame_arguments():
    source = Source("csv", {"path": "x.csv"})
    filter_ = Filter("plugin", {})
    action = _StubAction(filter_)
    frame = pd.DataFrame({"a": [1.0]})
    frame.attrs["anemoi_origin"] = source

    pipe = filter_.combine(None, action, frame)
    assert pipe.steps == [source, filter_]


def test_as_dict_is_json_serialisable():
    # Configs may contain dates, timedeltas, sets... and origins end up in
    # the zarr attributes, which only accept JSON.
    source = Source(
        "mars",
        {
            "date": datetime.date(2020, 12, 30),
            "base": datetime.datetime(2020, 12, 30, 12),
            "period": datetime.timedelta(hours=6),
            "params": ("2t", "msl"),
        },
    )
    pipe = Filter("rename", {}).combine(source, None, None)
    joined = Join([pipe, Source("grib", {})])

    serialised = json.loads(json.dumps(joined.as_dict()))
    assert serialised["type"] == "join"
    assert serialised["steps"][0]["type"] == "pipe"
    assert serialised["steps"][0]["steps"][0]["config"]["date"] == "2020-12-30"
    assert serialised["steps"][0]["steps"][0]["config"]["period"] == "6:00:00"


def test_un_dotdict_converts_nested_structures():
    converted = _un_dotdict({"a": [datetime.date(2020, 1, 1)], "b": {"c": (1, 2)}})
    assert converted == {"a": ["2020-01-01"], "b": {"c": [1, 2]}}


# ---------------------------------------------------------------------------
# Context.origin: tagging the actions' results
# ---------------------------------------------------------------------------


def test_context_origin_tags_fields():
    context = _context()
    source = Source("mars", {"param": "2t"})
    data = FieldList.from_fields([_field("2t"), _field("msl")])

    tagged = context.origin(data, _StubAction(source), None)
    for field in tagged:
        assert field.get("labels.anemoi_origin") is source


def test_context_origin_pipes_filters():
    context = _context()
    source = Source("mars", {})
    filter_ = Filter("rename", {})

    tagged = context.origin(FieldList.from_fields([_field("2t")]), _StubAction(source), None)
    tagged = context.origin(tagged, _StubAction(filter_), tagged)

    origin = tagged[0].get("labels.anemoi_origin")
    assert isinstance(origin, Pipe)
    assert origin.steps == [source, filter_]


def test_context_origin_respects_fall_through():
    # A field marked as passed-through keeps its origin untouched.
    context = _context()
    source = Source("mars", {})
    field = _field("2t", anemoi_origin=source, anemoi_fall_through=True)

    tagged = context.origin(FieldList.from_fields([field]), _StubAction(Filter("noop", {})), None)
    assert tagged[0].get("labels.anemoi_origin") is source


def test_context_origin_tags_frames():
    context = _context()
    source = Source("csv", {"path": "x.csv"})
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    tagged = context.origin(frame, _StubAction(source), None)
    assert tagged.attrs["anemoi_origin"] is source


def test_context_join_combines_frame_origins():
    context = _context()
    s1, s2 = Source("csv", {}), Source("odb", {})
    f1 = pd.DataFrame({"a": [1.0]})
    f1.attrs["anemoi_origin"] = s1
    f2 = pd.DataFrame({"a": [2.0]})
    f2.attrs["anemoi_origin"] = s2

    joined = context.join([f1, f2])
    origin = joined.attrs["anemoi_origin"]
    assert isinstance(origin, Join)
    assert set(origin.steps) == {s1, s2}

    # A single shared origin is kept as-is.
    f3 = pd.DataFrame({"a": [3.0]})
    f3.attrs["anemoi_origin"] = s1
    assert context.join([f1, f3]).attrs["anemoi_origin"] is s1


# ---------------------------------------------------------------------------
# TabularResult: origins of a frame
# ---------------------------------------------------------------------------


def _tabular_result(frame: pd.DataFrame):
    from anemoi.datasets.create.tabular.result import TabularResult

    argument = SimpleNamespace(
        start_range=datetime.datetime(2020, 1, 1),
        end_range=datetime.datetime(2020, 12, 31),
    )
    return TabularResult(context=None, argument=argument, frame=frame)


def _obs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01T00:00", "2020-06-01T06:00"]).astype("datetime64[s]"),
            "latitude": [10.0, 20.0],
            "longitude": [30.0, 40.0],
            "obs": [1.5, 2.5],
        }
    )


def test_tabular_result_origins():
    frame = _obs_frame()
    source = Source("csv", {"path": "x.csv"})
    frame.attrs["anemoi_origin"] = source

    origins = _tabular_result(frame).origins
    assert origins["version"] == 1
    (entry,) = origins["origins"]
    assert entry["origin"]["name"] == "csv"
    assert "obs" in entry["variables"]


def test_tabular_result_origins_missing():
    with pytest.raises(ValueError, match="no origin"):
        _tabular_result(_obs_frame()).origins
