# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from contextlib import contextmanager
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest

from anemoi.datasets.create.sources.bufr_support.bufr_reader import BUFRMessage
from anemoi.datasets.create.sources.bufr_support.bufr_reader import BUFRReader
from anemoi.datasets.create.sources.bufr_support.bufr_to_df import BUFRMessageSelector
from anemoi.datasets.create.sources.bufr_support.bufr_to_df import BUFRToDataFrame
from anemoi.datasets.create.sources.bufr_support.bufr_to_df import parse_slice


def create_bufr_message(
    *,
    lat: list[float],
    lon: list[float],
    datetime_spec: dict,
    datetime_prefix: str = "",
    extra_values: dict | None = None,
    per_report_arrays: dict | None = None,
    per_datum_arrays: dict | None = None,
) -> BUFRMessage:
    """Create a mocked BUFRMessage.

    ``nreports`` is derived from the length of the coordinate arrays.
    All per-report arrays (latitudes, longitudes, and any in
    ``per_report_arrays``) must have the same length.

    Parameters
    ----------
    lat, lon:
        Per-report coordinate arrays (required, must be the same length).
    datetime_spec:
        Datetime components (year, month, day, hour, minute, second).
        Required – every caller must supply an explicit datetime.
    datetime_prefix:
        Prefix prepended to datetime key names.
    extra_values:
        Additional scalar header values (merged with numberOfSubsets).
    per_report_arrays:
        Additional per-report arrays.  Must match the length of the
        coordinate arrays.
    per_datum_arrays:
        Additional per-datum arrays whose length may differ from
        ``nreports`` (e.g. multi-level observations).

    Returns
    -------
    BUFRMessage
        A mocked BUFRMessage instance.
    """
    arrays = {
        "lat": np.asarray(lat, dtype=float),
        "lon": np.asarray(lon, dtype=float),
        **(per_report_arrays or {}),
    }

    lengths = {name: len(arr) for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"All arrays must have the same length, got: {lengths}")

    nreports = len(arrays["lat"])

    # build datetimes
    datetime_arrays = {f"{datetime_prefix}{k}": np.full(nreports, v, dtype=int) for k, v in datetime_spec.items()}
    arrays.update(datetime_arrays)
    arrays.update(per_datum_arrays or {})

    values = {"numberOfSubsets": nreports}
    values.update(extra_values or {})

    msg = create_autospec(BUFRMessage, instance=True)

    def _get_value(key: str):
        if key in values:
            return values[key]
        from gribapi.errors import KeyValueNotFoundError

        raise KeyValueNotFoundError(key)

    per_datum_keys = set(per_datum_arrays or {})

    def _get_array(element: str, typ: type, nsubsets: int, missing_val=np.nan) -> np.ndarray:
        if element in arrays:
            arr = np.asarray(arrays[element], dtype=typ)
            if len(arr) == 1:
                arr = np.ones(nsubsets, dtype=typ) * arr
            if element in per_datum_keys:
                return arr
            return arr[:nsubsets]
        return np.ones(nsubsets, dtype=typ) * missing_val

    def _get_size(key: str):
        if key in arrays:
            return len(arrays[key])
        raise KeyError(key)

    msg.get_value.side_effect = _get_value
    msg.set_value.side_effect = lambda key, value: None
    msg.get_array.side_effect = _get_array
    msg.get_size.side_effect = _get_size
    return msg


def create_bufr_reader(messages):
    """Create a mocked BUFRReader that yields given messages."""
    reader = create_autospec(BUFRReader, instance=True)
    reader.__len__.return_value = len(messages)

    @contextmanager
    def _get_message(idx: int):
        yield messages[idx]

    reader.get_message.side_effect = _get_message
    return reader


@pytest.mark.parametrize(
    "op, msg_value, select_value, should_match",
    [
        ("==", 2, 2, True),
        ("==", 2, 3, False),
        ("!=", 0, 1, True),
        ("!=", 0, 0, False),
        (">", 2, 3, False),
        (">", 3, 2, True),
        (">=", 2, 2, True),
        ("<", 3, 2, False),
        ("<", 2, 3, True),
        ("<=", 2, 2, True),
    ],
)
def test_select_comparison_operators(op, msg_value, select_value, should_match):
    default_config = {
        "lat": [0.0],
        "lon": [0.0],
        "datetime_spec": {"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
    }
    msg = create_bufr_message(**default_config, extra_values={"key": msg_value})
    selector = BUFRMessageSelector("key", [op, select_value])
    assert selector.matches(msg) is should_match


def test_select_in_operator():
    default_config = {
        "lat": [0.0],
        "lon": [0.0],
        "datetime_spec": {"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
    }
    good_msg = create_bufr_message(**default_config, extra_values={"dataCategory": 2})
    bad_msg = create_bufr_message(**default_config, extra_values={"dataCategory": 5})
    selector = BUFRMessageSelector("dataCategory", ["in", [1, 2, 3]])
    assert selector.matches(good_msg)
    assert not selector.matches(bad_msg)


def test_select_not_in_operator():
    default_config = {
        "lat": [0.0],
        "lon": [0.0],
        "datetime_spec": {"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
    }
    good_msg = create_bufr_message(**default_config, extra_values={"dataCategory": 5})
    bad_msg = create_bufr_message(**default_config, extra_values={"dataCategory": 2})
    selector = BUFRMessageSelector("dataCategory", ["not in", [1, 2, 3]])
    assert selector.matches(good_msg)
    assert not selector.matches(bad_msg)


def test_select_inf_string_converted():
    selector = BUFRMessageSelector("value", ["<", "inf"])
    assert selector.expected == float("inf")


@pytest.mark.parametrize(
    "condition, match_msg",
    [
        ("bad_input", "expected a list of length 2"),
        (["==", 1, "extra"], "expected a list of length 2"),
        ([42, 1], "operator must be a string"),
        (["in", "not_a_list"], "requires a list value"),
        (["??", 1], "Invalid operator"),
    ],
)
def test_select_invalid_conditions(condition, match_msg):
    with pytest.raises(ValueError, match=match_msg):
        BUFRMessageSelector("key", condition)


def test_create_bufr_to_dataframe_minimal_config():
    _ = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
    )


def test_create_bufr_to_dataframe_invalid_per_datum_format():
    with pytest.raises(ValueError, match="Invalid per_datum_format"):
        BUFRToDataFrame(
            latitude="lat",
            longitude="lon",
            per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
            per_datum_format="bad_value",  # should be "wide" or "long"
        )


def test_per_report_rejects_reserved_output_names():
    with pytest.raises(ValueError, match="reserved coordinate column names"):
        BUFRToDataFrame(
            latitude="lat",
            longitude="lon",
            per_report={"latitude": "someOtherLatKey"},
        )


def test_per_report_basic():
    msg = create_bufr_message(
        lat=[10.0, 20.0, 30.0],
        lon=[100.0, 110.0, 120.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        per_report_arrays={"#1#brightnessTemperature": np.array([250.0, 260.0, 270.0])},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
    )

    df = converter.read_msg_to_df(msg)
    expected = pd.DataFrame(
        {
            "latitude": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "longitude": np.array([100.0, 110.0, 120.0], dtype=np.float32),
            "obsvalue_rawbt_1": np.array([250.0, 260.0, 270.0], dtype=np.float32),
            "datetime": pd.array([pd.Timestamp("2023-06-15T12:00:00")] * 3),
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_per_report_with_datetime_prefix():
    msg = create_bufr_message(
        lat=[1.0, 2.0],
        lon=[3.0, 4.0],
        datetime_prefix="pre_",
        datetime_spec=dict(year=2020, month=1, day=5, hour=6, minute=30, second=0),
        per_report_arrays={"#1#brightnessTemperature": np.array([255.0, 265.0])},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
        datetime_position_prefix="pre_",
    )

    df = converter.read_msg_to_df(msg)
    expected = pd.DataFrame(
        {
            "latitude": np.array([1.0, 2.0], dtype=np.float32),
            "longitude": np.array([3.0, 4.0], dtype=np.float32),
            "obsvalue_rawbt_1": np.array([255.0, 265.0], dtype=np.float32),
            "datetime": pd.array([pd.Timestamp("2020-01-05T06:30:00")] * 2),
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_header_preselect_removes_non_matching():
    msg = create_bufr_message(
        lat=[0.0],
        lon=[0.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        extra_values={"dataCategory": 5},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
        preselect_msg_header={"dataCategory": ["==", 2]},
    )

    assert converter.read_msg_to_df(msg).empty


def test_header_preselect_keeps_matching():
    msg = create_bufr_message(
        lat=[5.0],
        lon=[50.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        extra_values={"dataCategory": 2},
        per_report_arrays={"#1#brightnessTemperature": np.array([280.0])},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
        preselect_msg_header={"dataCategory": ["==", 2]},
    )

    df = converter.read_msg_to_df(msg)
    expected = pd.DataFrame(
        {
            "latitude": np.array([5.0], dtype=np.float32),
            "longitude": np.array([50.0], dtype=np.float32),
            "obsvalue_rawbt_1": np.array([280.0], dtype=np.float32),
            "datetime": pd.array([pd.Timestamp("2023-06-15T12:00:00")]),
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_data_preselect_removes_non_matching():
    msg = create_bufr_message(
        lat=[0.0],
        lon=[0.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        extra_values={"someFlag": 99},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
        preselect_msg_data={"someFlag": ["==", 0]},
    )

    assert converter.read_msg_to_df(msg).empty


def test_missing_select_key_removes_message():
    import eccodes

    msg = create_autospec(BUFRMessage, instance=True)
    msg.get_value.side_effect = eccodes.KeyValueNotFoundError

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
        preselect_msg_header={"missingKey": ["==", 1]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert converter.read_msg_to_df(msg).empty


def test_exception_in_message_returns_empty():
    msg = create_autospec(BUFRMessage, instance=True)
    msg.get_value.side_effect = RuntimeError("A bad thing happened")

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert converter.read_msg_to_df(msg).empty


def test_wide_format():
    nreports, ndatum = 2, 3
    temp_values = np.arange(nreports * ndatum, dtype=float)

    msg = create_bufr_message(
        lat=[10.0, 20.0],
        lon=[100.0, 110.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        per_report_arrays={"#1#brightnessTemperature": np.array([300.0, 310.0])},
        per_datum_arrays={"airTemperature": temp_values},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
        per_datum={"airTemperature": {"temp": "0:2"}},
        per_datum_format="wide",
    )

    df = converter.read_msg_to_df(msg)
    expected = pd.DataFrame(
        {
            "latitude": np.array([10.0, 20.0], dtype=np.float64),
            "longitude": np.array([100.0, 110.0], dtype=np.float64),
            "obsvalue_rawbt_1": np.array([300.0, 310.0], dtype=np.float64),
            "datetime": pd.array([pd.Timestamp("2023-06-15T12:00:00")] * 2),
            "temp_1": np.array([0.0, 3.0], dtype=np.float32),
            "temp_2": np.array([1.0, 4.0], dtype=np.float32),
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_wide_reshape_error_returns_empty():
    msg = create_bufr_message(
        lat=[0.0, 0.0],
        lon=[0.0, 0.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        per_datum_arrays={"airTemperature": np.arange(5, dtype=float)},
    )
    msg.get_size.side_effect = lambda k: 6  # reported size != actual data length

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
        per_datum={"airTemperature": {"temp": "0:2"}},
        per_datum_format="wide",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert converter.read_msg_to_df(msg).empty


def test_long_format():
    nreports, ndatum = 2, 3
    pressure_values = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

    msg = create_bufr_message(
        lat=[10.0, 20.0],
        lon=[100.0, 110.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        per_report_arrays={"#1#brightnessTemperature": np.array([290.0, 295.0])},
        per_datum_arrays={"pressure": pressure_values},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
        per_datum={"pressure": {"name": "pressure", "start_index": 0}},
        per_datum_format="long",
    )

    df = converter.read_msg_to_df(msg)
    expected = pd.DataFrame(
        {
            "latitude": np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0], dtype=np.float32),
            "longitude": np.array([100.0, 100.0, 100.0, 110.0, 110.0, 110.0], dtype=np.float32),
            "obsvalue_rawbt_1": np.array([290.0, 290.0, 290.0, 295.0, 295.0, 295.0], dtype=np.float32),
            "datetime": pd.array([pd.Timestamp("2023-06-15T12:00:00")] * (nreports * ndatum)),
            "pressure": np.array(pressure_values, dtype=np.float32),
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_long_format_start_index():
    nreports, ndatum = 1, 4
    pressure_values = np.arange(nreports * ndatum, dtype=float)

    msg = create_bufr_message(
        lat=[0.0],
        lon=[0.0],
        datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
        per_report_arrays={"#1#brightnessTemperature": np.array([275.0])},
        per_datum_arrays={"pressure": pressure_values},
    )

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
        per_datum={"pressure": {"name": "pressure", "start_index": 1}},
        per_datum_format="long",
    )

    df = converter.read_msg_to_df(msg)
    expected_ndatum = ndatum - 1  # first level skipped
    assert len(df) == nreports * expected_ndatum
    assert list(df.columns) == ["latitude", "longitude", "obsvalue_rawbt_1", "datetime", "pressure"]


def test_multiple_messages_combined_and_sorted():
    msgs = [
        create_bufr_message(
            lat=[30.0, 40.0],
            lon=[120.0, 130.0],
            datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 13, "minute": 0, "second": 0},
            per_report_arrays={"#1#brightnessTemperature": np.array([250.0, 255.0])},
        ),
        create_bufr_message(
            lat=[10.0, 20.0],
            lon=[100.0, 110.0],
            datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
            per_report_arrays={"#1#brightnessTemperature": np.array([240.0, 245.0])},
        ),
    ]
    reader = create_bufr_reader(msgs)

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
    )

    df = converter.read_bufr_to_df(reader)
    expected = pd.DataFrame(
        {
            "latitude": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            "longitude": np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32),
            "obsvalue_rawbt_1": np.array([240.0, 245.0, 250.0, 255.0], dtype=np.float32),
            "datetime": pd.array(
                [
                    pd.Timestamp("2023-06-15T12:00:00"),
                    pd.Timestamp("2023-06-15T12:00:00"),
                    pd.Timestamp("2023-06-15T13:00:00"),
                    pd.Timestamp("2023-06-15T13:00:00"),
                ]
            ),
        }
    )
    pd.testing.assert_frame_equal(df, expected)
    assert df["datetime"].is_monotonic_increasing


def test_subset_of_messages():
    msgs = [
        create_bufr_message(
            lat=[float(i)],
            lon=[float(i)],
            datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": i, "minute": 0, "second": 0},
            per_report_arrays={"#1#brightnessTemperature": np.array([200.0 + i])},
        )
        for i in range(5)
    ]
    reader = create_bufr_reader(msgs)

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={"obsvalue_rawbt_1": "#1#brightnessTemperature"},
    )

    df = converter.read_bufr_to_df(reader, msg_idxs=[1, 3])
    assert len(df) == 2
    np.testing.assert_array_equal(
        df["latitude"].values,
        np.array([1.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        df["obsvalue_rawbt_1"].values,
        np.array([201.0, 203.0], dtype=np.float32),
    )


def test_all_messages_filtered_returns_empty():
    msgs = [
        create_bufr_message(
            lat=[0.0],
            lon=[0.0],
            datetime_spec={"year": 2023, "month": 6, "day": 15, "hour": 12, "minute": 0, "second": 0},
            extra_values={"dataCategory": 99},
        ),
    ]
    reader = create_bufr_reader(msgs)

    converter = BUFRToDataFrame(
        latitude="lat",
        longitude="lon",
        per_report={},
        preselect_msg_header={"dataCategory": ["==", 0]},
    )
    assert converter.read_bufr_to_df(reader).empty


@pytest.mark.parametrize(
    "s, expected",
    [
        ("0:2", slice(0, 2)),
        ("1:5:2", slice(1, 5, 2)),
        (":5", slice(None, 5)),
        ("2:", slice(2, None)),
        ("::", slice(None, None, None)),
        ("::2", slice(None, None, 2)),
        ("1:", slice(1, None)),
        (" 1 : 3 ", slice(1, 3)),
        ("-1:", slice(-1, None)),
        ("3", slice(3, 4)),
        ("-1", slice(-1, None)),
        ("0", slice(0, 1)),
    ],
)
def test_parse_slice(s, expected):
    assert parse_slice(s) == expected


@pytest.mark.parametrize(
    "s",
    [
        "",
        "abc",
        "1:2:3:4",
        "a:b",
    ],
)
def test_parse_slice_invalid(s):
    with pytest.raises(ValueError):
        parse_slice(s)


def test_parse_slice_non_string():
    with pytest.raises(ValueError, match="Expected a string"):
        parse_slice(123)
