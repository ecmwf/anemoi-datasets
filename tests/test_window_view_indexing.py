# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for selecting several windows of a WindowView at once."""

import datetime

import numpy as np
import pytest
import zarr

from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.windows.view import WindowView

N_HOURS = 25
REPORTS_PER_HOUR = 2
FIRST_DATE = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


def _hourly_store(indexing: str) -> zarr.Group:
    """Build a tabular store with ``REPORTS_PER_HOUR`` reports at every hour.

    The reports of hour ``h`` sit at latitude ``h`` and longitudes ``0, 1, ...``,
    and their two variables hold ``h`` and ``-h``.
    """
    epochs = np.array(
        [int((FIRST_DATE + datetime.timedelta(hours=hour)).timestamp()) for hour in range(N_HOURS)],
        dtype=np.int64,
    )
    # Columns: days and seconds since 1970, latitude, longitude, then the variables.
    rows = [
        [epoch // 86400, epoch % 86400, hour, report, hour, -hour]
        for hour, epoch in enumerate(epochs)
        for report in range(REPORTS_PER_HOUR)
    ]
    root = zarr.group(store=zarr.storage.MemoryStore(), overwrite=True)
    root.create_array("data", data=np.asarray(rows, dtype=np.int64))

    index = create_date_indexing(indexing, root)
    root.attrs["date_indexing"] = indexing
    # One row per report time: its time, where its reports start, and how many there are.
    index.bulk_load(
        np.stack(
            [epochs, np.arange(N_HOURS) * REPORTS_PER_HOUR, np.full(N_HOURS, REPORTS_PER_HOUR)],
            axis=1,
        ).astype(np.int64)
    )
    return root


@pytest.fixture(scope="module", params=["bisect", "btree"])
def view(request) -> WindowView:
    """Hourly windows, each holding the reports of its own hour."""
    return WindowView(_hourly_store(request.param)).set_frequency(1).set_window("(-1,+0]")


def _assert_same_windows(result, windows) -> None:
    """Check ``result`` holds ``windows`` joined one after the other."""
    np.testing.assert_array_equal(result, np.concatenate(windows))
    np.testing.assert_array_equal(result.latitudes, np.concatenate([w.latitudes for w in windows]))
    np.testing.assert_array_equal(result.longitudes, np.concatenate([w.longitudes for w in windows]))
    np.testing.assert_array_equal(result.timedeltas, np.concatenate([w.timedeltas for w in windows]))
    np.testing.assert_array_equal(result.reference_dates, np.concatenate([w.reference_dates for w in windows]))
    ends = np.cumsum([len(w) for w in windows])
    assert result.boundaries == [slice(int(a), int(b)) for a, b in zip([0, *ends[:-1]], ends)]


def test_single_window_holds_its_hour(view: WindowView) -> None:
    window = view[6]

    np.testing.assert_array_equal(window, [[6, -6]] * REPORTS_PER_HOUR)
    np.testing.assert_array_equal(window.latitudes, [6] * REPORTS_PER_HOUR)
    np.testing.assert_array_equal(window.timedeltas, [0] * REPORTS_PER_HOUR)


@pytest.mark.parametrize(
    "index",
    [
        slice(0, 13, 6),
        slice(1, 14, 6),
        slice(3, 10, 3),
        slice(None, None, 12),
        [0, 6, 12],
        [12, 0, 7],
        [5, 5],
        np.array([2, 8, 14]),
        np.array([2, 8, 14], dtype=np.int32),
    ],
    ids=str,
)
def test_several_windows_are_the_single_windows_joined(view: WindowView, index) -> None:
    hours = range(*index.indices(len(view))) if isinstance(index, slice) else [int(i) for i in index]

    result = view[index]

    _assert_same_windows(result, [view[hour] for hour in hours])
    np.testing.assert_array_equal(result[:, 0], np.repeat(list(hours), REPORTS_PER_HOUR))


def test_contiguous_slice_is_unchanged(view: WindowView) -> None:
    _assert_same_windows(view[3:6], [view[3], view[4], view[5]])


def test_numpy_integer_selects_one_window(view: WindowView) -> None:
    np.testing.assert_array_equal(view[np.int64(6)], view[6])
    assert view[np.int64(6)].boundaries == [slice(0, REPORTS_PER_HOUR)]


def test_negative_index_counts_from_the_end(view: WindowView) -> None:
    np.testing.assert_array_equal(view[-1], view[len(view) - 1])
    np.testing.assert_array_equal(view[-len(view)], view[0])


def test_tuple_selects_columns_of_several_windows(view: WindowView) -> None:
    np.testing.assert_array_equal(view[[0, 6], 1], np.concatenate([view[0], view[6]])[:, 1])
    np.testing.assert_array_equal(view[0:13:6, 0], np.concatenate([view[0], view[6], view[12]])[:, 0])


@pytest.mark.parametrize("index", [slice(3, 3), slice(5, 0, 1), []], ids=str)
def test_selecting_no_windows_raises(view: WindowView, index) -> None:
    with pytest.raises(IndexError, match="selects no windows"):
        view[index]


@pytest.mark.parametrize("index", [[0.0, 6.0], np.zeros((2, 2), dtype=np.int64)], ids=["floats", "2d"])
def test_window_indices_must_be_a_list_of_integers(view: WindowView, index) -> None:
    with pytest.raises(TypeError, match="1-D sequence of integers"):
        view[index]


def test_out_of_range_window_raises(view: WindowView) -> None:
    with pytest.raises(IndexError):
        view[[0, len(view)]]
