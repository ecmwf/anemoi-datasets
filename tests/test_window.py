"""Unit tests for window.py"""

import datetime

import numpy as np
import pytest
import zarr

from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.dates.windows.view import WindowView

ROWS = 5_000_000
VARIABLES = 8
ROWS_PER_SECOND = 10

START_DATE = datetime.datetime(2020, 1, 1)
INDEXING = "bisect"

# INDEXING='btree'


def create_tabular_store():

    COLS_WITH_DATE_TIME_LAT_LON = VARIABLES + 4

    dates = []
    for i in range(ROWS // ROWS_PER_SECOND):
        date = START_DATE + datetime.timedelta(seconds=i)
        dates.append((int(date.timestamp()), i * ROWS_PER_SECOND, ROWS_PER_SECOND))

    data = np.random.rand(ROWS, COLS_WITH_DATE_TIME_LAT_LON).astype(np.float32)
    start_stamp = START_DATE.timestamp()

    data[:, 0] = (
        np.repeat(np.arange(24 * 60 * 60 + 1), ROWS // (24 * 60 * 60) + 1)[:ROWS] + start_stamp
    )  # seconds since midnight
    data[:, 1] = np.linspace(0, 24 * 60 * 60 - 1, 1).repeat(ROWS)[:ROWS]  # time offsets in seconds within the day
    data[:, 2] = np.linspace(-90, 90, 1).repeat(ROWS)[:ROWS]  # latitudes
    data[:, 3] = np.linspace(0, 359, 1).repeat(ROWS)[:ROWS]  # longitudes

    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("data", data=data, chunks=(10000, COLS_WITH_DATE_TIME_LAT_LON), dtype=np.float32)

    index = create_date_indexing(INDEXING, root)
    root.attrs["date_indexing"] = INDEXING

    dates = np.array(dates, dtype=np.int64)

    index.bulk_load(dates)

    return root


def windows_width(view):
    return int(
        (view.window.after - view.window.before).total_seconds()
    )  # - (1 if view.window.exclude_before else 0) - (1 if view.window.exclude_after else 0))


def _test_window_view(**kwargs):
    """Test the WindowView class for correct slicing and metadata handling."""

    store = create_tabular_store()
    view = WindowView(store)

    print(view.start_date, view.end_date)

    assert view.frequency == datetime.timedelta(hours=3)
    assert view.window.before == -datetime.timedelta(hours=3)
    assert view.window.after == datetime.timedelta(hours=0)
    assert view.window.exclude_before is True
    assert view.window.exclude_after is False

    # Make sure we can iterate over all samples
    total = 0
    for i, sample in enumerate(view):
        assert sample.shape[1] == VARIABLES
        assert sample.reference_date == START_DATE + i * view.frequency
        total += sample.shape[0]

    assert total == ROWS

    with pytest.raises(IndexError):
        view[len(view)]

    # sample = view[5]
    # assert isinstance(sample, AnnotatedNDArray)

    # # assert sample.shape == (2*windows_width(view) * ROWS_PER_SECOND, VARIABLES) # 2????

    # assert sample.boundaries == [
    #     slice(
    #         0,1
    #     )
    # ]
    # assert sample.reference_date == view.start_date + 5 * view.frequency
    # assert sample.reference_dates is None
    # assert sample.index == 5
    # assert sample.dates is None
    # assert sample.timedeltas is None
    # assert sample.latitudes is None
    # assert sample.longitudes is None

    return view


def test_window_view_1():
    """Test the WindowView class."""
    view = _test_window_view()

    assert view.start_date == START_DATE
    assert view.end_date == START_DATE + datetime.timedelta(seconds=(ROWS // ROWS_PER_SECOND - 1))

    print()


def test_window_view_2():
    """Start the dates before the data start date."""
    view = _test_window_view(start=2019)

    assert view.start_date == START_DATE
    assert view.end_date == START_DATE + datetime.timedelta(seconds=(ROWS // ROWS_PER_SECOND - 1))

    print()


if __name__ == "__main__":
    """Run all test functions in the module."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
