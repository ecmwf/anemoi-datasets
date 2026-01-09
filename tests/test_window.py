"""Unit tests for window.py"""

import datetime
from functools import cache

import numpy as np
import pytest
import zarr

from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.windows.view import WindowView

ROWS = 5_000_000
VARIABLES = 8
ROWS_PER_SECOND = 10

START_DATE = datetime.datetime(2020, 1, 1)
INDEXING = "bisect"

# INDEXING='btree'


@cache
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


def _make_view():
    """Test the WindowView class for correct slicing and metadata handling."""

    store = create_tabular_store()
    return WindowView(store)


def _test_window_view(view):

    print("+++++++++", view.start_date, view.end_date)

    assert view.frequency == datetime.timedelta(hours=3)
    assert view.window.before == -datetime.timedelta(hours=3)
    assert view.window.after == datetime.timedelta(hours=0)
    assert view.window.exclude_before is True
    assert view.window.exclude_after is False

    # Make sure we can iterate over all samples
    total = 0
    for i, sample in enumerate(view):
        print(f"+++++++++++++ Sample {i}: slice {sample.meta.slice_obj}, shape {sample.shape}")
        if sample.shape[0] != 0:

            assert sample.meta.slice_obj.start == total, (sample.meta.slice_obj, total)
        assert sample.shape[1] == VARIABLES
        # assert sample.reference_date == START_DATE + i * view.frequency, (sample.reference_date, START_DATE + i * view.frequency)
        total += sample.shape[0]

    assert total == ROWS, f"Total rows {total:,} does not match expected {ROWS:,} ({ROWS - total:,} missing)"

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
    """Test the WindowView class with default date range.

    Uses the original data range without modifications:

        [====================]
        ^                    ^
      start                end

    Original: [--------------------]
    Modified: [====================]
    """
    view = _make_view()
    _test_window_view(view)


def test_window_view_2():
    """Test with start date extended 90 days before the original data start date.

    Extends the view window to start earlier than the data:

              <---90d--->
    Original:            [--------------------]
    Modified: [==========================]
              ^          ^                    ^
            new        orig                 orig
            start      start                end
    """
    view = _make_view()
    view = view.set_start(view.start_date - datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_3():
    """Test with end date extended 90 days after the original data end date.

    Extends the view window to end later than the data:

                                        <---90d--->
    Original: [--------------------]
    Modified: [==========================]
              ^                    ^               ^
            orig                 orig             new
            start                end              end
    """
    view = _make_view()
    view = view.set_end(view.end_date + datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_4():
    """Test with both start and end dates extended by 90 days.

    Extends the view window on both sides:

              <---90d--->                   <---90d--->
    Original:            [--------------------]
    Modified: [==========================================]
              ^          ^                    ^          ^
            new        orig                 orig        new
            start      start                end         end
    """
    view = _make_view()
    view = view.set_start(view.start_date - datetime.timedelta(days=90))
    view = view.set_end(view.end_date + datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_5():
    """Test with start date moved 90 days after the original data start date.

    Narrows the view window from the start:

                   <---90d--->
    Original: [--------------------]
    Modified:            [==========]
              ^          ^          ^
            orig        new        orig
            start       start      end
    """
    view = _make_view()
    view = view.set_start(view.start_date + datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_6():
    """Test with both start moved forward and end moved backward by 90 days.

    Narrows the view window on both sides:

                   <---90d--->
    Original: [--------------------]
              <---90d--->
    Modified:            [====]
              ^          ^    ^     ^
            orig        new  new   orig
            start       start end  end
    """
    view = _make_view()
    view = view.set_syaty(view.start_date + datetime.timedelta(days=90))
    view = view.set_end(view.end_date - datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_7():
    """Test with start moved forward 90 days and end extended 90 days after.

    Shifts and extends the view window:

                   <---90d--->
    Original: [--------------------]
                                        <---90d--->
    Modified:            [==========================]
              ^          ^          ^               ^
            orig        new        orig            new
            start       start      end             end
    """
    view = _make_view()
    view = view.set_syaty(view.start_date + datetime.timedelta(days=90))
    view = view.set_end(view.end_date + datetime.timedelta(days=90))
    _test_window_view(view)


def test_window_view_8():
    """Test with start extended 90 days before and end moved 90 days backward.

    Extends the start and narrows the end:

              <---90d--->
    Original:            [--------------------]
                                   <---90d--->
    Modified: [====================]
              ^          ^          ^          ^
            new        orig        new        orig
            start      start       end        end
    """
    view = _make_view()
    view = view.set_start(view.start_date - datetime.timedelta(days=90))
    view = view.set_end(view.end_date - datetime.timedelta(days=90))
    _test_window_view(view)


if __name__ == "__main__":
    """Run all test functions in the module."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
