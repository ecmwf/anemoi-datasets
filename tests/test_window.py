"""Unit tests for window.py"""

import datetime
from functools import cache

import numpy as np
import pytest
import zarr

from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.windows.view import WindowView

ROWS = 63_072_000  # 2 years of per-second data
VARIABLES = 8

START_DATE = datetime.datetime(2020, 1, 1)
INDEXING = "bisect"

# INDEXING='btree'


def generate_event_data(years=2, base_rate=100, variability=0.5, gap_days=(365, 370)) -> np.ndarray:
    """Generates a deterministic time-series array representing events per second.

    This function creates a synthetic dataset for regression testing. It uses
    a sine-wave-based oscillation to simulate noise and a boxcar-style
    subtraction to create a data gap. Because it uses a fixed sine function
    rather than a random seed, the output is perfectly reproducible.

    Parameters
    ----------
    years : int, optional
        The duration of the dataset in years. Defaults to 2.
    base_rate : int, optional
        The average number of events per second. Defaults to 100.
    variability : float, optional
        The maximum percentage fluctuation from the base_rate (0.1 = +/- 10%).
        Defaults to 0.1.
    gap_days : tuple of (int, int), optional
        A tuple defining the start and end day of the missing data gap
        (inclusive). Defaults to (365, 370).

    Returns
    -------
    numpy.ndarray
        A 1D array of type float32 containing the number of events per second.
        The length of the array is (years * 365 * 24 * 3600).

    Notes
    -----
    The formula used is: E(t) = R * (1 + alpha * sin(t)) * Mask(t)
    where Mask(t) is 0 within the gap_days range and 1 elsewhere.
    """
    # 1. Setup Time Array (seconds)
    # 365 days is used as the standard year length for testing consistency
    seconds_in_day = 86400
    total_seconds = years * 365 * seconds_in_day
    t = np.arange(total_seconds, dtype=np.uint32)

    # 2. Base Signal with Deterministic Variability
    # Using sin(t) ensures the 'noise' is identical across test runs
    # without needing to manage a random seed.
    noise = variability * np.sin(t)
    events = base_rate * (1 + noise)

    # 3. Apply the Data Gap
    # Convert day offsets to absolute second offsets
    start_sec = gap_days[0] * seconds_in_day
    end_sec = gap_days[1] * seconds_in_day

    print(
        "GAP from",
        START_DATE + datetime.timedelta(seconds=start_sec),
        "to",
        START_DATE + datetime.timedelta(seconds=end_sec),
    )

    # Apply a boolean mask (the computational version of a Heaviside step)
    events[(t >= start_sec) & (t <= end_sec)] = 0

    return events.astype(np.float32)


def to_expanded_2d(events) -> np.ndarray:
    """Transforms a 1D event array into a 3-column 2D array,
    filtering out the gap periods.

    Parameters
    ----------
    events : numpy.ndarray
        A 1D array of event counts per second.

    Returns
    -------
    numpy.ndarray
        A 2D array with shape (N, 3):
        - Column 0: Time Index (seconds)
        - Column 1: Cumulative Sum of events up to this second
        - Column 2: Event count at this specific second
    """
    # 1. Calculate the running total (Cumulative Sum)
    # We do this before filtering so the total remains accurate
    # relative to the full timeline.

    raw_cumsum = np.cumsum(events)

    # 2. Shift the sum to make it "exclusive" (sum until this date)
    # We pad with a 0 at the start and remove the last element to keep length equal
    cumulative_until = np.zeros_like(raw_cumsum)
    cumulative_until[1:] = raw_cumsum[:-1]

    # 3. Identify non-gap indices
    mask = events > 0

    # 4. Create the Time Index
    indices = np.arange(len(events), dtype=np.uint32)

    # 5. Filter all three components using the mask
    filtered_indices = indices[mask] + int(START_DATE.timestamp())
    filtered_cumulative = cumulative_until[mask]
    filtered_values = events[mask]

    # 6. Stack into a 2D array (N rows, 3 columns)
    # Note: NumPy will cast indices to float to match the event data types.
    return np.column_stack((filtered_indices, filtered_cumulative, filtered_values))


# Example usage:
# events_1d = generate_event_data()
# sparse_data = to_sparse_2d(events_1d)


@cache
def create_tabular_store():

    COLS_WITH_DATE_TIME_LAT_LON = VARIABLES + 4

    events = generate_event_data()
    dates = to_expanded_2d(events)

    assert len(events) == ROWS, f"Generated events length {len(events)} does not match expected {ROWS}"

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
            slice_obj = sample.meta.slice_obj
            assert slice_obj.start == total, (slice_obj, total)
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
