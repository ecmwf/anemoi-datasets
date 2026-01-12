"""Unit tests for window.py"""

import datetime
import time

import numpy as np
import pytest
import zarr

from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.windows.view import WindowView

VARIABLES = 3

START_DATE = datetime.datetime(2020, 1, 1)


def _generate_event_data(years=1, base_rate=10, variability=0.2, gap_days=(160, 165)) -> np.ndarray:
    """Generates a deterministic time-series array representing events per second.

    This function creates a synthetic dataset for regression testing. It uses
    a sine-wave-based oscillation to simulate noise and a boxcar-style
    subtraction to create a data gap. Because it uses a fixed sine function
    rather than a random seed, the output is perfectly reproducible.

    Parameters
    ----------
    years : int, optional
        The duration of the dataset in years. Defaults to 1.
    base_rate : int, optional
        The average number of events per second. Defaults to 100.
    variability : float, optional
        The maximum percentage fluctuation from the base_rate (0.1 = +/- 10%).
        Defaults to 0.5.
    gap_days : tuple of (int, int), optional
        A tuple defining the start and end day of the missing data gap
        (inclusive). Defaults to (160, 165).

    Returns
    -------
    np.ndarray
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

    return events.astype(np.int64)


def _to_expanded_2d(events) -> np.ndarray:
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
    indices = np.arange(len(events), dtype=np.int64)

    # 5. Filter all three components using the mask
    filtered_indices = indices[mask] + int(START_DATE.timestamp())
    filtered_cumulative = cumulative_until[mask]
    filtered_values = events[mask]

    # 6. Stack into a 2D array (N rows, 3 columns)
    # Note: NumPy will cast indices to float to match the event data types.
    return np.column_stack((filtered_indices, filtered_cumulative, filtered_values))


def _create_tabular_store(indexing) -> zarr.Group:

    COLS_WITH_DATE_TIME_LAT_LON = VARIABLES + 4

    events = _generate_event_data()
    dates = _to_expanded_2d(events)

    number_of_samples = dates[-1][1] + dates[-1][2]
    start = time.time()
    print("Number of samples to generate:", f"{number_of_samples:,}")

    # data = np.random.rand(number_of_samples, COLS_WITH_DATE_TIME_LAT_LON).astype(np.float32)
    data = np.zeros((number_of_samples, COLS_WITH_DATE_TIME_LAT_LON), dtype=np.float32)
    start_stamp = START_DATE.timestamp()

    data[:, 0] = (
        np.repeat(np.arange(24 * 60 * 60 + 1), number_of_samples // (24 * 60 * 60) + 1)[:number_of_samples]
        + start_stamp
    )  # seconds since midnight
    data[:, 1] = np.linspace(0, 24 * 60 * 60 - 1, 1).repeat(number_of_samples)[
        :number_of_samples
    ]  # time offsets in seconds within the day
    data[:, 2] = np.linspace(-90, 90, 1).repeat(number_of_samples)[:number_of_samples]  # latitudes
    data[:, 3] = np.linspace(0, 359, 1).repeat(number_of_samples)[:number_of_samples]  # longitudes
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("data", data=data, chunks=(10000, COLS_WITH_DATE_TIME_LAT_LON), dtype=np.float32)

    print("Data generation took", time.time() - start, "seconds")

    index = create_date_indexing(indexing, root)
    root.attrs["date_indexing"] = indexing

    dates = np.array(dates, dtype=np.int64)

    # Commented out to test that the synthetic data is valid
    # It has been validated already. Uncomment if you modify the data generation.
    # index.validate_bulk_load_input(dates, data_length=len(data))

    start = time.time()
    print("Bulk loading date index...")
    index.bulk_load(dates)
    print("Bulk loading took", time.time() - start, "seconds")

    return root


def _test_window_view(view):

    print("+++++++++", view.start_date, view.end_date)

    assert view.frequency == datetime.timedelta(hours=3)
    assert view.window.before == -datetime.timedelta(hours=3)
    assert view.window.after == datetime.timedelta(hours=0)
    assert view.window.exclude_before is True
    assert view.window.exclude_after is False

    # Make sure we can iterate over all samples
    # Not 100% independent since we use the same code for whole_range, as in __getitem__

    whole_range = view.whole_range
    total = whole_range.start

    for i, sample in enumerate(view):
        assert 0 <= sample.shape[0] <= 2 * 100 * 60 * 60 * 3, f"Sample {i} has unexpected shape {sample.shape}"
        # print(f"+++++++++++++ Sample {i}: slice {sample.meta.slice_obj}, shape {sample.shape}")
        if sample.shape[0] != 0:
            slice_obj = sample.meta.slice_obj
            assert slice_obj.start == total, (slice_obj, total, total - slice_obj.start)

        assert sample.shape[1] == VARIABLES
        total += sample.shape[0]

    assert (
        total == whole_range.stop
    ), f"Total rows {total:,} does not match expected {whole_range.stop:,} ({whole_range.stop - total:,} missing)"

    with pytest.raises(IndexError):
        view[len(view)]

    return view


@pytest.fixture(scope="session")
def tabular_stores():
    # This dictionary lives for the entire test session
    return {}


@pytest.fixture
def tabular_store(tabular_stores, request):
    param = request.param
    if param not in tabular_stores:
        tabular_stores[param] = _create_tabular_store(param)

    return tabular_stores[param]


WINDOW_VIEW_TEST_CASES = {
    # 1. Default date range (no modification)
    "default_range": (0, 0),
    # 2. Start date extended 90 days before original
    "start_extended_90d_before": (-90, 0),
    # 3. End date extended 90 days after original
    "end_extended_90d_after": (0, 90),
    # 4. Both start and end extended by 90 days
    "both_extended_90d": (-90, 90),
    # 5. Start moved 90 days after original
    "start_moved_90d_after": (90, 0),
    # 6. Start moved forward 90 days, end moved backward 90 days
    "start_90d_after_end_90d_before": (90, -90),
    # 7. Start moved forward 90 days, end extended 90 days after
    "start_90d_after_end_90d_after": (90, 90),
    # 8. Start extended 90 days before, end moved 90 days backward
    "start_90d_before_end_90d_before": (-90, -90),
    # 9. Start in gap period (163 days after start)
    "start_in_gap_period": (163, 0),
    # 10. End in gap period (163 days before end)
    "end_in_gap_period": (0, -163),
    # 11. Before any available data (both start and end moved back 3650 days)
    "before_any_data": (-3650, -3650),
    # 12. After any available data (both start and end moved forward 3650 days)
    "after_any_data": (3650, 3650),
}


@pytest.mark.parametrize("tabular_store", ["bisect", "btree"], indirect=True)
@pytest.mark.parametrize(
    "start_delta,end_delta",
    WINDOW_VIEW_TEST_CASES.values(),
    ids=WINDOW_VIEW_TEST_CASES.keys(),
)
def test_window_view(tabular_store, start_delta, end_delta):
    view = WindowView(tabular_store)
    view = view.set_start(view.start_date + datetime.timedelta(days=start_delta))
    view = view.set_end(view.end_date + datetime.timedelta(days=end_delta))
    _test_window_view(view)
