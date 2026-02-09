# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pickle
import tempfile
import time
from copy import deepcopy

import numpy as np
import pytest

from anemoi.datasets.create.recipe.statistics import PicklableFilter
from anemoi.datasets.create.statistics import StatisticsCollector

# set numpy seed
seed = np.random.randint(0, 10000)
print(f"Using numpy random seed: {seed}")
np.random.seed(seed)


class DatasetMock:
    def __init__(self, data, dates):
        self.data = data
        self.dates = dates


def _create_random_stats(N, C=5, nan_fraction=0.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate random data with known statistics.

    Args:
        N: Number of rows
        C: Number of columns
        nan_fraction: Fraction of values to set as NaN (0.0 to 1.0)

    Returns:
        data: The generated data array
        target_stats: Dictionary with expected statistics
    """
    print("Generating random data...")

    # Define target means and stds for each column
    target_means = np.linspace(-10, 10, C)
    target_stds = np.linspace(1, 5, C)

    # Generate raw random data
    data = np.random.randn(N, C)

    # some variables might be constant
    constants = [np.random.rand() < 0.1 for _ in range(C)]
    data[:, constants] = target_means[constants]

    # Adjust each column to have EXACTLY the target mean and std
    for i in range(C):
        col = data[:, i]
        # Standardize to mean=0, std=1
        col = (col - col.mean()) / col.std() if col.std() > 0 else col - col.mean()
        # Scale and shift to targets
        data[:, i] = (col * target_stds[i]) + target_means[i]

    # Introduce NaNs if requested (only in non-constant variables)
    if nan_fraction > 0:
        constants_without_nan = [np.random.rand() < 0.1 for _ in constants]
        nan_mask = np.random.rand(N, C) < nan_fraction
        nan_mask[:, constants_without_nan] = False  # Do not introduce NaNs in constant columns
        data[nan_mask] = np.nan
        nan_count = np.sum(nan_mask)
        print(f"Introduced {nan_count} NaN values ({100*nan_count/(N*C):.1f}%)")

    # Capture the actual stats of the modified data
    target_stats = _compute_statistics(data, nan=(nan_fraction > 0))

    print("Verifying target statistics...")
    _check_statistics(data, target_stats)

    return data, target_stats


def _compute_statistics(data, nan=False):
    """Compute mean, stdev, min, max for each column, optionally ignoring NaNs."""
    if nan:
        mean = np.nanmean(data, axis=0)
        stdev = np.nanstd(data, axis=0)
        minimum = np.nanmin(data, axis=0)
        maximum = np.nanmax(data, axis=0)
    else:
        mean = data.mean(axis=0)
        stdev = data.std(axis=0)
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
    return {
        "mean": mean,
        "stdev": stdev,
        "minimum": minimum,
        "maximum": maximum,
    }


def _compute_tendency_statistics(data, delta, nan=False):
    """Compute statistics for tendencies (differences with lag delta)."""
    tendencies = data[delta:] - data[:-delta]
    return _compute_statistics(tendencies, nan=nan)


def _check_statistics(data, target_stats):
    """Verify that computed statistics match target statistics."""
    # Check if data has NaNs to determine which computation method to use
    has_nans = np.any(np.isnan(data))
    calc_stats = _compute_statistics(data, nan=has_nans)
    assert np.allclose(calc_stats["mean"], target_stats["mean"], equal_nan=True), "Mean check failed"
    assert np.allclose(calc_stats["stdev"], target_stats["stdev"], equal_nan=True), "Std deviation check failed"
    assert np.allclose(calc_stats["minimum"], target_stats["minimum"], equal_nan=True), "Min check failed"
    assert np.allclose(calc_stats["maximum"], target_stats["maximum"], equal_nan=True), "Max check failed"


def test_statistics_collector_single_batch(N=1_000_000, C=5):
    """Test statistics collector with a single batch."""
    data, target_stats = _create_random_stats(N, C)

    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)])

    start = time.time()
    print("Collecting statistics...")
    collector.collect(data, range(N))
    print(f"Collection took {time.time() - start:.2f} seconds")

    computed_stats = collector.statistics()

    print("\nResults:")
    for name, target in target_stats.items():
        print(f"{name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed_stats[name]}")
        assert np.allclose(
            computed_stats[name], target, rtol=1e-10, equal_nan=True
        ), f"{name.capitalize()} does not match!"

    print("✓ Single batch test PASSED")


def test_statistics_collector_multiple_batches(N=1_000_000, C=5, batch_size=10_000):
    """Test statistics collector with multiple batches."""
    data, target_stats = _create_random_stats(N, C)

    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)])

    # Split data into batches
    num_batches = (N + batch_size - 1) // batch_size
    print(f"Splitting data into {num_batches} batches...")

    start = time.time()
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch = data[i:end_idx]
        collector.collect(batch, range(i, end_idx))

        if (i // batch_size) % 10 == 0:
            print(f"  Processed batch {i // batch_size + 1}/{num_batches}")

    print(f"Collection took {time.time() - start:.2f} seconds")

    computed_stats = collector.statistics()

    print("\nResults:")
    for name, target in target_stats.items():
        print(f"{name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed_stats[name]}")
        diff = np.abs(computed_stats[name] - target)
        print(f"   Max diff: {np.max(diff):.2e}")
        assert np.allclose(
            computed_stats[name], target, rtol=1e-10, equal_nan=True
        ), f"{name.capitalize()} does not match!"

    print("✓ Multiple batches test PASSED")


def test_statistics_with_nans(N=100_000, C=3, nan_fraction=0.1):
    """Test statistics collector with NaN values."""
    data, target_stats = _create_random_stats(N, C, nan_fraction=nan_fraction)

    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)], allow_nans=True)

    # Test with batches
    batch_size = 10_000
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch = data[i:end_idx]
        collector.collect(batch, range(i, end_idx))

    computed_stats = collector.statistics()

    print("\nResults:")
    for name, target in target_stats.items():
        print(f"{name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed_stats[name]}")
        assert np.allclose(
            computed_stats[name], target, rtol=1e-9, equal_nan=True
        ), f"{name.capitalize()} does not match!"

    print("✓ NaN handling test PASSED")


def test_edge_cases():
    """Test edge cases."""
    # Test with very small batches
    print("\n1. Testing with single-row batches...")
    data, target_stats = _create_random_stats(100, 3)

    collector = StatisticsCollector(variables_names=["a", "b", "c"])
    for i in range(len(data)):
        collector.collect(data[i : i + 1], [i])

    computed_stats = collector.statistics()
    for name in target_stats:
        assert np.allclose(
            computed_stats[name], target_stats[name], equal_nan=True
        ), f"Single-row batch test failed for {name}"
    print("   ✓ Single-row batches work correctly")

    # Test with varying batch sizes
    print("\n2. Testing with varying batch sizes...")
    collector2 = StatisticsCollector(variables_names=["a", "b", "c"])
    batch_sizes = [1, 5, 10, 20, 30, 34]  # Total = 100
    idx = 0
    for bs in batch_sizes:
        collector2.collect(data[idx : idx + bs], range(idx, idx + bs))
        idx += bs

    computed_stats2 = collector2.statistics()
    for name in target_stats:
        assert np.allclose(
            computed_stats2[name], target_stats[name], equal_nan=True
        ), f"Varying batch size test failed for {name}"
    print("   ✓ Varying batch sizes work correctly")

    print("\n✓ All edge cases PASSED")


def test_tendencies_single_batch(N=1000, C=3, delta=1):
    """Test tendency statistics with a single batch."""
    # Generate data
    data, _ = _create_random_stats(N, C)

    # Compute expected tendency statistics
    target_stats = _compute_tendency_statistics(data, delta)

    print(f"Generated {N} samples, computing tendencies with delta={delta}")
    print(f"Expected {N - delta} tendency samples")

    # Create collector with tendencies
    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)], tendencies={f"delta_{delta}": delta})

    # Collect all data at once
    collector.collect(data, range(N))

    computed_stats = collector.statistics()

    print("\nResults:")
    for stat_name, target in target_stats.items():
        key = f"statistics_tendencies_delta_{delta}_{stat_name}"
        computed = computed_stats[key]
        print(f"{stat_name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed}")
        diff = np.abs(computed - target)
        print(f"   Max diff: {np.max(diff):.2e}")
        assert np.allclose(computed, target, rtol=1e-9, equal_nan=True), f"Tendency {stat_name} does not match!"

    print("✓ Tendencies single batch test PASSED")


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 11, 20])
@pytest.mark.parametrize("delta", [1, 5, 10])
def test_tendencies_multiple_batches(batch_size, delta, N=100, C=3):
    # def test_tendencies_multiple_batches(N=1000, C=3, delta=1, batch_size=110):
    """Test tendency statistics with multiple batches."""
    data, _ = _create_random_stats(N, C)

    # Compute expected tendency statistics
    target_stats = _compute_tendency_statistics(data, delta)

    print(f"Generated {N} samples, computing tendencies with delta={delta}")
    print(f"Expected {N - delta} tendency samples")

    # Create collector with tendencies
    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)], tendencies={f"delta_{delta}": delta})

    # Collect data in batches
    num_batches = (N + batch_size - 1) // batch_size
    print(f"Splitting data into {num_batches} batches...")

    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch = data[i:end_idx]
        collector.collect(batch, range(i, end_idx))

    computed_stats = collector.statistics()

    print("\nResults:")
    for stat_name, target in target_stats.items():
        key = f"statistics_tendencies_delta_{delta}_{stat_name}"
        computed = computed_stats[key]
        print(f"{stat_name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed}")
        diff = np.abs(computed - target)
        print(f"   Max diff: {np.max(diff):.2e}")
        assert np.allclose(computed, target, rtol=1e-9, equal_nan=True), f"Tendency {stat_name} does not match!"

    print("✓ Tendencies multiple batches test PASSED")


def test_tendencies_multiple_deltas(N=500, C=2):
    """Test computing multiple tendency deltas at the same time."""
    data, _ = _create_random_stats(N, C)
    deltas = {"delta_1": 1, "delta_6": 6, "delta_12": 12}

    # Compute expected statistics for each delta
    expected = {}
    for name, delta in deltas.items():
        expected[name] = _compute_tendency_statistics(data, delta)

    # Create collector with multiple tendencies
    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)], tendencies=deltas)

    # Collect data in batches
    batch_size = 50
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch = data[i:end_idx]
        collector.collect(batch, range(i, end_idx))

    computed_stats = collector.statistics()

    print("\nResults:")
    for tendency_name, delta in deltas.items():
        print(f"\n{tendency_name} (delta={delta}):")
        for stat_name, target in expected[tendency_name].items():
            key = f"statistics_tendencies_{tendency_name}_{stat_name}"
            computed = computed_stats[key]
            diff = np.abs(computed - target)
            print(f"  {stat_name}: max_diff={np.max(diff):.2e}")
            assert np.allclose(
                computed, target, rtol=1e-9, equal_nan=True
            ), f"Tendency {tendency_name}, {stat_name} does not match!"

    print("\n✓ Multiple tendencies simultaneously test PASSED")


@pytest.mark.parametrize("delta", [1, 2, 5])
@pytest.mark.parametrize("filter_end", [None, 1, 2, 3, 4, 5, 6, 50, 94, 95, 96, 97, 98, 99])
@pytest.mark.parametrize("filter_start", [None, 1, 2, 3, 4, 5, 6, 50, 94, 95, 96, 97, 98, 99, 100])
def test_tendencies_with_filter(filter_start, filter_end, delta, C=2):
    if filter_start is not None and filter_end is not None and filter_end < filter_start:
        return

    data, _ = _create_random_stats(100, C)
    dates = np.arange(len(data))

    if filter_start is None:
        filter_start = dates[0]
    if filter_end is None:
        filter_end = dates[-1]

    if filter_end - filter_start < delta:
        return  # Not enough data points for tendencies

    filter = PicklableFilter(filter_start, filter_end)

    # Actual filtered range is [filter_start, filter_end] (inclusive on both ends)
    # For tendencies, we also need the lookback data to be in range
    # So valid tendencies are for indices in [filter_start + delta, filter_end + 1)
    expected_tendencies = data[filter_start + delta : filter_end + 1] - data[filter_start : filter_end + 1 - delta]
    expected = _compute_statistics(expected_tendencies, nan=False)

    # Create collector with tendency and filter
    collector = StatisticsCollector(
        variables_names=[f"var{i}" for i in range(C)], tendencies={f"delta_{delta}": delta}, filter=filter
    )

    # Collect all data at once
    collector.collect(data, range(len(data)))

    computed_stats = collector.statistics()

    print("\nResults:")
    for stat_name, target in expected.items():
        key = f"statistics_tendencies_delta_{delta}_{stat_name}"
        computed = computed_stats[key]
        diff = np.abs(computed - target)
        print(f"{stat_name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Computed: {computed}")
        print(f"   Max diff: {np.max(diff):.2e}")
        assert np.allclose(
            computed, target, rtol=1e-9, equal_nan=True
        ), f"Tendency {stat_name} with filter does not match!"

    print("✓ Tendencies with filter test PASSED")


def test_serialisation():
    data, _ = _create_random_stats(1000, 3, nan_fraction=0.05)
    c = StatisticsCollector(variables_names=["a", "b", "c"], allow_nans=True, tendencies={"delta_1": 1})
    c.collect(data, range(len(data)))
    c_stats = deepcopy(c.statistics())
    c_constants = deepcopy(c.constant_variables())

    c2 = pickle.loads(pickle.dumps(c))

    compare_statistics(c_stats, c2.statistics())
    compare_constants(c_constants, c2.constant_variables())


@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param({"delta_1": 1, "delta_2": 2}, id="deltas_1_2"),
        pytest.param({"delta_10": 10}, id="delta10"),
        pytest.param({"delta_1": 1, "delta_5": 5, "delta_10": 10}, id="deltas_1_5_10"),
    ],
)
@pytest.mark.parametrize("filter_end", [None, 1, 2, 3, 4, 5, 6, 10, 20, 50, 94, 95, 96, 97, 98, 99, 100])
@pytest.mark.parametrize(
    "cutoff",
    [
        pytest.param(50, id="cutoff50"),
        pytest.param(70, id="cutoff70"),
    ],
)
@pytest.mark.parametrize("filter_start", [None, 1, 2, 3, 4, 5, 6, 10, 20, 50, 94, 95, 96, 97, 98, 99, 100])
def test_merge_statistic_with_filter(filter_start, filter_end, deltas, cutoff):
    if filter_start is not None and filter_end is not None and filter_end < filter_start:
        return

    data, _ = _create_random_stats(100, 2)
    dates = np.arange(len(data)).astype(float)

    if filter_start is None:
        filter_start = dates[0]
    if filter_end is None:
        filter_end = dates[-1]
    filter = PicklableFilter(filter_start, filter_end)

    c0 = StatisticsCollector(variables_names=["a", "b"], tendencies=deltas, filter=filter)
    data = data.copy()
    c0.collect(data, dates)
    c0_stats = deepcopy(c0.statistics())
    c0_constants = deepcopy(c0.constant_variables())

    c1 = StatisticsCollector(variables_names=["a", "b"], tendencies=deltas, filter=filter)
    c2 = StatisticsCollector(variables_names=["a", "b"], tendencies=deltas, filter=filter)

    slice1 = slice(0, cutoff)
    slice2 = slice(cutoff, 100)
    data1 = data[slice1].copy()
    data2 = data[slice2].copy()
    c1.collect(data1, dates[slice1])
    c2.collect(data2, dates[slice2])

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = f"{tmpdir}/collector1.pkl"
        path2 = f"{tmpdir}/collector2.pkl"
        c1.serialise(path1, group=0, start=0, end=len(data1))
        c2.serialise(path2, group=1, start=len(data1), end=len(data))
        reloaded = StatisticsCollector.load_precomputed(DatasetMock(data, dates), [path1, path2])

    compare_statistics(reloaded.statistics(), c0_stats)
    compare_constants(reloaded.constant_variables(), c0_constants)


def compare_statistics(actual, expected):
    print("\nResults:")
    for stat_name in expected:
        exp = expected[stat_name]
        act = actual[stat_name]
        print(f"{stat_name.capitalize()}:")
        print(f"   Expected: {exp}")
        print(f"   Actual:   {act}")
        assert np.allclose(act, exp, rtol=1e-10, equal_nan=True)


def compare_constants(actual, expected):
    print("Constant Variables:")
    print(f"   Expected: {expected}")
    print(f"   Actual:   {actual}")
    assert set(actual) == set(expected), "Constant variables do not match!"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING STATISTICS COLLECTOR TEST SUITE")
    print("=" * 60)

    # Run all tests
    test_statistics_collector_single_batch(N=10_000_000, C=5)
    test_statistics_collector_multiple_batches(N=1_000_000, C=5, batch_size=10_000)
    test_statistics_with_nans(N=1_000_000, C=3)
    test_edge_cases()

    # Tendency tests
    test_tendencies_single_batch(N=1000, C=3, delta=1)
    test_tendencies_single_batch(N=1000, C=3, delta=5)
    test_tendencies_multiple_batches(N=1000, C=3, delta=1, batch_size=100)
    test_tendencies_multiple_batches(N=1000, C=3, delta=10, batch_size=50)
    test_tendencies_multiple_deltas(N=500, C=2)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
