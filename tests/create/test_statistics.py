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

import numpy as np

from anemoi.datasets.create.statistics import StatisticsCollector


def _create_random_stats(N, C, nan_fraction=0.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

    # Adjust each column to have EXACTLY the target mean and std
    for i in range(C):
        col = data[:, i]
        # Standardize to mean=0, std=1
        col = (col - col.mean()) / col.std()
        # Scale and shift to targets
        data[:, i] = (col * target_stds[i]) + target_means[i]

    # Introduce NaNs if requested
    if nan_fraction > 0:
        nan_mask = np.random.rand(N, C) < nan_fraction
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
    assert np.allclose(calc_stats["mean"], target_stats["mean"]), "Mean check failed"
    assert np.allclose(calc_stats["stdev"], target_stats["stdev"]), "Std deviation check failed"
    assert np.allclose(calc_stats["minimum"], target_stats["minimum"]), "Min check failed"
    assert np.allclose(calc_stats["maximum"], target_stats["maximum"]), "Max check failed"


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
        assert np.allclose(computed_stats[name], target, rtol=1e-10), f"{name.capitalize()} does not match!"

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
        assert np.allclose(computed_stats[name], target, rtol=1e-10), f"{name.capitalize()} does not match!"

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
        assert np.allclose(computed_stats[name], target, rtol=1e-9), f"{name.capitalize()} does not match!"

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
        assert np.allclose(computed_stats[name], target_stats[name]), f"Single-row batch test failed for {name}"
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
        assert np.allclose(computed_stats2[name], target_stats[name]), f"Varying batch size test failed for {name}"
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
        assert np.allclose(computed, target, rtol=1e-9), f"Tendency {stat_name} does not match!"

    print("✓ Tendencies single batch test PASSED")


def test_tendencies_multiple_batches(N=1000, C=3, delta=1, batch_size=100):
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
        assert np.allclose(computed, target, rtol=1e-9), f"Tendency {stat_name} does not match!"

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
            assert np.allclose(computed, target, rtol=1e-9), f"Tendency {tendency_name}, {stat_name} does not match!"

    print("\n✓ Multiple tendencies simultaneously test PASSED")


def test_serialization():
    data, _ = _create_random_stats(1000, 3, nan_fraction=0.05)
    data2, _ = _create_random_stats(1000, 3, nan_fraction=0.05)

    c = StatisticsCollector(variables_names=["a", "b", "c"], allow_nans=True, tendencies={"delta_1": 1})

    c.collect(data, range(len(data)))
    target_stats = c.statistics()

    c2 = pickle.loads(pickle.dumps(c))
    deserialised_stats = c2.statistics()

    print("\nResults:")
    for stat_name, target in target_stats.items():
        print(f"{stat_name.capitalize()}:")
        print(f"   Target:   {target}")
        print(f"   Deserialised: {deserialised_stats[stat_name]}")
        assert np.allclose(
            deserialised_stats[stat_name], target, rtol=1e-10
        ), f"{stat_name.capitalize()} does not match after pickle!"


def test_merge_statistic_collectors():
    tendencies = {"delta_1": 1, "delta_5": 5}

    data, _ = _create_random_stats(1000, 2)
    c0 = StatisticsCollector(variables_names=["a", "b"], tendencies=tendencies)
    data = data.copy()
    c0.collect(data, range(len(data)))
    target = c0.statistics()

    c1 = StatisticsCollector(variables_names=["a", "b"], tendencies=tendencies)
    c2 = StatisticsCollector(variables_names=["a", "b"], tendencies=tendencies)

    data1 = data[:500].copy()
    data2 = data[500:].copy()
    c1.collect(data1, range(len(data1)))
    c2.collect(data2, range(len(data2)))

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = f"{tmpdir}/collector1.pkl"
        path2 = f"{tmpdir}/collector2.pkl"
        c1.serialise(path1, group=0, start=0, end=len(data1))
        c2.serialise(path2, group=1, start=len(data1), end=len(data))
        reloaded = StatisticsCollector.load_precomputed(data, [path1, path2], filter=None)
    merged = reloaded.statistics()

    print("\nResults:")
    for stat_name, target_values in target.items():
        merged_values = merged[stat_name]
        print(f"{stat_name.capitalize()}:")
        print(f"   Target:   {target_values}")
        print(f"   Merged:   {merged_values}")
        assert np.allclose(
            merged_values, target_values, rtol=1e-10
        ), f"{stat_name.capitalize()} does not match after merging!"
    print("✓ merge collectors test PASSED")


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
