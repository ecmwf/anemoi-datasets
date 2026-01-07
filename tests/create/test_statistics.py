# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time

import numpy as np

from anemoi.datasets.create.statistics import StatisticsCollector


def _create_random_stats(N, C):
    """Generate random data with known statistics."""
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

    # Capture the actual stats of the modified data
    target_stats = {
        "mean": data.mean(axis=0),
        "stdev": data.std(axis=0),
        "minimum": data.min(axis=0),
        "maximum": data.max(axis=0),
    }

    return data, target_stats


def _check_statistics(data, target_stats):
    """Verify that computed statistics match target statistics."""
    calc_mean = data.mean(axis=0)
    calc_std = data.std(axis=0)
    calc_min = data.min(axis=0)
    calc_max = data.max(axis=0)

    assert np.allclose(calc_mean, target_stats["mean"]), "Mean check failed"
    assert np.allclose(calc_std, target_stats["stdev"]), "Std deviation check failed"
    assert np.allclose(calc_min, target_stats["minimum"]), "Min check failed"
    assert np.allclose(calc_max, target_stats["maximum"]), "Max check failed"


def test_statistics_collector_single_batch(N=1_000_000, C=5):
    """Test statistics collector with a single batch."""
    print(f"\n{'='*60}")
    print(f"TEST: Single batch - {N} rows, {C} columns")
    print(f"{'='*60}")

    print("Generating random data...")
    data, target_stats = _create_random_stats(N, C)

    print("Verifying target statistics...")
    _check_statistics(data, target_stats)

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
    print(f"\n{'='*60}")
    print(f"TEST: Multiple batches - {N} rows, {C} columns, batch size {batch_size}")
    print(f"{'='*60}")

    print("Generating random data...")
    data, target_stats = _create_random_stats(N, C)

    print("Verifying target statistics...")
    _check_statistics(data, target_stats)

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


def test_statistics_with_nans(N=100_000, C=3):
    """Test statistics collector with NaN values."""
    print(f"\n{'='*60}")
    print(f"TEST: Handling NaN values - {N} rows, {C} columns")
    print(f"{'='*60}")

    # Generate data with some NaN values
    data = np.random.randn(N, C)

    # Introduce NaNs randomly (about 10% of values)
    nan_mask = np.random.rand(N, C) < 0.1
    data[nan_mask] = np.nan

    print(f"Introduced {np.sum(nan_mask)} NaN values ({100*np.mean(nan_mask):.1f}%)")

    # Compute target statistics ignoring NaNs
    target_stats = {
        "mean": np.nanmean(data, axis=0),
        "stdev": np.nanstd(data, axis=0),
        "minimum": np.nanmin(data, axis=0),
        "maximum": np.nanmax(data, axis=0),
    }

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
    print(f"\n{'='*60}")
    print("TEST: Edge cases")
    print(f"{'='*60}")

    # Test with very small batches
    print("\n1. Testing with single-row batches...")
    data = np.random.randn(100, 3)
    target_stats = {
        "mean": data.mean(axis=0),
        "stdev": data.std(axis=0),
        "minimum": data.min(axis=0),
        "maximum": data.max(axis=0),
    }

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


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING STATISTICS COLLECTOR TEST SUITE")
    print("=" * 60)

    # Run all tests
    test_statistics_collector_single_batch(N=10_000_000, C=1)
    test_statistics_collector_multiple_batches(N=1_000_000, C=5, batch_size=10_000)
    test_statistics_with_nans(N=100_000, C=3)
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
