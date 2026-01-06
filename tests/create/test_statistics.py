# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from anemoi.datasets.create.statistics import StatisticsCollector


def _create_random_stats(N, C):
    # 1. Define target means and stds for each column
    # We'll generate different targets for each column to ensure the test is thorough
    target_means = np.linspace(-10, 10, C)
    target_stds = np.linspace(1, 5, C)

    # 2. Generate raw random data
    data = np.random.randn(N, C)

    # 3. Adjust each column to have EXACTLY the target mean and std
    for i in range(C):
        col = data[:, i]
        # Standardize to mean=0, std=1
        col = (col - col.mean()) / col.std()
        # Scale and shift to targets
        data[:, i] = (col * target_stds[i]) + target_means[i]

    # 4. Capture the actual stats of the modified data
    # These will now be our "Ground Truth" for the unit test
    target_stats = {
        "mean": data.mean(axis=0),
        "stdev": data.std(axis=0),
        "minimum": data.min(axis=0),
        "maximum": data.max(axis=0),
    }

    return data, target_stats


def _check_statistics(data, target_stats):
    # Calculate stats from the data
    calc_mean = data.mean(axis=0)
    calc_std = data.std(axis=0)
    calc_min = data.min(axis=0)
    calc_max = data.max(axis=0)

    # Use np.isclose to handle floating point precision issues
    assert np.allclose(calc_mean, target_stats["mean"]), "Mean check failed"
    assert np.allclose(calc_std, target_stats["stdev"]), "Std deviation check failed"
    assert np.allclose(calc_min, target_stats["minimum"]), "Min check failed"
    assert np.allclose(calc_max, target_stats["maximum"]), "Max check failed"


def test_statistics_collector():
    N = 10000  # Number of rows
    C = 5  # Number of columns
    data, target_stats = _create_random_stats(N, C)
    _check_statistics(data, target_stats)

    collector = StatisticsCollector(variables_names=[f"var{i}" for i in range(C)])
    collector.collect(data, None)
    computed_stats = collector.statistics()

    for name, target in target_stats.items():
        print(f"{name.capitalize()}:")
        print(f"   - Target:   {target}")
        print(f"   - Computed: {computed_stats[name]}")
        assert np.allclose(computed_stats[name], target), f"{name.capitalize()} does not match!"


if __name__ == "__main__":
    test_statistics_collector()
