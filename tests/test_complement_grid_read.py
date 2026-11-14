# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

import numpy as np

from anemoi.datasets.usage.gridded.complement import ComplementNearest


class ArrayDataset:
    def __init__(self, values, variables):
        self.values = values
        self.variables = variables
        self.name_to_index = {name: index for index, name in enumerate(variables)}
        self.shape = values.shape
        self.dtype = values.dtype
        self.indices = []

    def __getitem__(self, index):
        self.indices.append(index)
        return self.values[index]


class TargetDataset:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def complement(source, nearest):
    result = ComplementNearest.__new__(ComplementNearest)
    result._source = source
    result._target = TargetDataset((2, 1, 1, len(nearest)), source.dtype)
    result._variables = ["added"]
    result._nearest_grid_points = np.asarray(nearest, dtype=np.int64)[:, None]
    result._distances = np.zeros((len(nearest), 1))
    result.k = 1
    return result


def test_nearest_reads_only_source_span_for_requested_target_points():
    values = np.arange(20, dtype=np.float32).reshape(2, 1, 1, 10)
    source = ArrayDataset(values, ["added"])
    dataset = complement(source, [2, 3, 4, 7, 8])

    result = dataset[:, :, :, 1:4]

    np.testing.assert_array_equal(result, values[:, :, :, [3, 4, 7]])
    assert source.indices[-1][-1] == slice(3, 8)


def test_nearest_preserves_nan_for_unreachable_target_points():
    values = np.arange(20, dtype=np.float32).reshape(2, 1, 1, 10)
    source = ArrayDataset(values, ["added"])
    dataset = complement(source, [2, 10, 7])

    result = dataset[:, :, :, :]

    np.testing.assert_array_equal(result[..., [0, 2]], values[..., [2, 7]])
    assert np.isnan(result[..., 1]).all()
    assert source.indices[-1][-1] == slice(2, 8)
