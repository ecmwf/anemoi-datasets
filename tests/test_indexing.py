# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from anemoi.datasets.data.indexing import length_to_slices


def test_length_to_slices() -> None:
    """Test the length_to_slices function with various inputs."""
    lengths = [5, 7, 11, 13]
    datasets = [np.random.rand(n) for n in lengths]
    total = sum(lengths)

    combined = np.concatenate(datasets)

    for start in range(total):
        for stop in range(start, total):
            for step in range(1, stop - start + 1):
                index = slice(start, stop, step)
                print(index)
                slices = length_to_slices(index, lengths)
                result = [d[i] for (d, i) in zip(datasets, slices) if i is not None]
                result = np.concatenate(result)

                if (combined[index].shape != result.shape) or not (combined[index] == result).all():
                    print(index)
                    print(combined[index])
                    print(result)
                    print(slices)
                assert (combined[index] == result).all(), index


if __name__ == "__main__":
    test_length_to_slices()
