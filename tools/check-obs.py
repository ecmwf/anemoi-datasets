#!/usr/bin/env python3
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging

import numpy as np

from anemoi.datasets import open_dataset

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="open two datasets and compare them")
    parser.add_argument("dataset", help="dataset to check")
    parser.add_argument("reference", help="reference dataset")
    args = parser.parse_args()
    compare(args.dataset, args.reference)


def _compare_nested_dicts(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_compare_nested_dicts(a[k], b[k]) for k in a)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return np.array_equal(a, b)
    assert False, f"Unsupported types for comparison: {type(a)} and {type(b)}"


def compare(input, reference):
    ds = open_dataset(input)
    ref = open_dataset(reference)

    if len(ds) != len(ref):
        raise ValueError(f"Datasets have different lengths: {len(ds)} != {len(ref)}")

    for i in range(len(ds)):
        if ds[i] != ref[i]:
            raise ValueError(f"Datasets differ at index {i}: {ds[i]} != {ref[i]}")
        if ds.dates[i] != ref.dates[i]:
            raise ValueError(f"Dates differ at index {i}: {ds.dates[i]} != {ref.dates[i]}")
    print("✅ Data and dates are identical")

    ds_metadata = ds.metadata.copy()
    ref_metadata = ref.metadata.copy()
    ds_metadata.pop("backend", None)
    ref_metadata.pop("backend", None)
    if ds_metadata != ref_metadata:
        raise ValueError("Metadata differs between datasets (excluding backend)")
    print("✅ Metadata is identical")

    if not _compare_nested_dicts(ds.statistics, ref.statistics):
        raise ValueError("Statistics differ between datasets")
    print("✅ Statistics are identical")


if __name__ == "__main__":
    main()
