# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

import numpy as np
import tqdm
import zarr

from anemoi.datasets import open_dataset

from . import Command


class Compare(Command):
    """Compare two datasets. This command compares the variables in two datasets and prints the mean of the common variables. It does not compare the data itself (yet)."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments are added.
        """
        command_parser.add_argument("dataset1")
        command_parser.add_argument("dataset2")
        command_parser.add_argument("--data", action="store_true", help="Compare the data.")
        command_parser.add_argument("--statistics", action="store_true", help="Compare the statistics.")

    def run(self, args: Any) -> None:
        """Run the compare command with the provided arguments.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """
        ds1 = open_dataset(args.dataset1)
        ds2 = open_dataset(args.dataset2)

        v1 = set(ds1.variables)
        v2 = set(ds2.variables)

        print("Only in dataset 1:", ", ".join(sorted(v1 - v2)))
        print("Only in dataset 2:", ", ".join(sorted(v2 - v1)))
        print()
        common = sorted(v1 & v2)
        print("Common:")
        print("-------")
        print()

        for v in common:
            print(
                f"{v:14}",
                f"{ds1.statistics['mean'][ds1.name_to_index[v]]:14g}",
                f"{ds2.statistics['mean'][ds2.name_to_index[v]]:14g}",
            )

        if args.data:
            print()
            print("Data:")
            print("-----")
            print()

            diff = 0
            for a, b in tqdm.tqdm(zip(ds1, ds2)):
                if not np.array_equal(a, b, equal_nan=True):
                    diff += 1

            print(f"Number of different rows: {diff}/{len(ds1)}")

        if args.data:
            print()
            print("Data 2:")
            print("-----")
            print()

            ds1 = zarr.open(args.dataset1, mode="r")
            ds2 = zarr.open(args.dataset2, mode="r")

            for name in (
                "data",
                "count",
                "sums",
                "squares",
                "mean",
                "stdev",
                "minimum",
                "maximum",
                "latitudes",
                "longitudes",
            ):
                a1 = ds1[name]
                a2 = ds2[name]

                if len(a1) != len(a2):
                    print(f"{name}: lengths mismatch {len(a1)} != {len(a2)}")
                    continue

                diff = 0
                for a, b in tqdm.tqdm(zip(a1, a2), leave=False):
                    if not np.array_equal(a, b, equal_nan=True):
                        if diff == 0:
                            print(f"\n{name}: first different row:")
                            print(a[a != b])
                            print(b[a != b])

                        diff += 1

                print(f"{name}: {diff} different rows out of {len(a1)}")


command = Compare
