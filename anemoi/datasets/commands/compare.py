#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from anemoi.datasets import open_dataset

from . import Command


class Compare(Command):
    def add_arguments(self, command_parser):
        command_parser.add_argument("dataset1")
        command_parser.add_argument("dataset2")

    def run(self, args):
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


command = Compare
