# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

from anemoi.datasets.data import open_dataset

LOG = logging.getLogger(__name__)


class LegacyDatasets:
    def __init__(self, paths, start=None, end=None, **kwargs):
        self.paths = paths

        if not start or not end:
            print(
                "❌❌ Warning: start and end not provided, using the minima first and maximal last dates of the datasets"
            )
            lst = [self._open_dataset(p, **kwargs) for p in paths]
            start = min([d.dates[0] for d in lst])
            end = max([d.dates[-1] for d in lst])

        self._datasets = {
            os.path.basename(p).split(".")[0]: self._open_dataset(p, start=start, end=end, padding="empty")
            for p in paths
        }

        first = list(self._datasets.values())[0]
        for name, dataset in self._datasets.items():
            if dataset.dates[0] != first.dates[0] or dataset.dates[-1] != first.dates[-1]:
                raise ValueError("Datasets have different start and end times")
            if dataset.frequency != first.frequency:
                raise ValueError("Datasets have different frequencies")

        self._keys = self._datasets.keys

        self._first = list(self._datasets.values())[0]

    def _open_dataset(self, p, **kwargs):
        if p.startswith("observations-"):
            return open_dataset(observations=p, **kwargs)
        else:
            print("❗ Opening non-observations dataset:", p)
            return open_dataset(p, **kwargs)

    def items(self):
        return self._datasets.items()

    @property
    def dates(self):
        return self._first.dates

    def __len__(self):
        return len(self._first)

    def __getitem__(self, i):
        return {k: d[i] for k, d in self._datasets.items()}
