# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import yaml
import os
import numpy as np
from anemoi.datasets.data import open_dataset


def open_multi_datasets(*datasets, **kwargs):
    if len(datasets) == 1 and datasets[0].endswith(".vz"):
        return open_vs_dataset(datasets[0], **kwargs)

    for d in datasets:
        assert not d.endswith(".vz"), f"mixing datasets type not implemented yet. {datasets}"

    return LegacyDatasets(datasets, **kwargs)


def open_vs_dataset(dataset, **kwargs):
    if not dataset.endswith(".vz"):
        raise ValueError("dataset must be a .vz file")
    return VzDatasets(dataset, **kwargs)


class LegacyDatasets:
    def __init__(self, paths, start=None, end=None, **kwargs):
        self.paths = paths

        if not start or not end:
            print("❌❌ Warning: start and end not provided, using the first and last dates of the datasets")
            lst = [self._open_dataset(p, **kwargs) for p in paths]
            start = min([d.dates[0] for d in lst])
            end = max([d.dates[-1] for d in lst])

        self._datasets = {
            os.path.basename(p).split(".")[0]: self._open_dataset(p, start=start, end=end, padding=True) for p in paths
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


class VzDatasets:
    _metadata = None

    def __init__(self, path, **kwargs):
        if kwargs:
            print("Warning: ignoring kwargs", kwargs)
        self.path = path
        self.keys = self.metadata["sources"].keys

    @property
    def metadata(self):
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    def items(self, *args, **kwargs):
        return {k: OneDataset(self, k) for k in self.keys()}.items(*args, **kwargs)

    def _load_metadata(self):
        if os.path.exists(os.path.join(self.path, "metadata.json")):
            with open(os.path.join(self.path, "metadata.json"), "r") as f:
                self._metadata = json.load(f)
            return
        with open(os.path.join(self.path, "recipe.yaml"), "r") as f:
            self._metadata = yaml.safe_load(f)

    def reopen_one_dataset(self):
        one_dataset_dict = list(self._metadata["sources"].values())[0]
        return open_dataset(**one_dataset_dict)

    def __len__(self):
        return len(self.reopen_one_dataset())

    @property
    def dates(self):
        return self.reopen_one_dataset().dates

    def __call__(self, i):
        if isinstance(i, str):
            return OneDataset(self, i)
        return LazyMultiElement(self, i)

    def __getitem__(self, i):
        path = os.path.join(self.path, "data", str(int(i / 10)), f"{i}.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))


class LazyMultiElement:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    @property
    def values(self):
        return self.dataset[self.n]

    def __getitem__(self, key):
        return self.values[key]


class OneDataset:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __getitem__(self, i):
        return self.dataset[i][self.name]
