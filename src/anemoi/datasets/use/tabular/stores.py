# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from anemoi.datasets.use.gridded.dataset import Dataset
from anemoi.datasets.use.gridded.dataset import Shape
from anemoi.datasets.use.gridded.stores import open_zarr

LOG = logging.getLogger(__name__)


class TabularZarr(Dataset):
    def __init__(self, path: str | zarr.hierarchy.Group, name: str = None) -> None:
        """Initialize the Zarr dataset with a path or zarr group."""
        if isinstance(path, zarr.hierarchy.Group):
            self.was_zarr = True
            self.path = name if name is not None else str(id(path))
            self.z = path
            self._name = name
        else:
            self.was_zarr = False
            self.path = str(path)
            self._name = name if name is not None else self.path
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z.data

    def __getitem__(self, n):
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def collect_input_sources(self, *args, **kwargs):
        raise NotImplementedError()

    def constant_fields(self, *args, **kwargs):
        raise NotImplementedError()

    def dates(self) -> list[datetime.datetime]:
        raise NotImplementedError()

    def dtype(self) -> np.dtype:
        raise NotImplementedError()

    def field_shape(self) -> Shape:
        raise NotImplementedError()

    def frequency(self) -> str:
        raise NotImplementedError()

    def get_dataset_names(self) -> list[str]:
        raise NotImplementedError()

    def latitudes(self) -> NDArray[np.float32]:
        raise NotImplementedError()

    def longitudes(self) -> NDArray[np.float32]:
        raise NotImplementedError()

    def missing(self) -> Any:
        raise NotImplementedError()

    def name_to_index(self) -> dict[str, int]:
        raise NotImplementedError()

    def resolution(self) -> tuple[float, float]:
        raise NotImplementedError()

    def shape(self) -> Shape:
        raise NotImplementedError()

    def source(self) -> str:
        raise NotImplementedError()

    def statistics(self) -> dict[str, Any]:
        raise NotImplementedError()

    def statistics_tendencies(self) -> dict[str, Any]:
        raise NotImplementedError()

    def tree(self) -> zarr.hierarchy.Group:
        raise NotImplementedError()

    def variables(self) -> list[str]:
        raise NotImplementedError()

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError()
