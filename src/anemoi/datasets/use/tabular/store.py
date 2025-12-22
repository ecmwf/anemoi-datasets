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

from anemoi.datasets.tabular.window import WindowView
from anemoi.datasets.use.dataset import Shape
from anemoi.datasets.use.store import ZarrStore

LOG = logging.getLogger(__name__)


class TabularZarr(ZarrStore):
    def __init__(self, group: zarr.hierarchy.Group, path: str = None) -> None:
        super().__init__(group, path=path)

        self._window_view = WindowView(self.z)

    def _subset(self, **kwargs):
        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency", None)
            self._window_view = self._window_view.set_frequency(frequency)

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            if start is not None:
                self._window_view = self._window_view.set_start(start)

            end = kwargs.pop("end", None)
            if end is not None:
                self._window_view = self._window_view.set_end(end)

        if "window" in kwargs:
            window = kwargs.pop("window", None)
            if window is not None:
                self._window_view = self._window_view.set_window(window)

        return super()._subset(**kwargs)

    def __getitem__(self, n):
        return self._window_view[n]

    def __len__(self) -> int:
        return len(self._window_view)

    @property
    def frequency(self) -> datetime.timedelta:
        return self._window_view.frequency

    @property
    def window(self) -> datetime.timedelta:
        return self._window_view.window

    def collect_input_sources(self, *args, **kwargs):
        raise NotImplementedError()

    def constant_fields(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def dates(self) -> NDArray[np.datetime64]:
        return self._window_view.dates

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def field_shape(self) -> Shape:
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

    def statistics_tendencies(self) -> dict[str, Any]:
        raise NotImplementedError()

    def tree(self) -> zarr.hierarchy.Group:
        raise NotImplementedError()

    def variables(self) -> list[str]:
        raise NotImplementedError()

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError()
