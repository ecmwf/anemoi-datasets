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
from functools import cached_property
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.store import ZarrStore
from anemoi.datasets.windows.view import WindowView

LOG = logging.getLogger(__name__)


class TabularZarr(ZarrStore):
    def __init__(self, group: zarr.hierarchy.Group, path: str = None) -> None:
        super().__init__(group, path=path)

        self._window_view = WindowView(self.store)

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

    @property
    def field_shape(self) -> Shape:
        raise NotImplementedError()

    def get_dataset_names(self) -> list[str]:
        raise NotImplementedError()

    @property
    def latitudes(self) -> np.ndarray:
        return None

    @property
    def longitudes(self) -> np.ndarray:
        return None

    def missing(self) -> Any:
        raise NotImplementedError()

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.variables)}

    @property
    def resolution(self) -> tuple[float, float]:
        return None

    @property
    def shape(self) -> Shape:
        return (None, len(self.variables))

    def source(self) -> str:
        raise NotImplementedError()

    def statistics_tendencies(self) -> dict[str, Any]:
        raise NotImplementedError()

    def tree(self) -> zarr.hierarchy.Group:
        raise NotImplementedError()

    @property
    def variables(self) -> list[str]:
        return [v for v in self.store.attrs["variables"] if not v.startswith("__")]

    def variables_metadata(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError()

    def usage_factory_load(self, name):
        # We need come here to get access to __package__
        return self._usage_factory_load(name, __package__)

    def plot_dates(self):
        return self._window_view.plot_dates()
