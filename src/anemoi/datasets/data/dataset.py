# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import warnings
from functools import cached_property

LOG = logging.getLogger(__name__)


class Dataset:
    arguments = {}

    @cached_property
    def _len(self):
        return len(self)

    def _subset(self, **kwargs):
        if not kwargs:
            return self

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            end = kwargs.pop("end", None)

            from .subset import Subset

            return Subset(self, self._dates_to_indices(start, end), dict(start=start, end=end))._subset(**kwargs)

        if "frequency" in kwargs:
            from .subset import Subset

            frequency = kwargs.pop("frequency")
            return Subset(self, self._frequency_to_indices(frequency), dict(frequency=frequency))._subset(**kwargs)

        if "select" in kwargs:
            from .select import Select

            select = kwargs.pop("select")
            return Select(self, self._select_to_columns(select), {"select": select})._subset(**kwargs)

        if "drop" in kwargs:
            from .select import Select

            drop = kwargs.pop("drop")
            return Select(self, self._drop_to_columns(drop), {"drop": drop})._subset(**kwargs)

        if "reorder" in kwargs:
            from .select import Select

            reorder = kwargs.pop("reorder")
            return Select(self, self._reorder_to_columns(reorder), {"reoder": reorder})._subset(**kwargs)

        if "rename" in kwargs:
            from .select import Rename

            rename = kwargs.pop("rename")
            return Rename(self, rename)._subset(**kwargs)

        if "statistics" in kwargs:
            from ..data import open_dataset
            from .statistics import Statistics

            statistics = kwargs.pop("statistics")

            return Statistics(self, open_dataset(statistics))._subset(**kwargs)

        if "thinning" in kwargs:
            from .masked import Thinning

            thinning = kwargs.pop("thinning")
            method = kwargs.pop("method", "every-nth")
            return Thinning(self, thinning, method)._subset(**kwargs)

        if "area" in kwargs:
            from .masked import Cropping

            bbox = kwargs.pop("area")
            return Cropping(self, bbox)._subset(**kwargs)

        # Keep last
        if "shuffle" in kwargs:
            from .subset import Subset

            shuffle = kwargs.pop("shuffle")

            if shuffle:
                return Subset(self, self._shuffle_indices(), dict(shuffle=True))._subset(**kwargs)

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def _frequency_to_indices(self, frequency):
        from .misc import _frequency_to_hours

        requested_frequency = _frequency_to_hours(frequency)
        dataset_frequency = _frequency_to_hours(self.frequency)
        assert requested_frequency % dataset_frequency == 0
        # Question: where do we start? first date, or first date that is a multiple of the frequency?
        step = requested_frequency // dataset_frequency

        return range(0, len(self), step)

    def _shuffle_indices(self):
        import numpy as np

        return np.random.permutation(len(self))

    def _dates_to_indices(self, start, end):
        from .misc import as_first_date
        from .misc import as_last_date

        # TODO: optimize

        start = self.dates[0] if start is None else as_first_date(start, self.dates)
        end = self.dates[-1] if end is None else as_last_date(end, self.dates)

        return [i for i, date in enumerate(self.dates) if start <= date <= end]

    def _select_to_columns(self, vars):
        if isinstance(vars, set):
            # We keep the order of the variables as they are in the zarr file
            nvars = [v for v in self.name_to_index if v in vars]
            assert len(nvars) == len(vars)
            return self._select_to_columns(nvars)

        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        return [self.name_to_index[v] for v in vars]

    def _drop_to_columns(self, vars):
        if not isinstance(vars, (list, tuple, set)):
            vars = [vars]

        assert set(vars) <= set(self.name_to_index)

        return sorted([v for k, v in self.name_to_index.items() if k not in vars])

    def _reorder_to_columns(self, vars):
        if isinstance(vars, (list, tuple)):
            vars = {k: i for i, k in enumerate(vars)}

        indices = []

        for k, v in sorted(vars.items(), key=lambda x: x[1]):
            indices.append(self.name_to_index[k])

        # Make sure we don't forget any variables
        assert set(indices) == set(range(len(self.name_to_index)))

        return indices

    def dates_interval_to_indices(self, start, end):
        return self._dates_to_indices(start, end)

    def provenance(self):
        return {}

    def sub_shape(self, drop_axis):
        shape = self.shape
        shape = list(shape)
        shape.pop(drop_axis)
        return tuple(shape)

    def metadata(self):
        import anemoi

        def tidy(v):
            if isinstance(v, (list, tuple)):
                return [tidy(i) for i in v]
            if isinstance(v, dict):
                return {k: tidy(v) for k, v in v.items()}
            if isinstance(v, str) and v.startswith("/"):
                return os.path.basename(v)
            return v

        return tidy(
            dict(
                version=anemoi.datasets.__version__,
                shape=self.shape,
                arguments=self.arguments,
                specific=self.metadata_specific(),
                frequency=self.frequency,
                variables=self.variables,
                start_date=self.dates[0].astype(str),
                end_date=self.dates[-1].astype(str),
            )
        )

    def metadata_specific(self, **kwargs):
        action = self.__class__.__name__.lower()
        assert isinstance(self.frequency, int), (self.frequency, self, action)
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            frequency=self.frequency,
            start_date=self.dates[0].astype(str),
            end_date=self.dates[-1].astype(str),
            **kwargs,
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"

    @property
    def grids(self):
        return (self.shape[-1],)

    def _check(ds):
        common = Dataset.__dict__.keys() & ds.__class__.__dict__.keys()
        overriden = [m for m in common if Dataset.__dict__[m] is not ds.__class__.__dict__[m]]

        for n in overriden:
            if n.startswith("_") and not n.startswith("__"):
                warnings.warn(f"Private method {n} is overriden in {ds.__class__.__name__}")

        # for n in ('metadata_specific', 'tree', 'source'):
        #     if n not in overriden:
        #         warnings.warn(f"Method {n} is not overriden in {ds.__class__.__name__}")

    def _repr_html_(self):
        return self.tree().html()

    @property
    def label(self):
        return self.__class__.__name__.lower()

    def get_dataset_names(self, names):
        raise NotImplementedError(self)
