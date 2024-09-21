# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
import pprint
import warnings
from functools import cached_property

from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOG = logging.getLogger(__name__)


class Dataset:
    arguments = {}

    def mutate(self) -> "Dataset":
        """
        Give an opportunity to a subclass to return a new Dataset
        object of a different class, if needed.
        """
        return self

    def swap_with_parent(self, parent):
        return parent

    @cached_property
    def _len(self):
        return len(self)

    def _subset(self, **kwargs):
        if not kwargs:
            return self.mutate()

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            end = kwargs.pop("end", None)

            from .subset import Subset

            return (
                Subset(self, self._dates_to_indices(start, end), dict(start=start, end=end))._subset(**kwargs).mutate()
            )

        if "frequency" in kwargs:
            from .subset import Subset

            if "interpolate_frequency" in kwargs:
                raise ValueError("Cannot use both `frequency` and `interpolate_frequency`")

            frequency = kwargs.pop("frequency")
            return (
                Subset(self, self._frequency_to_indices(frequency), dict(frequency=frequency))
                ._subset(**kwargs)
                .mutate()
            )

        if "interpolate_frequency" in kwargs:
            from .interpolate import InterpolateFrequency

            interpolate_frequency = kwargs.pop("interpolate_frequency")
            return InterpolateFrequency(self, interpolate_frequency)._subset(**kwargs).mutate()

        if "select" in kwargs:
            from .select import Select

            select = kwargs.pop("select")
            return Select(self, self._select_to_columns(select), {"select": select})._subset(**kwargs).mutate()

        if "drop" in kwargs:
            from .select import Select

            drop = kwargs.pop("drop")
            return Select(self, self._drop_to_columns(drop), {"drop": drop})._subset(**kwargs).mutate()

        if "reorder" in kwargs:
            from .select import Select

            reorder = kwargs.pop("reorder")
            return Select(self, self._reorder_to_columns(reorder), {"reoder": reorder})._subset(**kwargs).mutate()

        if "rename" in kwargs:
            from .select import Rename

            rename = kwargs.pop("rename")
            return Rename(self, rename)._subset(**kwargs).mutate()

        if "rescale" in kwargs:
            from .rescale import Rescale

            rescale = kwargs.pop("rescale")
            return Rescale(self, rescale)._subset(**kwargs).mutate()

        if "statistics" in kwargs:
            from ..data import open_dataset
            from .statistics import Statistics

            statistics = kwargs.pop("statistics")

            return Statistics(self, open_dataset(statistics))._subset(**kwargs).mutate()

        if "thinning" in kwargs:
            from .masked import Thinning

            thinning = kwargs.pop("thinning")
            method = kwargs.pop("method", "every-nth")
            return Thinning(self, thinning, method)._subset(**kwargs).mutate()

        if "area" in kwargs:
            from .masked import Cropping

            bbox = kwargs.pop("area")
            return Cropping(self, bbox)._subset(**kwargs).mutate()

        if "missing_dates" in kwargs:
            from .missing import MissingDates

            missing_dates = kwargs.pop("missing_dates")
            return MissingDates(self, missing_dates)._subset(**kwargs).mutate()

        if "skip_missing_dates" in kwargs:
            from .missing import SkipMissingDates

            if "expected_access" not in kwargs:
                raise ValueError("`expected_access` is required with `skip_missing_dates`")

            skip_missing_dates = kwargs.pop("skip_missing_dates")
            expected_access = kwargs.pop("expected_access")

            if skip_missing_dates:
                return SkipMissingDates(self, expected_access)._subset(**kwargs).mutate()

        # Keep last
        if "shuffle" in kwargs:
            from .subset import Subset

            shuffle = kwargs.pop("shuffle")

            if shuffle:
                return Subset(self, self._shuffle_indices(), dict(shuffle=True))._subset(**kwargs).mutate()

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def _frequency_to_indices(self, frequency):

        requested_frequency = frequency_to_seconds(frequency)
        dataset_frequency = frequency_to_seconds(self.frequency)
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
            if isinstance(v, (list, tuple, set)):
                return [tidy(i) for i in v]
            if isinstance(v, dict):
                return {k: tidy(v) for k, v in v.items()}
            if isinstance(v, str) and v.startswith("/"):
                return os.path.basename(v)
            if isinstance(v, datetime.datetime):
                return v.isoformat()
            if isinstance(v, datetime.date):
                return v.isoformat()
            if isinstance(v, datetime.timedelta):
                return frequency_to_string(v)

            if isinstance(v, Dataset):
                # That can happen in the `arguments`
                # if a dataset is passed as an argument
                return repr(v)

            if isinstance(v, slice):
                return (v.start, v.stop, v.step)

            return v

        md = dict(
            version=anemoi.datasets.__version__,
            arguments=self.arguments,
            **self.dataset_metadata(),
        )

        try:
            return json.loads(json.dumps(tidy(md)))
        except Exception:
            LOG.exception("Failed to serialize metadata")
            pprint.pprint(md)

            raise

    @property
    def start_date(self):
        return self.dates[0]

    @property
    def end_date(self):
        return self.dates[-1]

    def dataset_metadata(self):
        return dict(
            specific=self.metadata_specific(),
            frequency=self.frequency,
            variables=self.variables,
            shape=self.shape,
            start_date=self.start_date.astype(str),
            end_date=self.end_date.astype(str),
        )

    def metadata_specific(self, **kwargs):
        action = self.__class__.__name__.lower()
        # assert isinstance(self.frequency, datetime.timedelta), (self.frequency, self, action)
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            frequency=frequency_to_string(frequency_to_timedelta(self.frequency)),
            start_date=self.start_date.astype(str),
            end_date=self.end_date.astype(str),
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

    def _repr_html_(self):
        return self.tree().html()

    @property
    def label(self):
        return self.__class__.__name__.lower()

    def get_dataset_names(self, names):
        raise NotImplementedError(self)
