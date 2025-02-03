# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import json
import logging
import pprint
import warnings
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOG = logging.getLogger(__name__)


def _tidy(v):
    if isinstance(v, (list, tuple, set)):
        return [_tidy(i) for i in v]
    if isinstance(v, dict):
        return {k: _tidy(v) for k, v in v.items()}
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

    if isinstance(v, np.integer):
        return int(v)

    return v


class Dataset:
    arguments = {}
    _name = None

    def mutate(self) -> "Dataset":
        """Give an opportunity to a subclass to return a new Dataset
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

        name = kwargs.pop("name", None)
        result = self.__subset(**kwargs)
        result._name = name

        return result

    @property
    def name(self):
        return self._name

    def __subset(self, **kwargs):
        if not kwargs:
            return self.mutate()

        # This one must be first
        if "fill_missing_dates" in kwargs:
            from .fill_missing import fill_missing_dates_factory

            fill_missing_dates = kwargs.pop("fill_missing_dates")
            ds = fill_missing_dates_factory(self, fill_missing_dates, kwargs)
            return ds._subset(**kwargs).mutate()

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

        if "number" in kwargs or "numbers" in kwargs or "member" in kwargs or "members" in kwargs:
            from .ensemble import Number

            members = {}
            for key in ["number", "numbers", "member", "members"]:
                if key in kwargs:
                    members[key] = kwargs.pop(key)

            return Number(self, **members)._subset(**kwargs).mutate()

        if "set_missing_dates" in kwargs:
            from .missing import MissingDates

            set_missing_dates = kwargs.pop("set_missing_dates")
            return MissingDates(self, set_missing_dates)._subset(**kwargs).mutate()

        if "skip_missing_dates" in kwargs:
            from .missing import SkipMissingDates

            if "expected_access" not in kwargs:
                raise ValueError("`expected_access` is required with `skip_missing_dates`")

            skip_missing_dates = kwargs.pop("skip_missing_dates")
            expected_access = kwargs.pop("expected_access")

            if skip_missing_dates:
                return SkipMissingDates(self, expected_access)._subset(**kwargs).mutate()

        if "interpolate_frequency" in kwargs:
            from .interpolate import InterpolateFrequency

            interpolate_frequency = kwargs.pop("interpolate_frequency")
            return InterpolateFrequency(self, interpolate_frequency)._subset(**kwargs).mutate()

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

        if not set(vars) <= set(self.name_to_index):
            raise ValueError(f"drop: unknown variables: {set(vars) - set(self.name_to_index)}")

        return sorted([v for k, v in self.name_to_index.items() if k not in vars])

    def _reorder_to_columns(self, vars):
        if isinstance(vars, str) and vars == "sort":
            # Sorting the variables alphabetically.
            # This is cruical for pre-training then transfer learning in combination with
            # cutout and adjust = 'all'

            indices = [self.name_to_index[k] for k, v in sorted(self.name_to_index.items(), key=lambda x: x[0])]
            assert set(indices) == set(range(len(self.name_to_index)))
            return indices

        if isinstance(vars, (list, tuple)):
            vars = {k: i for i, k in enumerate(vars)}

        indices = [self.name_to_index[k] for k, v in sorted(vars.items(), key=lambda x: x[1])]

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

    @property
    def typed_variables(self):
        from anemoi.transform.variables import Variable

        constants = self.constant_fields

        result = {}
        for k, v in self.variables_metadata.items():

            # TODO: Once all datasets are updated, we can remove this
            v = v.copy()
            if k in constants:
                v["constant_in_time"] = True

            if "is_constant_in_time" in v:
                del v["is_constant_in_time"]

            result[k] = Variable.from_dict(k, v)

        return result

    def _input_sources(self):
        sources = []
        self.collect_input_sources(sources)
        return sources

    def metadata(self):
        import anemoi

        _, source_to_arrays = self._supporting_arrays_and_sources()

        sources = []
        for i, source in enumerate(self._input_sources()):
            source_metadata = source.dataset_metadata().copy()
            source_metadata["supporting_arrays"] = source_to_arrays[id(source)]
            sources.append(source_metadata)

        md = dict(
            version=anemoi.datasets.__version__,
            arguments=self.arguments,
            **self.dataset_metadata(),
            sources=sources,
            supporting_arrays=source_to_arrays[id(self)],
        )

        try:
            return json.loads(json.dumps(_tidy(md)))
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
            variables_metadata=self.variables_metadata,
            shape=self.shape,
            dtype=str(self.dtype),
            start_date=self.start_date.astype(str),
            end_date=self.end_date.astype(str),
            name=self.name,
        )

    def _supporting_arrays(self, *path):

        import numpy as np

        def _path(path, name):
            return "/".join(str(_) for _ in [*path, name])

        result = {
            _path(path, "latitudes"): self.latitudes,
            _path(path, "longitudes"): self.longitudes,
        }
        collected = []

        self.collect_supporting_arrays(collected, *path)

        for path, name, array in collected:
            assert isinstance(path, tuple) and isinstance(name, str)
            assert isinstance(array, np.ndarray)

            name = _path(path, name)

            if name in result:
                raise ValueError(f"Duplicate key {name}")

            result[name] = array

        return result

    def supporting_arrays(self):
        """Arrays to be saved in the checkpoints"""
        arrays, _ = self._supporting_arrays_and_sources()
        return arrays

    def _supporting_arrays_and_sources(self):

        source_to_arrays = {}

        # Top levels arrays
        result = self._supporting_arrays()
        source_to_arrays[id(self)] = sorted(result.keys())

        # Arrays from the input sources
        for i, source in enumerate(self._input_sources()):
            name = source.name if source.name is not None else f"source{i}"
            src_arrays = source._supporting_arrays(name)
            source_to_arrays[id(source)] = sorted(src_arrays.keys())

            for k in src_arrays:
                assert k not in result

            result.update(src_arrays)

        return result, source_to_arrays

    def collect_supporting_arrays(self, collected, *path):
        # Override this method to add more arrays
        pass

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

    def computed_constant_fields(self):
        # Call `constant_fields` instead of `computed_constant_fields`
        try:
            # If the tendencies are computed, we can use them
            return sorted(self._compute_constant_fields_from_statistics())
        except KeyError:
            # This can happen if the tendencies are not computed
            pass

        return sorted(self._compute_constant_fields_from_a_few_samples())

    def _compute_constant_fields_from_a_few_samples(self):

        import numpy as np

        # Otherwise, we need to compute them
        dates = self.dates
        indices = set(range(len(dates)))
        indices -= self.missing

        sample_count = min(4, len(indices))
        count = len(indices)

        p = slice(0, count, count // max(1, sample_count - 1))
        samples = list(range(*p.indices(count)))

        samples.append(count - 1)  # Add last
        samples = sorted(set(samples))
        indices = list(indices)
        samples = [indices[i] for i in samples]

        assert set(samples) <= set(indices)  # Make sure we have the samples

        first = None
        constants = [True] * len(self.variables)

        first = self[samples.pop(0)]

        for sample in samples:
            row = self[sample]
            for i, (a, b) in enumerate(zip(row, first)):
                if np.any(a != b):
                    constants[i] = False

        return [v for i, v in enumerate(self.variables) if constants[i]]

    def _compute_constant_fields_from_statistics(self):
        result = []

        t = self.statistics_tendencies()

        for i, v in enumerate(self.variables):
            if t["mean"][i] == 0 and t["stdev"][i] == 0:
                result.append(v)

        return result

    def plot(self, date, variable, member=0, **kwargs):
        """For debugging purposes, plot a field.

        Parameters
        ----------
        date : int or datetime.datetime or numpy.datetime64 or str
            The date to plot.
        variable : int or str
            The variable to plot.
        member : int, optional
            The ensemble member to plot.

        **kwargs:
            Additional arguments to pass to matplotlib.pyplot.tricontourf


        Returns
        -------
            matplotlib.pyplot.Axes
        """

        from anemoi.utils.devtools import plot_values
        from earthkit.data.utils.dates import to_datetime

        if not isinstance(date, int):
            date = np.datetime64(to_datetime(date)).astype(self.dates[0].dtype)
            index = np.where(self.dates == date)[0]
            if len(index) == 0:
                raise ValueError(
                    f"Date {date} not found in the dataset {self.dates[0]} to {self.dates[-1]} by {self.frequency}"
                )
            date_index = index[0]
        else:
            date_index = date

        if isinstance(variable, int):
            variable_index = variable
        else:
            if variable not in self.variables:
                raise ValueError(f"Unknown variable {variable} (available: {self.variables})")

            variable_index = self.name_to_index[variable]

        values = self[date_index, variable_index, member]

        return plot_values(values, self.latitudes, self.longitudes, **kwargs)
