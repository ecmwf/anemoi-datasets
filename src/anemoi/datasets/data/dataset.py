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
from abc import ABC
from abc import abstractmethod
from functools import cached_property

try:
    from types import EllipsisType
except ImportError:
    # Python 3.9
    EllipsisType = type(Ellipsis)
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as np
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from .debug import Node
from .debug import Source

if TYPE_CHECKING:
    import matplotlib

LOG = logging.getLogger(__name__)


Shape = Tuple[int, ...]
TupleIndex = Tuple[Union[int, slice, EllipsisType], ...]
FullIndex = Union[int, slice, TupleIndex]


def _tidy(v: Any) -> Any:
    """Tidy up the input value.

    Parameters
    ----------
    v : Any
        The input value to tidy up.

    Returns
    -------
    Any
        The tidied value.
    """
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


class Dataset(ABC, Sized):
    arguments: Dict[str, Any] = {}
    _name: Union[str, None] = None

    def mutate(self) -> "Dataset":
        """Give an opportunity to a subclass to return a new Dataset object of a different class, if needed.

        Returns
        -------
        Dataset
            The mutated dataset.
        """
        return self

    def swap_with_parent(self, parent: "Dataset") -> "Dataset":
        """Swap the current dataset with its parent dataset.

        Parameters
        ----------
        parent : Dataset
            The parent dataset.

        Returns
        -------
        Dataset
            The parent dataset.
        """
        return parent

    @cached_property
    def _len(self) -> int:
        """Cache and return the length of the dataset."""
        return len(self)

    def _subset(self, **kwargs: Any) -> "Dataset":
        """Create a subset of the dataset based on the provided keyword arguments.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for creating the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        if not kwargs:
            return self.mutate()

        name = kwargs.pop("name", None)
        result = self.__subset(**kwargs)
        result._name = name

        return result

    @property
    def name(self) -> Union[str, None]:
        """Return the name of the dataset."""
        return self._name

    def __subset(self, **kwargs: Any) -> "Dataset":
        """Internal method to create a subset of the dataset based on the provided keyword arguments.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for creating the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
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

        # Note: trim_edge should go before thinning
        if "trim_edge" in kwargs:
            from .masked import TrimEdge

            edge = kwargs.pop("trim_edge")
            return TrimEdge(self, edge)._subset(**kwargs).mutate()

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

        if "interpolate_variables" in kwargs:
            from .interpolate import InterpolateNearest

            interpolate_variables = kwargs.pop("interpolate_variables")
            max_distance = kwargs.pop("max_distance", None)
            return InterpolateNearest(self, interpolate_variables, max_distance=max_distance)._subset(**kwargs).mutate()

        # Keep last
        if "shuffle" in kwargs:
            from .subset import Subset

            shuffle = kwargs.pop("shuffle")

            if shuffle:
                return Subset(self, self._shuffle_indices(), dict(shuffle=True))._subset(**kwargs).mutate()

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def _frequency_to_indices(self, frequency: str) -> list[int]:
        """Convert a frequency string to a list of indices.

        Parameters
        ----------
        frequency : str
            The frequency string.

        Returns
        -------
        list of int
            The list of indices.
        """
        requested_frequency = frequency_to_seconds(frequency)
        dataset_frequency = frequency_to_seconds(self.frequency)

        if requested_frequency % dataset_frequency != 0:
            raise ValueError(
                f"Requested frequency {frequency} is not a multiple of the dataset frequency {self.frequency}. Did you mean to use `interpolate_frequency`?"
            )

        # Question: where do we start? first date, or first date that is a multiple of the frequency?
        step = requested_frequency // dataset_frequency

        return range(0, len(self), step)

    def _shuffle_indices(self) -> NDArray[Any]:
        """Return a shuffled array of indices.

        Returns
        -------
        numpy.ndarray
            The shuffled array of indices.
        """
        return np.random.permutation(len(self))

    def _dates_to_indices(
        self,
        start: Union[None, str, datetime.datetime],
        end: Union[None, str, datetime.datetime],
    ) -> List[int]:
        """Convert date range to a list of indices.

        Parameters
        ----------
        start : None, str, or datetime.datetime
            The start date.
        end : None, str, or datetime.datetime
            The end date.

        Returns
        -------
        list of int
            The list of indices.
        """
        from .misc import as_first_date
        from .misc import as_last_date

        # TODO: optimize

        start = self.dates[0] if start is None else as_first_date(start, self.dates)
        end = self.dates[-1] if end is None else as_last_date(end, self.dates)

        return [i for i, date in enumerate(self.dates) if start <= date <= end]

    def _select_to_columns(self, vars: Union[str, List[str], Tuple[str], set]) -> List[int]:
        """Convert variable names to a list of column indices.

        Parameters
        ----------
        vars : str, list of str, tuple of str, or set
            The variable names.

        Returns
        -------
        list of int
            The list of column indices.
        """
        if isinstance(vars, set):
            # We keep the order of the variables as they are in the zarr file
            nvars = [v for v in self.name_to_index if v in vars]
            assert len(nvars) == len(vars)
            return self._select_to_columns(nvars)

        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        return [self.name_to_index[v] for v in vars]

    def _drop_to_columns(self, vars: Union[str, Sequence[str]]) -> List[int]:
        """Convert variable names to a list of column indices to drop.

        Parameters
        ----------
        vars : str, list of str, tuple of str, or set
            The variable names.

        Returns
        -------
        list of int
            The list of column indices to drop.
        """
        if not isinstance(vars, (list, tuple, set)):
            vars = [vars]

        if not set(vars) <= set(self.name_to_index):
            raise ValueError(f"drop: unknown variables: {set(vars) - set(self.name_to_index)}")

        return sorted([v for k, v in self.name_to_index.items() if k not in vars])

    def _reorder_to_columns(self, vars: Union[str, List[str], Tuple[str], Dict[str, int]]) -> List[int]:
        """Convert variable names to a list of reordered column indices.

        Parameters
        ----------
        vars : str, list of str, tuple of str, or dict of str to int
            The variable names.

        Returns
        -------
        list of int
            The list of reordered column indices.
        """
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

    def dates_interval_to_indices(
        self, start: Union[None, str, datetime.datetime], end: Union[None, str, datetime.datetime]
    ) -> List[int]:
        """Convert date interval to a list of indices.

        Parameters
        ----------
        start : None, str, or datetime.datetime
            The start date.
        end : None, str, or datetime.datetime
            The end date.

        Returns
        -------
        list of int
            The list of indices.
        """
        return self._dates_to_indices(start, end)

    def provenance(self) -> Dict[str, Any]:
        """Return the provenance information of the dataset.

        Returns
        -------
        dict
            The provenance information.
        """
        return {}

    def sub_shape(self, drop_axis: int) -> TupleIndex:
        """Return the shape of the dataset with one axis dropped.

        Parameters
        ----------
        drop_axis : int
            The axis to drop.

        Returns
        -------
        tuple
            The shape with one axis dropped.
        """
        shape = list(self.shape)
        shape.pop(drop_axis)
        return tuple(shape)

    @property
    def typed_variables(self) -> Dict[str, Any]:
        """Return the variables with their types."""
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

    def _input_sources(self) -> List[Any]:
        """Return the input sources of the dataset.

        Returns
        -------
        list
            The input sources.
        """
        sources = []
        self.collect_input_sources(sources)
        return sources

    def metadata(self) -> Dict[str, Any]:
        """Return the metadata of the dataset.

        Returns
        -------
        dict
            The metadata.
        """
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
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    def dataset_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the dataset.

        Returns
        -------
        dict
            The metadata.
        """
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

    def _supporting_arrays(self, *path: str) -> Dict[str, NDArray[Any]]:
        """Return the supporting arrays of the dataset.

        Parameters
        ----------
        *path : str
            The path components.

        Returns
        -------
        dict
            The supporting arrays.
        """

        def _path(path, name: str) -> str:
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

    def supporting_arrays(self) -> Dict[str, NDArray[Any]]:
        """Return the supporting arrays to be saved in the checkpoints.

        Returns
        -------
        dict
            The supporting arrays.
        """
        arrays, _ = self._supporting_arrays_and_sources()
        return arrays

    def _supporting_arrays_and_sources(self) -> Tuple[Dict[str, NDArray], Dict[int, List[str]]]:
        """Return the supporting arrays and their sources.

        Returns
        -------
        tuple
            The supporting arrays and their sources.
        """
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

    def collect_supporting_arrays(self, collected: List[Tuple[Tuple[str, ...], str, NDArray[Any]]], *path: str) -> None:
        """Collect supporting arrays.

        Parameters
        ----------
        collected : list of tuple
            The collected supporting arrays.
        *path : str
            The path components.
        """
        # Override this method to add more arrays
        pass

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Return specific metadata of the dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        dict
            The specific metadata.
        """
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

    def __repr__(self) -> str:
        """Return the string representation of the dataset.

        Returns
        -------
        str
            The string representation.
        """
        return self.__class__.__name__ + "()"

    @property
    def grids(self) -> TupleIndex:
        """Return the grid shape of the dataset."""
        return (self.shape[-1],)

    def _check(self) -> None:
        """Check for overridden private methods in the dataset."""
        common = Dataset.__dict__.keys() & self.__class__.__dict__.keys()
        overriden = [m for m in common if Dataset.__dict__[m] is not self.__class__.__dict__[m]]

        for n in overriden:
            if n.startswith("_") and not n.startswith("__") and n not in ("_abc_impl",):
                warnings.warn(f"Private method {n} is overriden in {self.__class__.__name__}")

    def _repr_html_(self) -> str:
        """Return the HTML representation of the dataset.

        Returns
        -------
        str
            The HTML representation.
        """
        return self.tree().html()

    @property
    def label(self) -> str:
        """Return the label of the dataset."""
        return self.__class__.__name__.lower()

    def computed_constant_fields(self) -> List[str]:
        """Return the computed constant fields of the dataset.

        Returns
        -------
        list of str
            The computed constant fields.
        """
        try:
            # If the tendencies are computed, we can use them
            return sorted(self._compute_constant_fields_from_statistics())
        except KeyError:
            # This can happen if the tendencies are not computed
            pass

        return sorted(self._compute_constant_fields_from_a_few_samples())

    def _compute_constant_fields_from_a_few_samples(self) -> List[str]:
        """Compute constant fields from a few samples.

        Returns
        -------
        list of str
            The computed constant fields.
        """
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

    def _compute_constant_fields_from_statistics(self) -> List[str]:
        """Compute constant fields from statistics.

        Returns
        -------
        list of str
            The computed constant fields.
        """
        result = []

        t = self.statistics_tendencies()

        for i, v in enumerate(self.variables):
            if t["mean"][i] == 0 and t["stdev"][i] == 0:
                result.append(v)

        return result

    def plot(
        self,
        date: Union[int, datetime.datetime, np.datetime64, str],
        variable: Union[int, str],
        member: int = 0,
        **kwargs: Any,
    ) -> "matplotlib.pyplot.Axes":
        """For debugging purposes, plot a field.

        Parameters
        ----------
        date : int or datetime.datetime or numpy.datetime64 or str
            The date to plot.
        variable : int or str
            The variable to plot.
        member : int, optional
            The ensemble member to plot.
        **kwargs : Any
            Additional arguments to pass to matplotlib.pyplot.tricontourf.

        Returns
        -------
        matplotlib.pyplot.Axes
            The plot axes.
        """
        from anemoi.utils.devtools import plot_values

        values = self[self.to_index(date, variable, member)]

        return plot_values(values, self.latitudes, self.longitudes, **kwargs)

    def to_index(
        self,
        date: Union[int, datetime.datetime, np.datetime64, str],
        variable: Union[int, str],
        member: int = 0,
    ) -> Tuple[int, int, int]:
        """Convert date, variable, and member to indices.

        Parameters
        ----------
        date : int or datetime.datetime or numpy.datetime64 or str
            The date.
        variable : int or str
            The variable.
        member : int, optional
            The ensemble member.

        Returns
        -------
        tuple of int
            The indices.
        """
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

        date_index = int(date_index)  # because np.int64 is not instance of int

        if isinstance(variable, int):
            variable_index = variable
        else:
            if variable not in self.variables:
                raise ValueError(f"Unknown variable {variable} (available: {self.variables})")

            variable_index = self.name_to_index[variable]

        return (date_index, variable_index, member)

    @abstractmethod
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the item at the specified index.

        Parameters
        ----------
        n : FullIndex
            Index to retrieve.

        Returns
        -------
        NDArray[Any]
            Retrieved item.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """Return the list of variables in the dataset."""
        pass

    @property
    @abstractmethod
    def frequency(self) -> datetime.timedelta:
        """Return the frequency of the dataset."""
        pass

    @property
    @abstractmethod
    def dates(self) -> NDArray[np.datetime64]:
        """Return the dates in the dataset."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> str:
        """Return the resolution of the dataset."""
        pass

    @property
    @abstractmethod
    def name_to_index(self) -> Dict[str, int]:
        """Return the mapping of variable names to indices."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Return the shape of the dataset."""
        pass

    @property
    @abstractmethod
    def field_shape(self) -> Shape:
        """Return the shape of the fields in the dataset."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Return the data type of the dataset."""
        pass

    @property
    @abstractmethod
    def latitudes(self) -> NDArray[Any]:
        """Return the latitudes in the dataset."""
        pass

    @property
    @abstractmethod
    def longitudes(self) -> NDArray[Any]:
        """Return the longitudes in the dataset."""
        pass

    @property
    @abstractmethod
    def variables_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the variables in the dataset."""
        pass

    @abstractmethod
    @cached_property
    def missing(self) -> Set[int]:
        """Return the set of missing indices in the dataset."""
        pass

    @abstractmethod
    @cached_property
    def constant_fields(self) -> List[str]:
        """Return the list of constant fields in the dataset."""
        pass

    @abstractmethod
    @cached_property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Return the statistics of the dataset."""
        pass

    @abstractmethod
    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Return the tendencies of the statistics in the dataset.

        Parameters
        ----------
        delta : datetime.timedelta, optional
            The time delta for computing tendencies.

        Returns
        -------
        dict
            The tendencies.
        """
        pass

    @abstractmethod
    def source(self, index: int) -> Source:
        """Return the source of the dataset at the specified index.

        Parameters
        ----------
        index : int
            The index.

        Returns
        -------
        Source
            The source.
        """
        pass

    @abstractmethod
    def tree(self) -> Node:
        """Return the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation.
        """
        pass

    @abstractmethod
    def collect_input_sources(self, sources: List[Any]) -> None:
        """Collect the input sources of the dataset.

        Parameters
        ----------
        sources : list
            The input sources.
        """
        pass

    @abstractmethod
    def get_dataset_names(self, names: Set[str]) -> None:
        """Get the names of the datasets.

        Parameters
        ----------
        names : set of str
            The dataset names.
        """
        pass
