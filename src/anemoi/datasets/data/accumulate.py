# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import warnings
from abc import abstractmethod
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import debug_indexing
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import length_to_slices
from .indexing import update_tuple
from .indexing import index_to_slices
from .indexing import make_slice_or_index_from_list_or_tuple

from anemoi.utils.dates import frequency_to_timedelta, frequency_to_string

LOG = logging.getLogger(__name__)


class Accumulate(Dataset):
    """A class to represent a dataset that forwards its properties and methods to another dataset."""

    def __init__(self, forward: Dataset, param: str | List[str], accum_steps: int) -> None:
        """Initialize the Accumulate class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be accumulated.
        param : str
            Name of the parameter to be accumulated
        accum_steps : int
            Number of accumulation steps
        """
        if isinstance(param,str):
            param = [param]
        
        for p in param:
            assert p in forward.variables, f"Missing parameter {p} in original dataset, cannot accumulate"
                
        super().__init__(forward.__subset({"select": param}))
        
        self.forward = forward
        assert accum_steps > 0, f"Accumulation steps should be larger than 0, but accum_steps={accum_steps}"
        
        self.accum_steps = accum_steps
        
        

    def __len__(self) -> int:
        """Returns the length of the forward dataset.

        Returns
        -------
        int
            Length of the forward dataset.
        """
        return len(self.forward) // self.accum_steps

    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieve data from the accumulate dataset based on the given index.
           Perform accumulation on indices from the forward dataset.
        Parameters
        ----------
        n : Index
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data from the forward dataset based on the index.
        """
        
        if isinstance(n, tuple):
            return self._get_tuple(n)
            
        elif isinstance(n, slice):
            return self._get_slice(n)
        else:
            assert n>=0, n
            index_in_forward = self.accum_steps - 1 + n * self.accum_steps
            steps = [index_in_forward - i for i in range(self.accum_steps)]
        
            accum = np.nansum(self.forward[steps])
        
        return accum

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the interpolated data for a tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get the interpolated data for a slice.

        Parameters
        ----------
        s : slice
            The slice to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the slice.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @property
    def name(self) -> Optional[str]:
        """Returns the name of the forward dataset."""
        if self._name is not None:
            return self._name
        return self.forward.name

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Returns the dates of the dataset.
           These are a subset of the dates of the forwarded dataset"""
        
        return self.forward.dates[self.accum_steps-1::self.accum_steps]

    @property
    def resolution(self) -> str:
        """Returns the resolution of the forward dataset."""
        return self.forward.resolution

    @property
    def field_shape(self) -> Shape:
        """Returns the field shape of the forward dataset."""
        return self.forward.field_shape

    @property
    def frequency(self) -> datetime.timedelta:
        """Returns the frequency of the forward dataset."""
        return frequency_to_string(frequency_to_timedelta(self.forward.frequency) * self.accum_steps)

    @property
    def latitudes(self) -> NDArray[Any]:
        """Returns the latitudes of the forward dataset."""
        return self.forward.latitudes

    @property
    def longitudes(self) -> NDArray[Any]:
        """Returns the longitudes of the forward dataset."""
        return self.forward.longitudes

    @property
    def name_to_index(self) -> Dict[str, int]:
        """Returns a dictionary mapping variable names to their indices."""
        return self.forward.name_to_index

    @property
    def variables(self) -> List[str]:
        """Returns the variables of the forward dataset."""
        return self.forward.variables

    @property
    def variables_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the variables in the forward dataset."""
        return self.forward.variables_metadata

    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Returns the statistics of the forward dataset."""
        return self.forward.statistics

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Returns the statistics tendencies of the forward dataset.

        Parameters
        ----------
        delta : Optional[Any]
            Time delta for calculating tendencies.

        Returns
        -------
        Any
            Statistics tendencies of the forward dataset.
        """
        if delta is None:
            delta = self.frequency
        return self.forward.statistics_tendencies(delta)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the forward dataset."""
        return self.forward.shape

    @property
    def dtype(self) -> Any:
        """Returns the data type of the forward dataset."""
        return self.forward.dtype

    @property
    def missing(self) -> Set[int]:
        """Returns the missing data information."""
        
        missing = set()
        
        # for each missing data in the forward dataset
        # we must exclude the next accum_steps - 1
        # as accumulating will be impossible
        for original_missing in self.forward.missing:
            missing |= set(range(original_missing, original_missing + self.accum_steps)) 
        
        return missing

    @property
    def grids(self) -> Any:
        """Returns the grids of the forward dataset."""
        return self.forward.grids

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Returns metadata specific to the forward dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the forward dataset.
        """
        return super().metadata_specific(
            forward=self.forward.metadata_specific(),
            **self.forwards_subclass_metadata_specific(),
            **kwargs,
        )

    def collect_supporting_arrays(self, collected: List[Any], *path: Any) -> None:
        """Collects supporting arrays from the forward dataset.

        Parameters
        ----------
        collected : List[Any]
            List to which the supporting arrays are appended.
        *path : Any
            Variable length argument list specifying the paths for the arrays.
        """
        self.forward.collect_supporting_arrays(collected, *path)

    def collect_input_sources(self, collected: List[Any]) -> None:
        """Collects input sources from the forward dataset.

        Parameters
        ----------
        collected : List[Any]
            List to which the input sources are appended.
        """
        self.forward.collect_input_sources(collected)

    def source(self, index: int) -> Any:
        """Returns the source of the data at the specified index.

        Parameters
        ----------
        index : int
            Index specifying the data source.

        Returns
        -------
        Any
            Source of the data at the specified index.
        """
        return self.forward.source(index)

    @abstractmethod
    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Returns metadata specific to the subclass."""
        pass

    def get_dataset_names(self, names: Set[str]) -> None:
        """Collects the names of the datasets.

        Parameters
        ----------
        names : set
            Set to which the dataset names are added.
        """
        self.forward.get_dataset_names(names)

    @property
    def constant_fields(self) -> List[str]:
        """Returns the constant fields of the forward dataset."""
        return self.forward.constant_fields
    
def accumulate_factory(args: tuple, kwargs: dict) -> Dataset:
    """Create a accumulated dataset.

    Parameters
    ----------
    args : tuple
        The positional arguments.
    kwargs : dict
        The keyword arguments.

    Returns
    -------
    Dataset
        The joined dataset.
    """
    datasets = kwargs.pop("accumulate")
    
    assert len(args) == 0

    forward = _open(datasets.pop('forward'))

    accum_steps = datasets.pop('accum_steps')
    
    param = datasets.pop('param')

    return Accumulate(forward, accum_steps=accum_steps, param=param)
