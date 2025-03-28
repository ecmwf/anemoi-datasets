# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..grids import cropping_mask
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Masked(Forwards):
    """A class to represent a masked dataset."""

    def __init__(self, forward: Dataset, mask: NDArray[np.bool_]) -> None:
        """Initialize the Masked class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be masked.
        mask : NDArray[np.bool_]
            The mask array.
        """
        super().__init__(forward)
        assert len(forward.shape) == 4, "Grids must be 1D for now"
        self.mask = mask
        self.axis = 3

        self.mask_name = f"{self.__class__.__name__.lower()}_mask"

    @cached_property
    def shape(self) -> Shape:
        """Get the shape of the masked dataset."""
        return self.forward.shape[:-1] + (np.count_nonzero(self.mask),)

    @cached_property
    def latitudes(self) -> NDArray[Any]:
        """Get the masked latitudes."""
        return self.forward.latitudes[self.mask]

    @cached_property
    def longitudes(self) -> NDArray[Any]:
        """Get the masked longitudes."""
        return self.forward.longitudes[self.mask]

    @debug_indexing
    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        """Get the masked data at the specified index.

        Parameters
        ----------
        index : FullIndex
            The index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The masked data at the specified index.
        """
        if isinstance(index, tuple):
            return self._get_tuple(index)

        result = self.forward[index]
        # We don't support subsetting the grid values
        assert result.shape[-1] == len(self.mask), (result.shape, len(self.mask))

        return result[..., self.mask]

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the masked data for a tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The masked data for the tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, self.axis, slice(None))
        result = self.forward[index]
        result = result[..., self.mask]
        result = result[..., previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def collect_supporting_arrays(self, collected: List[Tuple], *path: Any) -> None:
        """Collect supporting arrays.

        Parameters
        ----------
        collected : List[Tuple]
            The list to collect supporting arrays into.
        path : Any
            Additional path arguments.
        """
        super().collect_supporting_arrays(collected, *path)
        collected.append((path, self.mask_name, self.mask))


class Thinning(Masked):
    """A class to represent a thinned dataset."""

    def __init__(self, forward: Dataset, thinning: Optional[int], method: str) -> None:
        """Initialize the Thinning class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be thinned.
        thinning : Optional[int]
            The thinning factor.
        method : str
            The thinning method.
        """
        self.thinning = thinning
        self.method = method

        if thinning is not None:

            shape = forward.field_shape
            if len(shape) != 2:
                raise ValueError("Thinning only works latitude/longitude fields")

            # Make a copy, so we read the data only once from zarr
            forward_latitudes = forward.latitudes.copy()
            forward_longitudes = forward.longitudes.copy()

            latitudes = forward_latitudes.reshape(shape)
            longitudes = forward_longitudes.reshape(shape)
            latitudes = latitudes[::thinning, ::thinning].flatten()
            longitudes = longitudes[::thinning, ::thinning].flatten()

            # TODO: This is not very efficient

            mask = [lat in latitudes and lon in longitudes for lat, lon in zip(forward_latitudes, forward_longitudes)]
            mask = np.array(mask, dtype=bool)
        else:
            mask = None

        super().__init__(forward, mask)

    def mutate(self) -> Dataset:
        """Mutate the dataset.

        Returns
        -------
        Dataset
            The mutated dataset.
        """
        if self.thinning is None:
            return self.forward.mutate()
        return super().mutate()

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], thinning=self.thinning, method=self.method)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the Thinning subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the Thinning subclass.
        """
        return dict(thinning=self.thinning, method=self.method)


class Cropping(Masked):
    """A class to represent a cropped dataset."""

    def __init__(self, forward: Dataset, area: Union[Dataset, Tuple[float, float, float, float]]) -> None:
        """Initialize the Cropping class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be cropped.
        area : Union[Dataset, Tuple[float, float, float, float]]
            The cropping area.
        """
        from ..data import open_dataset

        area = area if isinstance(area, (list, tuple)) else open_dataset(area)

        if isinstance(area, Dataset):
            north = np.amax(area.latitudes)
            south = np.amin(area.latitudes)
            east = np.amax(area.longitudes)
            west = np.amin(area.longitudes)
            area = (north, west, south, east)

        self.area = area
        mask = cropping_mask(forward.latitudes, forward.longitudes, *area)

        super().__init__(forward, mask)

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], area=self.area)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the Cropping subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the Cropping subclass.
        """
        return dict(area=self.area)


class TrimEdge(Masked):
    """A class that removes the boundary of a dataset."""

    def __init__(self, forward, edge):
        if isinstance(edge, int):
            self.edge = [edge] * 4
        elif isinstance(edge, (list, tuple)) and len(edge) == 4:
            self.edge = edge
        else:
            raise ValueError("'edge' must be an integer or a list of 4 integers")

        for e in self.edge:
            if not isinstance(e, int) or e < 0:
                raise ValueError("'edge' must be integer(s) 0 or greater")

        shape = forward.field_shape
        if len(shape) != 2:
            raise ValueError("TrimEdge only works on regular grids")

        if self.edge[0] + self.edge[1] >= shape[0]:
            raise ValueError("Too much triming of the first grid dimension, resulting in an empty dataset")
        if self.edge[2] + self.edge[3] >= shape[1]:
            raise ValueError("Too much triming of the second grid dimension, resulting in an empty dataset")

        mask = np.full(shape, True, dtype=bool)
        mask[0 : self.edge[0], :] = False
        mask[:, 0 : self.edge[2]] = False
        if self.edge[1] != 0:
            mask[-self.edge[1] :, :] = False
        if self.edge[3] != 0:
            mask[:, -self.edge[3] :] = False

        mask = mask.flatten()

        super().__init__(forward, mask)

    def mutate(self) -> Dataset:
        """Mutate the dataset.

        Returns
        -------
        Dataset
            The mutated dataset.
        """
        if self.edge is None:
            return self.forward.mutate()
        return super().mutate()

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], edge=self.edge)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the TrimEdge subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the TrimEdge subclass.
        """
        return dict(edge=self.edge)

    @property
    def field_shape(self) -> Shape:
        """Returns the field shape of the dataset."""
        x, y = self.forward.field_shape
        x -= self.edge[0] + self.edge[1]
        y -= self.edge[2] + self.edge[3]
        return x, y
