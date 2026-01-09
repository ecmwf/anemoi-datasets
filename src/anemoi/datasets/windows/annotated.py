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

import numpy as np

LOG = logging.getLogger(__name__)


class AnnotatedNDArray(np.ndarray):
    """Extends numpy.ndarray to include additional metadata attributes.

    This class attaches a meta object that holds metadata information, allowing
    for the recommended way to add metadata to numpy arrays.
    """

    def __new__(cls, input_array, *, dtype=None, copy=False, meta=None) -> "AnnotatedNDArray":
        """Create a new AnnotatedNDArray with attached metadata.

        Parameters
        ----------
        input_array : array_like
            Input data to be converted to an AnnotatedNDArray.
        dtype : data-type, optional
            Desired data-type for the array.
        copy : bool, optional
            If True, then the object is copied.
        meta : object, optional
            Metadata object to attach to the array.

        Returns
        -------
        AnnotatedNDArray
            The new array with attached metadata.
        """
        obj = np.array(input_array, dtype=dtype, copy=copy).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj) -> None:
        """Finalise the creation of the AnnotatedNDArray, ensuring metadata is attached.

        Parameters
        ----------
        obj : object
            The source object from which the new array is derived.
        """
        if obj is None:
            return
        self.meta = getattr(obj, "meta", None)

    @property
    def dates(self) -> np.ndarray:
        """Array of dates associated with the data."""
        return self.meta.dates

    @property
    def latitudes(self) -> np.ndarray:
        """Array of latitudes associated with the data."""
        return self.meta.latitudes

    @property
    def longitudes(self) -> np.ndarray:
        """Array of longitudes associated with the data."""
        return self.meta.longitudes

    @property
    def timedeltas(self) -> np.ndarray:
        """Array of time deltas associated with the data."""
        return self.meta.timedeltas

    @property
    def reference_date(self) -> datetime.datetime:
        """The reference date for the data."""
        return self.meta.reference_date

    @property
    def boundaries(self) -> list[slice]:
        """The boundaries for the data."""
        return self.meta.boundaries

    @property
    def index(self) -> list[slice]:
        """The index for the data."""
        return self.meta.index
