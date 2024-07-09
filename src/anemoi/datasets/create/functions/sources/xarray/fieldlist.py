# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from earthkit.data.core.fieldlist import FieldList

from .coordinates import extract_single_value
from .coordinates import is_scalar as is_scalar
from .field import EmptyFieldList
from .flavour import CoordinateGuesser
from .metadata import XArrayMetadata as XArrayMetadata
from .variable import FilteredVariable
from .variable import Variable

LOG = logging.getLogger(__name__)


class XarrayFieldList(FieldList):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables.copy()
        self.total_length = sum(v.length for v in variables)

    def __repr__(self):
        return f"XarrayFieldList({self.total_length})"

    def __len__(self):
        return self.total_length

    def __getitem__(self, i):
        k = i

        if i < 0:
            i = self.total_length + i

        for v in self.variables:
            if i < v.length:
                return v[i]
            i -= v.length

        raise IndexError(k)

    @classmethod
    def from_xarray(cls, ds, flavour=None):
        variables = []
        guess = CoordinateGuesser.from_flavour(ds, flavour)

        skip = set()

        def _skip_attr(v, attr_name):
            attr_val = getattr(v, attr_name, "")
            if isinstance(attr_val, str):
                skip.update(attr_val.split(" "))

        for name in ds.data_vars:
            v = ds[name]
            _skip_attr(v, "coordinates")
            _skip_attr(v, "bounds")
            _skip_attr(v, "grid_mapping")

        forecast_reference_time = None
        # Special variables
        for name in ds.data_vars:
            if name in skip:
                continue

            v = ds[name]
            if v.attrs.get("standard_name", "").lower() == "forecast_reference_time":
                forecast_reference_time = extract_single_value(v)
                continue

        # Select only geographical variables
        for name in ds.data_vars:

            if name in skip:
                continue

            v = ds[name]
            coordinates = []

            for coord in v.coords:

                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in v.dims:
                    c.is_dim = False
                coordinates.append(c)

            grid_coords = sum(1 for c in coordinates if c.is_grid and c.is_dim)
            assert grid_coords <= 2

            if grid_coords < 2:
                continue

            variables.append(
                Variable(
                    ds=ds,
                    var=v,
                    coordinates=coordinates,
                    grid=guess.grid(coordinates),
                    forecast_reference_time=forecast_reference_time,
                    metadata={},
                )
            )

        return cls(ds, variables)

    def sel(self, **kwargs):
        variables = []
        for v in self.variables:
            match, rest = v.match(**kwargs)

            if match:
                missing = {}
                v = v.sel(missing, **rest)
                if missing and v is not None:
                    v = FilteredVariable(v, **missing)

                if v is not None:
                    variables.append(v)

        if not variables:
            return EmptyFieldList()

        return self.__class__(self.ds, variables)
