# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging

import yaml
from earthkit.data.core.fieldlist import FieldList

from .field import EmptyFieldList
from .flavour import CoordinateGuesser
from .patch import patch_dataset
from .time import Time
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
    def from_xarray(cls, ds, *, flavour=None, patch=None):

        if patch is not None:
            ds = patch_dataset(ds, patch)

        variables = []

        if isinstance(flavour, str):
            with open(flavour) as f:
                if flavour.endswith(".yaml") or flavour.endswith(".yml"):
                    flavour = yaml.safe_load(f)
                else:
                    flavour = json.load(f)

        if isinstance(flavour, dict):
            flavour_coords = [coords["name"] for coords in flavour["rules"].values()]
            ds_dims = [dim for dim in ds._dims]
            for dim in ds_dims:
                if dim in flavour_coords and dim not in ds._coord_names:
                    ds = ds.assign_coords({dim: ds[dim]})
                else:
                    pass

        guess = CoordinateGuesser.from_flavour(ds, flavour)

        skip = set()

        def _skip_attr(v, attr_name):
            attr_val = getattr(v, attr_name, "")
            if isinstance(attr_val, str):
                skip.update(attr_val.split(" "))

        for name in ds.data_vars:
            variable = ds[name]
            _skip_attr(variable, "coordinates")
            _skip_attr(variable, "bounds")
            _skip_attr(variable, "grid_mapping")

        LOG.debug("Xarray data_vars: %s", ds.data_vars)

        # Select only geographical variables
        for name in ds.data_vars:

            if name in skip:
                continue

            variable = ds[name]
            coordinates = []

            for coord in variable.coords:

                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in variable.dims:
                    LOG.debug("%s: coord=%s (not a dimension): dims=%s", variable, coord, variable.dims)
                    c.is_dim = False
                coordinates.append(c)

            grid_coords = sum(1 for c in coordinates if c.is_grid and c.is_dim)
            assert grid_coords <= 2

            if grid_coords < 2:
                LOG.debug("Skipping %s (not 2D): %s", variable, [(c, c.is_grid, c.is_dim) for c in coordinates])
                continue

            v = Variable(
                ds=ds,
                variable=variable,
                coordinates=coordinates,
                grid=guess.grid(coordinates, variable),
                time=Time.from_coordinates(coordinates),
                metadata={},
            )

            variables.append(v)

        return cls(ds, variables)

    def sel(self, **kwargs) -> FieldList:
        """Override the FieldList's sel method

        Parameters
        ----------
        kwargs : dict
            The selection criteria

        Returns
        -------
        FieldList
            The new FieldList

        The algorithm is as follows:
        1 - Use the kwargs to select the variables that match the selection (`param` or `variable`)
        2 - For each variable, use the remaining kwargs to select the coordinates (`level`, `number`, ...)
        3 - Some mars like keys, like `date`, `time`, `step` are not found in the coordinates,
            but added to the metadata of the selected fields. A example is `step` that is added to the
            metadata of the field. Step 2 may return a variable that contain all the fields that
            verify at the same `valid_datetime`, with different base `date` and `time` and a different `step`.
            So we get an extra chance to filter the fields by the metadata.
        """

        variables = []
        count = 0

        for v in self.variables:

            # First, select matching variables
            # This will consume 'param' or 'variable' from kwargs
            # and return the rest
            match, rest = v.match(**kwargs)

            if match:
                count += 1
                missing = {}

                # Select from the variable's coordinates (time, level, number, ....)
                # This may return a new variable with a isel() slice of the selection
                # or None if the selection is not possible. In this case, missing is updated
                # with the values of kwargs (rest) that are not relevant for this variable
                v = v.sel(missing, **rest)
                if missing:
                    if v is not None:
                        # The remaining kwargs are passed used to create a FilteredVariable
                        # that will select 2D slices based on their metadata
                        v = FilteredVariable(v, **missing)
                    else:
                        LOG.warning(f"Variable {v} has missing coordinates: {missing}")

                if v is not None:
                    variables.append(v)

        if count == 0:
            LOG.warning("No variable found for %s", kwargs)
            LOG.warning("Variables: %s", sorted([v.name for v in self.variables]))

        if not variables:
            return EmptyFieldList()

        return self.__class__(self.ds, variables)
