# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``standard_name: height`` coordinate-guesser patch.

anemoi-datasets used to ship its own xarray coordinate guesser, which learnt
to recognise vertical coordinates declared with ``standard_name: height``
(#707).  That guesser was removed in favour of earthkit-data's, which only
matches ``long_name: height``, so ``_patch_height_level_coordinate`` reinstates
the rule.  These tests cover the patched behaviour and check the patch
delegates everything else unchanged.
"""

import numpy as np
import pytest
import xarray as xr

from anemoi.datasets.create.sources.xarray import _patch_height_level_coordinate


def _make_dataset(coord_name, standard_name, long_name, units, coord_length=5):
    attrs = {
        k: v for k, v in [("standard_name", standard_name), ("long_name", long_name), ("units", units)] if v is not None
    }
    return xr.Dataset(
        {"x_wind": ([coord_name], np.random.rand(coord_length))},
        coords={coord_name: xr.DataArray(np.arange(coord_length), dims=coord_name, attrs=attrs)},
    )


def _guess(ds, coord_name):
    _patch_height_level_coordinate()

    from earthkit.data.readers.xarray.flavour import DefaultCoordinateGuesser

    guesser = DefaultCoordinateGuesser(ds)
    return guesser.guess(ds[coord_name], coord_name)


@pytest.mark.parametrize(
    "coord_name, standard_name, long_name, units",
    [
        ("h", "height", None, "m"),  # the case earthkit-data misses (#707)
        ("h", None, "height", "m"),  # already handled by earthkit-data
        ("height", "height", "height", "m"),
    ],
)
def test_height_is_a_level_coordinate(coord_name, standard_name, long_name, units):
    from earthkit.data.readers.xarray.coordinates import LevelCoordinate

    ds = _make_dataset(coord_name, standard_name, long_name, units)
    guess = _guess(ds, coord_name)
    assert isinstance(guess, LevelCoordinate)
    assert guess.levtype == "height"


def test_other_level_coordinates_are_delegated():
    from earthkit.data.readers.xarray.coordinates import LevelCoordinate

    ds = _make_dataset("level", "air_pressure", None, "hPa")
    guess = _guess(ds, "level")
    assert isinstance(guess, LevelCoordinate)
    assert guess.levtype == "pl"


def test_height_without_units_is_not_a_level_coordinate():
    from earthkit.data.readers.xarray.coordinates import LevelCoordinate

    ds = _make_dataset("h", "height", None, None)
    guess = _guess(ds, "h")
    assert not isinstance(guess, LevelCoordinate)


def test_patch_is_idempotent():
    from earthkit.data.readers.xarray import flavour

    _patch_height_level_coordinate()
    patched = flavour.DefaultCoordinateGuesser._is_level
    _patch_height_level_coordinate()
    assert flavour.DefaultCoordinateGuesser._is_level is patched


def test_load_one_applies_the_patch():
    # ``load_one`` is the funnel every xarray-based source goes through; the
    # patch is applied there, lazily, so that importing the sources package
    # does not import xarray.
    import inspect

    from anemoi.datasets.create.sources.xarray import load_one

    source = inspect.getsource(load_one)
    assert "_patch_height_level_coordinate()" in source
