# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Oracle: two-step read must never *silently diverge* from the eager reader.

The eager recursive ``__getitem__`` is the reference.  For every dataset type and
a matrix of edge-case indices (out-of-bounds, negative, empty, step, negative
step, single int per axis, cutout grid shards spanning the boundary, …) the
two-step path must give the **same result** and the **same error behaviour**
(raise iff eager raises).  This pins the class of bug where two-step returned
wrong data instead of raising (e.g. an out-of-range int silently wrapped via
``i % size``).
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

import anemoi.datasets.usage.read_parts as rp
from anemoi.datasets.usage.gridded.concat import Concat
from anemoi.datasets.usage.gridded.grids import Grids
from anemoi.datasets.usage.gridded.join import Join
from anemoi.datasets.usage.gridded.merge import Merge
from anemoi.datasets.usage.gridded.select import Select
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.gridded.subset import Subset

S = slice


def _group(n_dates=8, n_vars=4, n_grid=6, vars="abcd", seed=0,
           start=datetime.datetime(2021, 1, 1), lats=None, lons=None):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_vars, 1, n_grid)).astype(np.float32)
    root = zarr.group()
    root.create_array("data", data=data, compressor=None)
    freq = datetime.timedelta(hours=6)
    dates = np.array([start + i * freq for i in range(n_dates)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    root.create_array("latitudes", data=lats if lats is not None else np.linspace(-90, 90, n_grid), compressor=None)
    root.create_array("longitudes", data=lons if lons is not None else np.linspace(0, 360, n_grid, endpoint=False), compressor=None)
    for nm, v in (("mean", 0.0), ("stdev", 1.0), ("maximum", 1.0), ("minimum", 0.0)):
        root.create_array(nm, data=np.full(n_vars, v), compressor=None)
    var_list = list(vars[:n_vars])
    root.attrs.update({
        "frequency": frequency_to_string(freq), "resolution": "o96",
        "name_to_index": {v: i for i, v in enumerate(var_list)},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {v: {} for v in var_list},
        "field_shape": [1, n_grid],
    })
    return root


def _store(**k):
    return GriddedZarr(_group(**k), path="t.zarr")


def _eager(ds, n):
    rp.READ_PARTS_ENABLED = False
    try:
        return ("ok", np.asarray(ds[n]))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)
    finally:
        rp.READ_PARTS_ENABLED = True


def _two_step(ds, n):
    rp.READ_PARTS_ENABLED = True
    try:
        return ("ok", np.asarray(ds[n]))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)


def _assert_no_divergence(ds, indices):
    for n in indices:
        eager = _eager(ds, n)
        two = _two_step(ds, n)
        # Same error behaviour (raise iff eager raises).
        assert eager[0] == two[0], f"{n!r}: eager={eager} two-step={two}"
        if eager[0] == "ok":
            assert eager[1].shape == two[1].shape, f"{n!r}: shape {eager[1].shape} vs {two[1].shape}"
            np.testing.assert_array_equal(two[1], eager[1], err_msg=f"{n!r}")


# Edge-case index matrix (8 dates, 4 vars, 1 ens, 6 grid for the plain store).
_EDGES = [
    0, 3, 7, -1, -8,                                   # int incl negative
    8, 99, -9, -100,                                   # OOB int (must raise both)
    S(0, 4), S(2, 2), S(5, 3), S(None), S(None, None, 2), S(1, 8, 3), S(-3, None),
    S(None, None, -1), S(7, 0, -2),                    # negative step
    (0, S(None), S(None), S(None)), (-1, S(None), S(None), S(None)),
    (8, S(None), S(None), S(None)),                    # OOB int in tuple (the bug class)
    (S(0, 4), 0, S(None), S(None)), (S(0, 4), -1, S(None), S(None)),
    (S(None), 99, S(None), S(None)),                   # OOB var
    (S(None), S(None), S(None), 0), (S(None), S(None), S(None), -1),
    (S(None), S(None), S(None), 99),                   # OOB grid
    (S(0, 2), S(0, 2), S(None), S(None)),
]


def test_plain_store():
    _assert_no_divergence(_store(), _EDGES)


def test_select():
    _assert_no_divergence(Select(_store(), [0, 2], reason={"select": [0, 2]}), _EDGES)


def test_subset():
    _assert_no_divergence(Subset(_store(), [0, 2, 4, 6], reason={}), _EDGES)


def test_concat():
    ds = Concat([
        _store(n_dates=4, seed=0, start=datetime.datetime(2021, 1, 1)),
        _store(n_dates=4, seed=1, start=datetime.datetime(2021, 1, 2)),
    ])
    _assert_no_divergence(ds, _EDGES)


def test_join():
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        ds = Join([_store(n_vars=2, vars="ab", seed=0), _store(n_vars=2, vars="cd", seed=1)])
    _assert_no_divergence(ds, [0, -1, 7, 8, 99, S(0, 4), S(None), (0, S(None), S(None), S(None))])


def test_merge():
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        ds = Merge([
            _store(n_dates=4, seed=0, start=datetime.datetime(2021, 1, 1)),
            _store(n_dates=4, seed=1, start=datetime.datetime(2021, 1, 2)),
        ])
    _assert_no_divergence(ds, [0, 3, 7, -1, 8, 99, S(0, 8), S(2, 6)])


def test_grids_given_axis():
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        ds = Grids([_store(n_grid=3, seed=0), _store(n_grid=3, seed=1)], axis=3)
    _assert_no_divergence(ds, _EDGES)


def test_cutout_grid_edges():
    pytest.importorskip("anemoi.transform.spatial")
    from anemoi.datasets.usage.gridded.grids import Cutout

    lam = GriddedZarr(_group(n_vars=2, vars="ab", n_grid=5, seed=10,
                             lats=np.array([5.0, 10.0, 15.0, 5.0, 10.0]),
                             lons=np.array([5.0, 5.0, 5.0, 10.0, 10.0])), path="lam")
    globe = GriddedZarr(_group(n_vars=2, vars="ab", n_grid=20, seed=20,
                               lats=np.linspace(-80, 80, 20), lons=np.linspace(0, 350, 20)), path="globe")
    ds = Cutout([lam, globe], axis=3, cropping_distance=500.0)
    total = ds.shape[-1]
    lam_len = ds.lams[0].shape[-1]
    idxs = [
        0, -1, 3, S(0, 4), S(None),
        (0, S(None), S(None), S(0, 3)),                       # within LAM
        (0, S(None), S(None), S(lam_len - 1, lam_len + 2)),   # spans boundary
        (0, S(None), S(None), S(total - 3, total)),           # within globe
        (0, S(None), S(None), 0), (0, S(None), S(None), lam_len), (0, S(None), S(None), -1),
        (0, S(None), S(None), S(None, None, 2)), (0, S(None), S(None), S(2, 2)),
        (0, S(None), S(None), S(None, None, -1)),             # negative step grid
        (0, S(None), S(None), S(total - 1, total + 5)),       # over-stop
        (S(0, 2), S(0, 1), S(None), S(0, 4)),
        (8, S(None), S(None), S(None)),                       # OOB date
    ]
    _assert_no_divergence(ds, idxs)
