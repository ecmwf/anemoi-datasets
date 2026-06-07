# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Phase 5 tests: execute_parts (sequential).

execute_parts reads factorized parts one after another (cross-part threading was
removed — it is GIL-bound on decompress and slower for the few-large-chunks case;
within-array S3 fetch is already parallel in zarr/anemoi getitems).

Verifies that:
- empty parts list returns {}
- each factorized part's cached array matches a direct read of that part
- multi-source reads (Join / Concat) populate the buffer correctly
- two_step_read end-to-end matches ds[n]
"""

import datetime

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.gridded.concat import Concat as GriddedConcat
from anemoi.datasets.usage.gridded.join import Join
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import ReadPart, execute_parts, factorize, two_step_read

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 8
N_VARS = 4
N_ENS = 1
N_GRID = 6


def _make_zarr_group(
    n_dates=N_DATES, n_vars=N_VARS, n_ens=N_ENS, n_grid=N_GRID,
    vars="abcd", start_date=datetime.datetime(2021, 1, 1), seed=0,
):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_vars, n_ens, n_grid)).astype(np.float32)
    root = zarr.group()
    root.create_array("data", data=data, compressor=None)
    freq = datetime.timedelta(hours=6)
    dates = np.array([start_date + i * freq for i in range(n_dates)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    root.create_array("latitudes", data=np.linspace(-90, 90, n_grid), compressor=None)
    root.create_array("longitudes", data=np.linspace(0, 360, n_grid, endpoint=False), compressor=None)
    root.create_array("mean", data=np.zeros(n_vars), compressor=None)
    root.create_array("stdev", data=np.ones(n_vars), compressor=None)
    root.create_array("maximum", data=np.ones(n_vars), compressor=None)
    root.create_array("minimum", data=np.zeros(n_vars), compressor=None)
    var_list = list(vars[:n_vars])
    root.attrs.update({
        "frequency": frequency_to_string(freq),
        "resolution": "o96",
        "name_to_index": {v: i for i, v in enumerate(var_list)},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {v: {} for v in var_list},
        "field_shape": [1, n_grid],
    })
    return root


def _make_store(path="test.zarr", **kwargs):
    return GriddedZarr(_make_zarr_group(**kwargs), path=path)


# ---------------------------------------------------------------------------
# execute_parts: basic contract
# ---------------------------------------------------------------------------


def test_execute_parts_empty():
    assert execute_parts([]) == {}


def test_execute_parts_each_part_matches_direct_read():
    """Each cached array equals a direct read of that part."""
    store = _make_store()
    parts = store.collect_read_parts(0)
    factorized, _ = factorize(parts)
    buffer = execute_parts(factorized)
    assert set(buffer) == set(factorized)
    for part in factorized:
        np.testing.assert_array_equal(buffer[part], part.execute())


# ---------------------------------------------------------------------------
# execute_parts populates the buffer correctly for single + multi-source reads
# ---------------------------------------------------------------------------


def test_execute_parts_single_store():
    store = _make_store()
    for n in [0, 3, 7, slice(0, 4), slice(2, 8)]:
        factorized, _ = factorize(store.collect_read_parts(n))
        buffer = execute_parts(factorized)
        for part in factorized:
            np.testing.assert_array_equal(buffer[part], part.execute(), err_msg=f"n={n}")


def test_execute_parts_join():
    """Join produces one ReadPart per child store; all must be read."""
    from unittest.mock import patch
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        d1 = _make_store("j1.zarr", n_vars=2, vars="ab", seed=0)
        d2 = _make_store("j2.zarr", n_vars=2, vars="cd", seed=1)
        ds = Join([d1, d2])

    for n in [0, slice(0, 4)]:
        factorized, _ = factorize(ds.collect_read_parts(n))
        assert len(factorized) >= 2, "Join should produce at least 2 parts"
        buffer = execute_parts(factorized)
        for part in factorized:
            np.testing.assert_array_equal(buffer[part], part.execute())


def test_execute_parts_concat_boundary():
    """GriddedConcat spanning a boundary produces 2 parts."""
    from unittest.mock import patch
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        d1 = _make_store("c1.zarr", n_dates=4, seed=0, start_date=datetime.datetime(2021, 1, 1))
        d2 = _make_store("c2.zarr", n_dates=4, seed=1, start_date=datetime.datetime(2021, 1, 2))
        ds = GriddedConcat([d1, d2])

    factorized, _ = factorize(ds.collect_read_parts(slice(2, 6)))  # spans boundary
    assert len(factorized) == 2, "cross-boundary slice should produce 2 factorized parts"
    buffer = execute_parts(factorized)
    for part in factorized:
        np.testing.assert_array_equal(buffer[part], part.execute())


# ---------------------------------------------------------------------------
# End-to-end: two_step_read matches ds[n]
# ---------------------------------------------------------------------------


def test_end_to_end_equals_direct():
    store = _make_store()
    for n in [0, slice(0, 4), (slice(0, 4), slice(None), slice(None), slice(None))]:
        np.testing.assert_array_equal(two_step_read(store, n), store[n], err_msg=f"n={n}")


def test_execute_parts_takes_no_thread_argument():
    # The cross-part thread pool was removed; execute_parts is sequential and
    # takes only the parts list.
    import inspect

    from anemoi.datasets.usage import read_parts as rp

    assert list(inspect.signature(execute_parts).parameters) == ["parts"]
    assert not hasattr(rp, "READ_PARTS_THREADS")
