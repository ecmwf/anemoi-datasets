# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for opening datasets with ``open_zarr``."""

import re
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from anemoi.datasets.compat import DebugStore
from anemoi.datasets.compat import ZarrFileNotFoundError
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.usage import store as store_module
from anemoi.datasets.usage.store import open_zarr

DATA = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    """A dataset with a single "data" array, chunked by row."""
    path = str(tmp_path / "test.zarr")
    zarr_open(path, "w").create_array("data", data=DATA, chunks=(1, 6))
    return path


@pytest.fixture
def http_server(tmp_path: Path) -> Iterator[tuple[str, list[str]]]:
    """Serve ``tmp_path`` over HTTP. Yields the server URL and the list of requested paths."""
    requested = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            requested.append(self.path)
            super().do_GET()

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requested
    server.shutdown()
    server.server_close()


@pytest.mark.usefixtures("dataset")
def test_open_zarr_cache(http_server: tuple[str, list[str]]) -> None:
    url, requested = http_server

    chunks = {f"/test.zarr/data/{row}.0" for row in range(len(DATA))}

    root = open_zarr(f"{url}/test.zarr", cache=1_000_000)

    np.testing.assert_array_equal(root["data"][:], DATA)
    assert chunks <= set(requested)
    first_read = len(requested)
    np.testing.assert_array_equal(root["data"][:], DATA)

    # The second read gets the chunks from the cache. Keys that were not found, which
    # zarr-python 3 looks up again, are not cached.
    assert not chunks & set(requested[first_read:])


def test_open_zarr_cache_local_path(dataset: str) -> None:
    root = open_zarr(dataset, cache=1_000_000)
    np.testing.assert_array_equal(root["data"][:], DATA)


def test_open_zarr_debug_loading(dataset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "DEBUG_ZARR_LOADING", 1)

    root = open_zarr(dataset)

    assert isinstance(root.store, DebugStore)
    np.testing.assert_array_equal(root["data"][:], DATA)


def test_open_zarr_missing_store(tmp_path: Path) -> None:
    path = str(tmp_path / "missing.zarr")

    with pytest.raises(ZarrFileNotFoundError, match=re.escape(path)):
        open_zarr(path)
