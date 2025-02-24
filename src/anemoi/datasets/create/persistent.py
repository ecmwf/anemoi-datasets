# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import glob
import hashlib
import json
import logging
import os
import pickle
import shutil
import socket
from typing import Any
from typing import Iterator
from typing import Tuple

import numpy as np
from anemoi.utils.provenance import gather_provenance_info

LOG = logging.getLogger(__name__)


class PersistentDict:
    """A dictionary-like object that persists its contents to disk using pickle files.

    Attributes
    ----------
    version : int
        The version of the PersistentDict.
    dirname : str
        The directory where the data is stored.
    name : str
        The name of the directory.
    ext : str
        The extension of the directory.
    """

    version = 3

    # Used in parrallel, during data loading,
    # to write data in pickle files.
    def __init__(self, directory: str, create: bool = True):
        """Initialize the PersistentDict.

        Parameters
        ----------
        directory : str
            The directory where the data will be stored.
        create : bool, optional
            Whether to create the directory if it doesn't exist.
        """
        self.dirname = directory
        self.name, self.ext = os.path.splitext(os.path.basename(self.dirname))
        if create:
            self.create()

    def create(self) -> None:
        """Create the directory if it doesn't exist."""
        os.makedirs(self.dirname, exist_ok=True)

    def delete(self) -> None:
        """Delete the directory and its contents."""
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def __str__(self) -> str:
        """Return a string representation of the PersistentDict."""
        return f"{self.__class__.__name__}({self.dirname})"

    def items(self) -> Iterator[Any]:
        """Yield items stored in the directory.

        Yields
        ------
        Iterator[Any]
            An iterator over the items.
        """
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.pickle")
        LOG.debug(f"Reading {self.name} data, found {len(files)} files in {self.dirname}")
        assert len(files) > 0, f"No files found in {self.dirname}"
        for f in files:
            with open(f, "rb") as f:
                yield pickle.load(f)

    def add_provenance(self, **kwargs: Any) -> None:
        """Add provenance information to the directory.

        Parameters
        ----------
        **kwargs : Any
            Additional provenance information.
        """
        path = os.path.join(self.dirname, "provenance.json")
        if os.path.exists(path):
            return
        out = dict(provenance=gather_provenance_info(), **kwargs)
        with open(path, "w") as f:
            json.dump(out, f)

    def add(self, elt: Any, *, key: Any) -> None:
        """Add an element to the PersistentDict.

        Parameters
        ----------
        elt : Any
            The element to add.
        key : Any
            The key associated with the element.
        """
        self[key] = elt

    def __setitem__(self, key: Any, elt: Any) -> None:
        """Set an item in the PersistentDict.

        Parameters
        ----------
        key : Any
            The key associated with the element.
        elt : Any
            The element to set.
        """
        h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        path = os.path.join(self.dirname, f"{h}.pickle")

        if os.path.exists(path):
            LOG.warning(f"{path} already exists")

        tmp_path = path + f".tmp-{os.getpid()}-on-{socket.gethostname()}"
        with open(tmp_path, "wb") as f:
            pickle.dump((key, elt), f)
        shutil.move(tmp_path, path)

        LOG.debug(f"Written {self.name} data for len {key} in {path}")

    def flush(self) -> None:
        """Flush the PersistentDict (no-op)."""
        pass


class BufferedPersistentDict(PersistentDict):
    """A buffered version of PersistentDict that stores elements in memory before persisting them to disk.

    Attributes
    ----------
    buffer_size : int
        The size of the buffer.
    elements : list
        The list of elements in the buffer.
    keys : list
        The list of keys in the buffer.
    storage : PersistentDict
        The underlying PersistentDict used for storage.
    """

    def __init__(self, buffer_size: int = 1000, **kwargs: Any):
        """Initialize the BufferedPersistentDict.

        Parameters
        ----------
        buffer_size : int, optional
            The size of the buffer.
        **kwargs : Any
            Additional arguments for PersistentDict.
        """
        self.buffer_size = buffer_size
        self.elements = []
        self.keys = []
        self.storage = PersistentDict(**kwargs)

    def add(self, elt: Any, *, key: Any) -> None:
        """Add an element to the BufferedPersistentDict.

        Parameters
        ----------
        elt : Any
            The element to add.
        key : Any
            The key associated with the element.
        """
        self.elements.append(elt)
        self.keys.append(key)
        if len(self.keys) > self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush the buffer and store the elements in PersistentDict."""
        k = sorted(self.keys)
        self.storage.add(self.elements, key=k)
        self.elements = []
        self.keys = []

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield items stored in the BufferedPersistentDict.

        Yields
        ------
        Iterator[Tuple[Any, Any]]
            An iterator over the items.
        """
        for keys, elements in self.storage.items():
            for key, elt in zip(keys, elements):
                yield key, elt

    def delete(self) -> None:
        """Delete the storage directory and its contents."""
        self.storage.delete()

    def create(self) -> None:
        """Create the storage directory if it doesn't exist."""
        self.storage.create()


def build_storage(directory: str, create: bool = True) -> BufferedPersistentDict:
    """Build a BufferedPersistentDict storage.

    Parameters
    ----------
    directory : str
        The directory where the data will be stored.
    create : bool, optional
        Whether to create the directory if it doesn't exist.

    Returns
    -------
    BufferedPersistentDict
        The created BufferedPersistentDict.
    """
    return BufferedPersistentDict(directory=directory, create=create)


if __name__ == "__main__":
    N = 3
    P = 2
    directory = "h"
    p = PersistentDict(directory=directory)
    print(p)
    assert os.path.exists(directory)
    import numpy as np

    arrs = [np.random.randint(1, 101, size=(P,)) for _ in range(N)]
    dates = [np.array([np.datetime64(f"2021-01-0{_+1}") + np.timedelta64(i, "h") for i in range(P)]) for _ in range(N)]

    print()
    print("Writing the data")
    for i in range(N):
        _arr = arrs[i]
        _dates = dates[i]
        print(f"Writing : {i=}, {_arr=} {_dates=}")
        p[_dates] = (i, _arr)

    print()
    print("Reading the data back")

    p = PersistentDict(directory="h")
    for _dates, (i, _arr) in p.items():
        print(f"{i=}, {_arr=}, {_dates=}")

        assert np.allclose(_arr, arrs[i])

        assert len(_dates) == len(dates[i])
        for a, b in zip(_dates, dates[i]):
            assert a == b
