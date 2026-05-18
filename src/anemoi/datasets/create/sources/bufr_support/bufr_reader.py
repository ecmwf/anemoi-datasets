# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import mmap
from contextlib import contextmanager

import eccodes
import numpy as np
from gribapi.errors import KeyValueNotFoundError


class BUFRMessage:
    def __init__(self, bid: int):
        self.bid = bid

    def get_array(self, element: str, typ: type, nsubsets: int, missing_val=np.nan) -> np.ndarray:
        """Wrapper for codes_get_array to work around the inconsistent handling of arrays in eccodes when data is constant"""
        try:
            arr = eccodes.codes_get_array(self.bid, element, typ)
            if len(arr) == 1:
                arr = np.ones(nsubsets, dtype=typ) * arr
        except KeyValueNotFoundError:
            arr = np.ones(nsubsets, dtype=typ) * missing_val
        return arr

    def get_value(self, key: str):
        return eccodes.codes_get(self.bid, key)

    def get_size(self, key: str):
        return eccodes.codes_get_size(self.bid, key)

    def set_value(self, key: str, value):
        eccodes.codes_set(self.bid, key, value)


class BUFRReader:
    def __init__(self, filename: str):
        with open(filename, "rb") as fobj:
            self._nmsg = eccodes.codes_count_in_file(fobj)
            self._datablock = self._read_datablock(fobj)
            self._msg_offsets = self._get_msg_offsets_from_file(fobj, self._nmsg)

    @contextmanager
    def get_message(self, idx: int):
        msg_data = self[idx]
        bid = eccodes.codes_new_from_message(msg_data)
        try:
            yield BUFRMessage(bid)
        finally:
            eccodes.codes_release(bid)

    @property
    def datablock(self):
        return self._datablock

    @property
    def nmsg(self):
        return self._nmsg

    @property
    def msg_offsets(self):
        return self._msg_offsets

    @staticmethod
    def _read_datablock(fobj) -> bytes:
        with mmap.mmap(fobj.fileno(), length=0, access=mmap.ACCESS_READ) as mobj:
            return mobj.read()

    @staticmethod
    def _get_msg_offsets_from_file(fobj, nmsg: int) -> list[tuple[int, int]]:
        fobj.seek(0)
        offsets = []
        for _ in range(nmsg):
            bid = eccodes.codes_bufr_new_from_file(fobj)
            offset = eccodes.codes_get_message_offset(bid)
            size = eccodes.codes_get_message_size(bid)
            offsets.append((offset, size))
            eccodes.codes_release(bid)
        return offsets

    def _get_msg_data(self, idx):
        msg_offset, msg_size = self._msg_offsets[idx]
        return self._datablock[msg_offset : msg_offset + msg_size]

    def __len__(self) -> int:
        return self._nmsg

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = tuple(range(*idx.indices(self._nmsg)))

        if isinstance(idx, (list, tuple)):
            return [self._get_msg_data(i) for i in idx]

        return self._get_msg_data(idx)

    def __iter__(self):
        for i in range(self._nmsg):
            yield self._get_msg_data(i)
