import logging
import os
from typing import Any

import numpy as np
import rich

from ..creator import Creator
from ..locking import Locking
from .context import TabularContext

LOG = logging.getLogger(__name__)

VERSION = "0.30"


class TabularDataset:
    def __init__(self, path: str):
        self.path = path
        self.data_array = None

    def print_info(self):
        rich.print(f"[green]TabularDataset at {self.path}[/green]")


class TabularCreator(Creator):

    allow_nans = True

    ######################################################

    def context(self):
        return TabularContext()

    def load_result(self, result: Any):
        np.save(
            os.path.join(
                self.work_dir,
                f"{result.start_date.isoformat()}-{result.end_date.isoformat()}.npy",
            ),
            result.to_numpy(),
        )

    def finalise(self):
        from .finalise import finalise_tabular_dataset

        finalise_tabular_dataset(self.work_dir, self.path, delete_files=True)

    def statistics(self):
        pass

    ######################################################
    @property
    def check_name(self) -> str:
        return False

    def shape_and_chunks(self, dates: Any) -> tuple[int, ...]:
        total_shape = (len(dates), len(self.minimal_input.variables))
        chunks = (min(100, total_shape[0]), total_shape[1])
        return total_shape, chunks


class SimpleRegistry:
    def __init__(self, path: str):
        self.path = os.path.join(path, "_build_registry")
        os.makedirs(self.path, exist_ok=True)
        self.lock = Locking(os.path.join(self.path, "registry.lock"))

    def create(self, lengths: tuple[int, ...]):
        with self.lock:
            np_lengths = np.array(lengths, dtype="i4")
            np_flags = np.array([False] * len(lengths), dtype=bool)
            np.save(os.path.join(self.path, "lengths.npy"), np_lengths)
            np.save(os.path.join(self.path, "flags.npy"), np_flags)
            self.add_to_history("initialised")

    def get_flags(self):
        with self.lock:
            np_flags = np.load(os.path.join(self.path, "flags.npy"))
            return np_flags

    def get_flag(self, index: int) -> bool:
        return self.get_flags()[index]

    def set_flag(self, index: int) -> bool:
        with self.lock:
            np_flags = np.load(os.path.join(self.path, "flags.npy"))
            if not np_flags[index]:
                np_flags[index] = True
                np.save(os.path.join(self.path, "flags.npy"), np_flags)

    def add_provenance(self, name: str):
        with self.lock:
            pass

    def add_to_history(self, action: str):
        with self.lock:
            pass
