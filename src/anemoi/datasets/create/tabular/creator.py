import logging
import os
from functools import cached_property
from typing import Any

import numpy as np
import rich

from ..creator import Creator
from ..locking import Locking
from ..parts import PartFilter
from .context import TabularContext

LOG = logging.getLogger(__name__)


class TabularDataset:
    def __init__(self, path: str):
        self.path = path
        self.data_array = None

    def print_info(self):
        rich.print(f"[green]TabularDataset at {self.path}[/green]")


class TabularCreator(Creator):

    def init(self):
        rich.print(self.minimal_input)

        lengths = tuple(len(g) for g in self.groups)
        self.registry.create(lengths=lengths)

    def load(self):
        self.dataset = TabularDataset(self.path)
        total = len(self.registry.get_flags())
        self.chunk_filter = PartFilter(parts=self.parts, total=total)

        self.data_array = self.dataset.data_array
        self.n_groups = len(self.groups)
        self.read_dataset_metadata(self.dataset.path)

        for igroup, group in enumerate(self.groups):
            if not self.chunk_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue

            # assert isinstance(group[0], datetime.datetime), type(group[0])
            LOG.debug(f"Building data for group {igroup}/{self.n_groups}")

            result = self.input.select(self.context(), argument=group)
            # assert result.group_of_dates == group, (len(result.group_of_dates), len(group), group)

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_provenance(name="provenance_load")
        self.tmp_statistics.add_provenance(name="provenance_load", config=self.main_config)

        self.dataset.print_info()

    def statistics(self):
        rich.print("[red]Statistics method not implemented yet.[/red]")

    def size(self):
        rich.print("[red]Size method not implemented yet.[/red]")

    def cleanup(self):
        rich.print("[red]Cleanup method not implemented yet.[/red]")

    def init_additions(self):
        rich.print("[red]Init additions method not implemented yet.[/red]")

    def load_additions(self):
        rich.print("[red]Load additions method not implemented yet.[/red]")

    def finalise_additions(self):
        rich.print("[red]Finalise additions method not implemented yet.[/red]")

    def patch(self):
        rich.print("[red]Patch method not implemented yet.[/red]")

    def verify(self):
        rich.print("[red]Verify method not implemented yet.[/red]")

    ######################################################

    def context(self):
        return TabularContext()

    @cached_property
    def registry(self) -> Any:
        return SimpleRegistry(self.path)

    def read_dataset_metadata(self, path: str):
        pass

    def load_result(self, result: Any):
        pass

    @cached_property
    def tmp_statistics(self):
        return TmpSatistics(self.path)


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


class TmpSatistics:
    def __init__(self, path: str):
        self.path = path

    def add_provenance(self, name: str, config: Any):
        pass
