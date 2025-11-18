# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from ..gridded.tasks import FieldTask
from ..gridded.tasks import HasRegistryMixin
from ..gridded.tasks import HasStatisticTempMixin
from .additions import _InitAdditions

LOG = logging.getLogger(__name__)


class Cleanup(FieldTask, HasRegistryMixin, HasStatisticTempMixin):
    """A class to clean up temporary data and registry entries."""

    def __init__(
        self,
        path: str,
        statistics_temp_dir: str | None = None,
        delta: list = [],
        use_threads: bool = False,
        **kwargs: Any,
    ):
        """Initialize a Cleanup instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        delta : list, optional
            The delta values.
        use_threads : bool, optional
            Whether to use threads.
        """
        super().__init__(path)
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.additinon_temp_dir = statistics_temp_dir
        self.tasks = [
            _InitAdditions(path, delta=d, use_threads=use_threads, statistics_temp_dir=statistics_temp_dir)
            for d in delta
        ]

    def run(self) -> None:
        """Run the cleanup."""

        self.tmp_statistics.delete()
        self.registry.clean()
        for actor in self.tasks:
            actor.cleanup()
