# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import Any

import numpy as np

from anemoi.datasets.create.config import build_output
from anemoi.datasets.create.config import loader_config
from anemoi.datasets.create.input import InputBuilder
from anemoi.datasets.dates.groups import Groups

LOG = logging.getLogger(__name__)


class Creator(ABC):
    def __init__(self, path: str, config: dict, **kwargs: Any):
        # Catch all floating point errors, including overflow, sqrt(<0), etc

        np.seterr(all="raise", under="warn")

        self.path = path
        self.config = config

        # self.main_config = loader_config(config)
        self.use_threads = kwargs.pop("use_threads", False)
        self.statistics_temp_dir = kwargs.pop("statistics_temp_dir", None)
        self.addition_temp_dir = kwargs.pop("addition_temp_dir", None)
        self.parts = kwargs.pop("parts", None)

        self.kwargs = kwargs

    #####################################################

    @abstractmethod
    def init(self): ...

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def statistics(self): ...

    @abstractmethod
    def size(self): ...

    @abstractmethod
    def cleanup(self): ...

    @abstractmethod
    def init_additions(self): ...

    @abstractmethod
    def load_additions(self): ...

    @abstractmethod
    def finalise_additions(self): ...

    @abstractmethod
    def patch(self): ...

    @abstractmethod
    def verify(self): ...

    #####################################################

    # @abstractmethod
    # @cached_property
    def context(self):
        # Cannot use abstractmethod with mixins
        raise NotImplementedError("Subclasses must implement context property")

    @cached_property
    def groups(self):
        return Groups(**self.main_config.dates)

    @cached_property
    def minimal_input(self):
        one_date = self.groups.one_date()
        return self.input.select(self.context(), one_date)

    @cached_property
    def output(self):
        return build_output(self.main_config.output, parent=self)

    @cached_property
    def input(self):

        return InputBuilder(
            self.main_config.input,
            data_sources=self.main_config.get("data_sources", {}),
        )

    @cached_property
    def main_config(self):
        if self.config is None:
            return self.dataset.get_main_config()
        return loader_config(self.config)
