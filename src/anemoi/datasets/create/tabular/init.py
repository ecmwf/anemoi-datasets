# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from ..base.init import InitTask
from . import TabularTaskMixin


class Init(InitTask, TabularTaskMixin):

    def context(self):
        return TabularTaskMixin.context(self)

    def _run(self) -> None:
        print(f"Init dataset at {self.path}")
        self.minimal_input
