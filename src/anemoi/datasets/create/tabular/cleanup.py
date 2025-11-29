# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from ..base.cleanup import CleanupTask
from . import TabularTaskMixin


class Cleanup(CleanupTask, TabularTaskMixin):

    def _run(self) -> None:
        print(f"Cleanup dataset at {self.path}")
        # Here would be the logic to Cleanup the dataset
