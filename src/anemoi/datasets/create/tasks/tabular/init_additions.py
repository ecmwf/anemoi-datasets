# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from ..init_additions import InitAdditionsTask


class InitAdditions(InitAdditionsTask):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def _run(self) -> None:
        print(f"Init additions for dataset at {self.path}")
        # Here would be the logic to initialize additions


task = InitAdditions
