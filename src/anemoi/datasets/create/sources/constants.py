# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

from . import source_registry
from .forcings import ForcingsSource


@source_registry.register("constants")
class ConstantsSource(ForcingsSource):

    def execute(self, dates):
        warnings.warn(
            "The source `constants` is deprecated, use `forcings` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(self.template) == 0:
            raise ValueError("Forcings template is empty.")

        return super().execute(dates)
