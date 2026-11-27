# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from ..common.residual_statistics import ResidualStatistics as CommonResidualStatistics
from .metadata import trajectory_metadata

LOG = logging.getLogger(__name__)


class ResidualStatistics(CommonResidualStatistics):
    """Attach residual statistics to a trajectories dataset.

    Residual statistics are indexed by variable only, so the whole data and
    statistics path is inherited from the layout-agnostic wrapper. Only the
    metadata methods are overridden, because trajectory datasets have two
    frequencies and no single ``frequency`` property.
    """

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return super().metadata_specific(**trajectory_metadata(self), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        md = super().dataset_metadata()
        md.update(trajectory_metadata(self))
        return md
