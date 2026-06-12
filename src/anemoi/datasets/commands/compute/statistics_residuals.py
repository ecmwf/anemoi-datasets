# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Residual helpers for the ``compute`` command.

The residual is ``dsA - dsB``, intended to be used with regrid/select options on
either dataset so that two datasets at different resolutions are brought to a
common shape before differencing. The differencing and accumulation are driven by
:mod:`engine`; this module only validates that the two datasets are compatible.
"""

import logging
from typing import Any

LOG = logging.getLogger(__name__)


def _check_compatible(ds_a: Any, ds_b: Any) -> None:
    """Validate that two datasets can be differenced element-wise.

    Parameters
    ----------
    ds_a, ds_b : Dataset
        The two opened datasets.

    Raises
    ------
    ValueError
        If lengths, variables or per-step field shapes do not match.
    """
    if len(ds_a) != len(ds_b):
        raise ValueError(
            f"Datasets have different lengths: {len(ds_a)} vs {len(ds_b)}. "
            "Use start=/end= or regrid/select options to align them."
        )
    if list(ds_a.variables) != list(ds_b.variables):
        raise ValueError(
            f"Datasets have different variables:\n  A: {list(ds_a.variables)}\n  B: {list(ds_b.variables)}"
        )
    if tuple(ds_a.shape[1:]) != tuple(ds_b.shape[1:]):
        raise ValueError(
            f"Datasets have different field shapes: {tuple(ds_a.shape[1:])} vs {tuple(ds_b.shape[1:])}. "
            "Use regrid/select options so both datasets share a resolution."
        )
