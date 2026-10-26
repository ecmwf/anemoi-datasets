# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Residual helpers for the ``compute`` command.

The residual is ``dsA - dsB`` (``--minus``), formed value by value, so the two
datasets must have the same dates, the same variables in the same order, and the
same field shape. ``--grid`` interpolates both of them onto a common grid first,
in which case their grids are allowed to differ. The differencing and accumulation
are driven by :mod:`engine`; this module only validates that the two datasets are
compatible.
"""

import logging
from typing import Any

LOG = logging.getLogger(__name__)


def _check_compatible(ds_a: Any, ds_b: Any, ignore_grid: bool = False) -> None:
    """Validate that two datasets can be differenced element-wise.

    Parameters
    ----------
    ds_a, ds_b : Dataset
        The two opened datasets.
    ignore_grid : bool, optional
        Whether to accept two different grids. This is what ``--grid`` does: both
        datasets are interpolated onto the requested grid before they are
        subtracted, so only the shape of what is *not* the grid has to match here.

    Raises
    ------
    ValueError
        If lengths, variables or per-step field shapes do not match.
    """
    if len(ds_a) != len(ds_b):
        raise ValueError(
            f"Datasets have different lengths: {len(ds_a)} vs {len(ds_b)}. "
            "Use start=/end= or select options to align them."
        )
    if list(ds_a.variables) != list(ds_b.variables):
        raise ValueError(
            f"Datasets have different variables:\n  A: {list(ds_a.variables)}\n  B: {list(ds_b.variables)}"
        )

    # The last axis is the grid: --grid makes it common, the others never are.
    shape_a = tuple(ds_a.shape[1:-1]) if ignore_grid else tuple(ds_a.shape[1:])
    shape_b = tuple(ds_b.shape[1:-1]) if ignore_grid else tuple(ds_b.shape[1:])

    if shape_a != shape_b:
        raise ValueError(
            f"Datasets have different field shapes: {shape_a} vs {shape_b}. "
            + ("" if ignore_grid else "Use --grid to interpolate them onto a common grid.")
        )
