# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared metadata helper for trajectories datasets.

Trajectories datasets have two frequencies (``base_frequency`` and
``step_frequency``) rather than a single ``frequency``. The trajectory-specific
metadata keys produced here are merged into both the recursive ``specific`` tree
(via ``metadata_specific``) and the flat top-level summary (via
``dataset_metadata``) of every trajectory dataset and wrapper.
"""

from typing import Any

from anemoi.utils.dates import frequency_to_string


def trajectory_metadata(ds: Any) -> dict[str, Any]:
    """Return the trajectory-specific metadata keys shared by all wrappers.

    Parameters
    ----------
    ds : Any
        A trajectories dataset or wrapper exposing the trajectory properties
        (``base_frequency``, ``step_frequency``, ``base_start_date``,
        ``base_end_date``, ``step_start``, ``step_end``).

    Returns
    -------
    dict[str, Any]
        The trajectory-specific metadata keys.
    """
    step_frequency = ds.step_frequency
    return dict(
        base_frequency=frequency_to_string(ds.base_frequency),
        step_frequency=frequency_to_string(step_frequency) if step_frequency is not None else None,
        base_start_date=str(ds.base_start_date),
        base_end_date=str(ds.base_end_date),
        step_start=frequency_to_string(ds.step_start),
        step_end=frequency_to_string(ds.step_end),
    )
