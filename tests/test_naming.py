# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``anemoi.datasets.create.naming.check_dataset_name``.

Three layout-dependent forms are recognised:

- ``layout="gridded"``      → ``...-<start>-<end>-<freq>-v<n>[-extra]``.
- ``layout="trajectories"`` → ``...-<start>-<end>-<date-freq>-<step-freq>-v<n>[-extra]``.
- ``layout="tabular"``      → ``...-<start>-<end>-v<n>[-extra]``.

The validator must accept each form when called with the matching layout
and reject every other combination. Tabular is never inferred — callers
must opt in explicitly.
"""

import datetime


class TestCheckDatasetName:
    """Cover all three layout-dependent naming forms."""

    GRIDDED_NAME = "aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5"
    TRAJECTORY_NAME = "aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3"
    TABULAR_NAME = "aifs-od-fc-oper-0001-mars-o96-2016-2025-v3"

    def _check(self, name, **kwargs):
        from anemoi.datasets.create.naming import check_dataset_name

        return check_dataset_name(name, **kwargs)

    # -- match cases ------------------------------------------------------

    def test_trajectory_name_with_trajectory_call_passes(self):
        msgs = self._check(
            self.TRAJECTORY_NAME,
            layout="trajectories",
            resolution="o96",
            start_date=datetime.date(2016, 1, 1),
            end_date=datetime.date(2025, 1, 1),
            frequency=datetime.timedelta(hours=18),
            step_frequency=datetime.timedelta(hours=1),
        )
        assert msgs == []

    def test_gridded_name_with_gridded_call_passes(self):
        msgs = self._check(
            self.GRIDDED_NAME,
            layout="gridded",
            resolution="o96",
            start_date=datetime.date(1979, 1, 1),
            end_date=datetime.date(2022, 1, 1),
            frequency=datetime.timedelta(hours=1),
        )
        assert msgs == []

    def test_tabular_name_with_tabular_call_passes(self):
        msgs = self._check(
            self.TABULAR_NAME,
            layout="tabular",
            start_date=datetime.date(2016, 1, 1),
            end_date=datetime.date(2025, 1, 1),
        )
        assert msgs == []

    # -- cross-layout mismatches -----------------------------------------

    def test_trajectory_name_with_gridded_call_warns(self):
        msgs = self._check(self.TRAJECTORY_NAME, layout="gridded")
        assert any("two-frequency form" in m and "gridded" in m for m in msgs)

    def test_trajectory_name_with_tabular_call_warns(self):
        msgs = self._check(self.TRAJECTORY_NAME, layout="tabular")
        assert any("two-frequency form" in m and "tabular" in m for m in msgs)

    def test_gridded_name_with_trajectory_call_warns(self):
        msgs = self._check(
            self.GRIDDED_NAME,
            layout="trajectories",
            step_frequency=datetime.timedelta(hours=1),
        )
        assert any("single-frequency form" in m and "trajectories" in m for m in msgs)

    def test_gridded_name_with_tabular_call_warns(self):
        msgs = self._check(self.GRIDDED_NAME, layout="tabular")
        assert any("single-frequency form" in m and "tabular" in m for m in msgs)

    def test_tabular_name_with_gridded_call_warns(self):
        msgs = self._check(self.TABULAR_NAME, layout="gridded")
        assert any("no-frequency form" in m and "gridded" in m for m in msgs)

    def test_tabular_name_with_trajectory_call_warns(self):
        msgs = self._check(
            self.TABULAR_NAME,
            layout="trajectories",
            step_frequency=datetime.timedelta(hours=1),
        )
        assert any("no-frequency form" in m and "trajectories" in m for m in msgs)

    # -- frequency-value mismatches --------------------------------------

    def test_trajectory_step_frequency_mismatch_warns(self):
        msgs = self._check(
            self.TRAJECTORY_NAME,
            layout="trajectories",
            frequency=datetime.timedelta(hours=18),
            step_frequency=datetime.timedelta(hours=6),
        )
        assert any("step frequency" in m and "1h" in m for m in msgs)

    # -- extra suffix -----------------------------------------------------

    def test_trajectory_with_extra_suffix_passes(self):
        msgs = self._check(
            self.TRAJECTORY_NAME + "-experimental",
            layout="trajectories",
            frequency=datetime.timedelta(hours=18),
            step_frequency=datetime.timedelta(hours=1),
        )
        assert msgs == []

    def test_tabular_with_extra_suffix_passes(self):
        msgs = self._check(self.TABULAR_NAME + "-experimental", layout="tabular")
        assert msgs == []

    # -- backward-compat layout inference --------------------------------

    def test_layout_inferred_from_kwargs(self):
        # No layout kwarg; the validator must infer "gridded" or "trajectories"
        # from frequency / step_frequency.  Tabular is never inferred —
        # callers must pass layout="tabular" explicitly.
        assert self._check(self.GRIDDED_NAME, frequency=datetime.timedelta(hours=1)) == []
        assert (
            self._check(
                self.TRAJECTORY_NAME,
                frequency=datetime.timedelta(hours=18),
                step_frequency=datetime.timedelta(hours=1),
            )
            == []
        )

    def test_tabular_is_never_default(self):
        # Without layout="tabular", a tabular-shaped name must be flagged
        # as gridded-non-conformant rather than silently accepted.
        msgs = self._check(self.TABULAR_NAME)
        assert any("no-frequency form" in m and "gridded" in m for m in msgs)
