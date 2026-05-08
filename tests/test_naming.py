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
    # When the requested layout doesn't match, the diagnostic identifies
    # the convention the name actually follows (via guess_layout_from_name).

    def test_trajectory_name_with_gridded_call_warns(self):
        msgs = self._check(self.TRAJECTORY_NAME, layout="gridded")
        assert any("trajectories naming convention" in m and "gridded" in m for m in msgs)

    def test_trajectory_name_with_tabular_call_warns(self):
        msgs = self._check(self.TRAJECTORY_NAME, layout="tabular")
        assert any("trajectories naming convention" in m and "tabular" in m for m in msgs)

    def test_gridded_name_with_trajectory_call_warns(self):
        msgs = self._check(
            self.GRIDDED_NAME,
            layout="trajectories",
            step_frequency=datetime.timedelta(hours=1),
        )
        assert any("gridded naming convention" in m and "trajectories" in m for m in msgs)

    def test_gridded_name_with_tabular_call_warns(self):
        msgs = self._check(self.GRIDDED_NAME, layout="tabular")
        assert any("gridded naming convention" in m and "tabular" in m for m in msgs)

    def test_tabular_name_with_gridded_call_warns(self):
        msgs = self._check(self.TABULAR_NAME, layout="gridded")
        assert any("tabular naming convention" in m and "gridded" in m for m in msgs)

    def test_tabular_name_with_trajectory_call_warns(self):
        msgs = self._check(
            self.TABULAR_NAME,
            layout="trajectories",
            step_frequency=datetime.timedelta(hours=1),
        )
        assert any("tabular naming convention" in m and "trajectories" in m for m in msgs)

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
        assert any("tabular naming convention" in m and "gridded" in m for m in msgs)

    # -- optional resolution (tabular only) ------------------------------

    def test_tabular_without_resolution_passes(self):
        # Tabular datasets often lack a meaningful spatial resolution
        # (e.g. station observations).  The resolution token is optional
        # for tabular only.
        msgs = self._check("aifs-od-fc-oper-0001-mars-2016-2025-v3", layout="tabular")
        assert msgs == []

    def test_tabular_observations_real_name_passes(self):
        msgs = self._check(
            "observations-ea-ofb-0001-1979-2023-v2-combined-aircraft-from-dop",
            layout="tabular",
        )
        assert msgs == []

    def test_tabular_dop_real_name_passes(self):
        msgs = self._check(
            "dop-ea-ofb-0001-1979-2023-v2-combined-aircraft",
            layout="tabular",
        )
        assert msgs == []

    def test_tabular_dop_with_o96_resolution_passes(self):
        msgs = self._check(
            "dop-ea-ofb-0001-o96-1979-2023-v2-combined-aircraft",
            layout="tabular",
        )
        assert msgs == []

    def test_tabular_dop_with_2km_resolution_passes(self):
        # ``2km``-style resolutions are explicitly supported (starts with a digit).
        msgs = self._check(
            "dop-ea-ofb-0001-2km-1979-2023-v2-combined-aircraft",
            layout="tabular",
        )
        assert msgs == []

    def test_gridded_without_resolution_warns(self):
        # Resolution is mandatory for gridded layouts; a name lacking the
        # resolution token does not match the gridded regex (and does not
        # match any other layout's regex either, since it still has a
        # frequency token).
        msgs = self._check(
            "aifs-od-an-oper-0001-mars-1979-2022-1h-v5",
            layout="gridded",
        )
        assert any("does not follow any known naming convention" in m for m in msgs)

    def test_trajectories_without_resolution_warns(self):
        # Resolution is mandatory for trajectory layouts.
        msgs = self._check(
            "aifs-od-fc-oper-0001-mars-2016-2025-18h-1h-v3",
            layout="trajectories",
            step_frequency=datetime.timedelta(hours=1),
        )
        assert any("does not follow any known naming convention" in m for m in msgs)


class TestGuessLayoutFromName:
    """Cover the public ``guess_layout_from_name`` helper."""

    def _guess(self, name):
        from anemoi.datasets.create.naming import guess_layout_from_name

        return guess_layout_from_name(name)

    def test_gridded(self):
        assert self._guess("aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5") == "gridded"

    def test_trajectories(self):
        assert self._guess("aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3") == "trajectories"

    def test_tabular_with_resolution(self):
        assert self._guess("aifs-od-fc-oper-0001-mars-o96-2016-2025-v3") == "tabular"

    def test_tabular_without_resolution(self):
        assert self._guess("dop-ea-ofb-0001-1979-2023-v2-combined-aircraft") == "tabular"

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Cannot guess layout"):
            self._guess("not-a-valid-dataset-name")
