# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for MARS licence allowlist checks in the create flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from anemoi.datasets.create.check import mars_licence as checker


CHECK_FIXTURES = Path(__file__).parent


def test_good_mars_licence() -> None:
    """Allowlisted MARS request with top-level licence should pass file validation."""

    checker.validate_mars_ccby_allowlist_file(CHECK_FIXTURES / "mars_good_licence.yaml")


def test_bad_mars_licence() -> None:
    """Disallowed MARS request should fail file validation for CC-BY licence."""

    with pytest.raises(ValueError, match="not in the allowlist"):
        checker.validate_mars_ccby_allowlist_file(CHECK_FIXTURES / "mars_bad_licence.yaml")

