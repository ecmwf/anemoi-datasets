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
import yaml

from anemoi.datasets.create import check_mars_licence as checker


FIXTURES = Path(__file__).parent / "fixtures"


class _FakeRecipe:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self) -> dict:
        return self.payload


def _load_fixture(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert isinstance(parsed, dict)
    return parsed


def test_file_validation_uses_anemoi_loader_good_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlisted MARS request with top-level licence should pass file validation."""

    parsed = _load_fixture("mars_ccby_allowlist_good_licence.yaml")
    monkeypatch.setattr(checker, "loader_recipe_from_yaml", lambda _path: _FakeRecipe(parsed))

    checker.validate_mars_ccby_allowlist_file(Path("/tmp/fake.yaml"))


def test_file_validation_uses_anemoi_loader_bad_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disallowed MARS request should fail file validation for CC-BY licence."""

    parsed = _load_fixture("mars_ccby_allowlist_bad_licence.yaml")
    monkeypatch.setattr(checker, "loader_recipe_from_yaml", lambda _path: _FakeRecipe(parsed))

    with pytest.raises(ValueError, match="not in the allowlist"):
        checker.validate_mars_ccby_allowlist_file(Path("/tmp/fake.yaml"))


def test_text_validation_uses_recipe_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text validation should pass when Recipe model emits allowlisted mapping."""

    parsed = _load_fixture("mars_ccby_allowlist_good_licence.yaml")

    class FakeRecipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump(self) -> dict:
            return parsed

    monkeypatch.setattr(checker, "Recipe", FakeRecipe)

    checker.validate_mars_ccby_allowlist_text(
        """
name: demo
dates:
  start: 2024-01-01
  end: 2024-01-02
licence: CC-BY-4.0
input:
  mars:
    class: od
    expver: 1
    stream: oper
"""
    )


def test_text_validation_rejects_american_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level license should be ignored so missing licence guard still triggers."""

    class FakeRecipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump(self) -> dict:
            return {
                "license": "CC-BY-4.0",
                "input": {
                    "mars": {
                        "class": "od",
                        "expver": 1,
                        "stream": "oper",
                    }
                },
            }

    monkeypatch.setattr(checker, "Recipe", FakeRecipe)

    with pytest.raises(ValueError, match="Top-level licence is missing"):
        checker.validate_mars_ccby_allowlist_text(
            """
name: demo
dates:
  start: 2024-01-01
  end: 2024-01-02
license: CC-BY-4.0
input:
  mars:
    class: od
    expver: 1
    stream: oper
"""
        )
