# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Protocol
from typing import runtime_checkable

from .mars_licence import DEFAULT_MARS_LICENCE_POLICY
from .mars_licence import MarsLicencePolicy
from .mars_licence import validate_mars_ccby_allowlist_file


@runtime_checkable
class LicencePolicy(Protocol):
    """Protocol for source-specific licence validation policies."""

    def validate(self, parsed: dict) -> None:
        """Validate a parsed recipe mapping against this policy.

        Parameters
        ----------
        parsed : dict
            Raw YAML recipe as a dictionary.

        Raises
        ------
        ValueError
            If the recipe violates this policy.
        """
        ...


def collect_recipe_sources(parsed: dict[str, Any], known_sources: set[str]) -> set[str]:
    """Recursively scan the input section of a recipe for known source names.

    Parameters
    ----------
    parsed : dict
        Raw YAML recipe as a dictionary.
    known_sources : set[str]
        Source names to look for (typically the keys of LICENCE_POLICIES).

    Returns
    -------
    set[str]
        Source names found in the recipe input section.
    """
    found: set[str] = set()

    def _scan(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in known_sources:
                    found.add(key)
                _scan(value)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(parsed.get("input") or {})
    return found


LICENCE_POLICIES: dict[str, LicencePolicy] = {
    "mars": DEFAULT_MARS_LICENCE_POLICY,
}

__all__ = [
	"LicencePolicy",
	"LICENCE_POLICIES",
	"MarsLicencePolicy",
	"collect_recipe_sources",
	"validate_mars_ccby_allowlist_file",
]
