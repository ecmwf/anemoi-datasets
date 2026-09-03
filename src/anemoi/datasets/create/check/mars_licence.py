# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Pattern
import yaml

from anemoi.datasets.create.sources.mars import MARS_KEYS
from anemoi.datasets.create.sources.mars import _normalise_time as mars_normalise_time

__all__ = [
    "MarsLicencePolicy",
    "validate_mars_ccby_allowlist_file",
]


@dataclass(frozen=True)
class MarsLicencePolicy:
    """Configuration for MARS CC-BY allowlist validation rules."""

    cc_by_pattern: Pattern[str] = field(default_factory=lambda: re.compile(r"\bcc[- ]?by\b", re.IGNORECASE))
    allowed_patterns: tuple[dict[str, frozenset[str]], ...] = field(
        default_factory=lambda: (
            {
                "class": frozenset({"od"}),
                "expver": frozenset({"1"}),
                "stream": frozenset(
                    {
                        "oper",
                        "wave",
                        "mnth",
                        "wamo",
                        "scda",
                        "scwv",
                        "enfh",
                        "efhs",
                        "efho",
                        "efov",
                        "ewhc",
                        "enfo",
                        "waef",
                        "enwh",
                        "wehs",
                        "ewho",
                        "weov",
                        "eefh",
                        "eehs",
                        "eefo",
                        "weef",
                        "weeh",
                        "wees",
                    }
                ),
            },
            {
                "class": frozenset({"od"}),
                "expver": frozenset({"1"}),
                "origin": frozenset({"ecmf"}),
                "stream": frozenset({"mmsf", "msmm", "mmsa"}),
            },
            {
                "class": frozenset({"ai"}),
                "expver": frozenset({"1"}),
            },
            {
                "class": frozenset({"ea"}),
            },
                        {
                "class": frozenset({"rr"}),
            }
        )
    )
    non_mars_keys: frozenset[str] = field(
        default_factory=lambda: frozenset({"netcdf", "grib", "opendap", "zarr", "fdb", "xarray", "file", "files", "url"})
    )

    def validate(self, parsed: dict[str, Any]) -> None:
        """Validate a parsed recipe mapping against this MARS CC-BY allowlist policy."""
        _validate_parsed_mapping(parsed, self)


DEFAULT_MARS_LICENCE_POLICY = MarsLicencePolicy()
MARS_REQUEST_ID_KEYS = tuple(k for k in ("class", "expver", "stream", "origin") if k in MARS_KEYS)


def _normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_request_field_value(key: str, value: Any) -> str:
    """Remove leading zeroes, strip spaces, make lowercase."""

    normalized = _normalize_value(value)
    if key == "expver" and normalized.isdigit():
        return str(int(normalized))
    if key == "time" and normalized.isdigit():
        return mars_normalise_time(normalized)
    return normalized


def _is_cc_by_licence(licence_value: Any, policy: MarsLicencePolicy) -> bool:
    """Check if a licence value is CC-BY."""

    if not isinstance(licence_value, str):
        return False
    return policy.cc_by_pattern.search(licence_value.strip()) is not None

def _is_mars_request(request: dict[str, Any]) -> bool:
    """Check if a dictionary represents a MARS request."""

    return any(key in request for key in MARS_REQUEST_ID_KEYS)


def _collect_mars_requests(node: Any, collected: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Recursively collect all MARS requests from a parsed recipe."""

    if collected is None:
        collected = []

    if isinstance(node, dict):
        if _is_mars_request(node):
            collected.append(node)
        for value in node.values():
            _collect_mars_requests(value, collected)
    elif isinstance(node, list):
        for item in node:
            _collect_mars_requests(item, collected)
    return collected


def _deduplicate_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate MARS requests based on their normalized field values."""

    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for request in requests:
        key = tuple(sorted((str(k), _normalize_request_field_value(str(k), v)) for k, v in request.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(request)

    return unique


def _input_is_mars_only(parsed: dict[str, Any], policy: MarsLicencePolicy) -> bool:
    """Check if the input section contains only MARS requests (no other providers)."""

    input_section = parsed.get("input")
    if not isinstance(input_section, (dict, list)):
        return False

    def _contains_non_mars_source(node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if _normalize_value(key) in policy.non_mars_keys:
                    return True
                if _contains_non_mars_source(value):
                    return True
            return False
        if isinstance(node, list):
            return any(_contains_non_mars_source(item) for item in node)
        return False

    return not _contains_non_mars_source(input_section)


def _normalize_request_for_matching(request: dict[str, Any]) -> dict[str, Any]:
    """Normalize a MARS request for allowlist matching."""

    normalized = dict(request)
    if "stream" not in normalized and _normalize_value(normalized.get("class")) == "od":
        normalized["stream"] = "oper"
    return normalized


def _matches_allowed_pattern(request: dict[str, Any], pattern: dict[str, frozenset[str]]) -> bool:
    """Check if a MARS request matches an allowed pattern."""

    for key, allowed_values in pattern.items():
        raw_value = request.get(key)
        if raw_value is None:
            return False
        if _normalize_request_field_value(key, raw_value) not in allowed_values:
            return False
    return True


def _format_request(request: dict[str, Any]) -> str:
    """Format a MARS request as a readable string."""

    keys = sorted(k for k in request.keys() if k in ("class", "expver", "origin", "stream"))
    return ", ".join(f"{key}={request[key]}" for key in keys) or "<no mars keys>"


def _validate_parsed_mapping(parsed: dict[str, Any], policy: MarsLicencePolicy) -> None:
    """Validate a parsed recipe mapping against MARS CC-BY allowlist policy."""

    licence_value = parsed.get("licence")
    mars_requests = _deduplicate_requests(_collect_mars_requests(parsed))

    if not mars_requests:
        return

    request_match_flags: list[bool] = []
    for request in mars_requests:
        normalized_request = _normalize_request_for_matching(request)
        request_match_flags.append(
            any(_matches_allowed_pattern(normalized_request, pattern) for pattern in policy.allowed_patterns)
        )

    if not _is_cc_by_licence(licence_value, policy):
        return

    for idx, request in enumerate(mars_requests, start=1):
        if request_match_flags[idx - 1]:
            continue
        raise ValueError(
            "CC-BY licenced MARS request is not in the allowlist "
            f"(request #{idx}: {_format_request(request)})."
        )


def validate_mars_ccby_allowlist_file(file_path: Path) -> None:
    """Validate a dataset recipe file against MARS CC-BY allowlist policy."""

    with open(file_path) as f:
        parsed = yaml.safe_load(f)
    _validate_parsed_mapping(parsed, DEFAULT_MARS_LICENCE_POLICY)
