# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Validate CC-BY licensed MARS requests against an allowlist using anemoi recipe loading.

This module validates dataset recipe files to ensure:
1. MARS requests that are in the approved allowlist require CC-BY licence.
2. All CC-BY licensed MARS requests are in the approved allowlist.
3. When MARS is the only input source and no licence is provided, validation fails.

The module uses anemoi's recipe loader for strict validation of recipe schema,
which ensures compatibility with the latest anemoi dataset format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .recipe import Recipe
from .recipe import loader_recipe_from_yaml

__all__ = ["validate_mars_ccby_allowlist_file", "validate_mars_ccby_allowlist_text"]


_CC_BY_PATTERN = re.compile(r"\bcc[- ]?by\b", re.IGNORECASE)

_ALLOWED_PATTERNS: list[dict[str, set[str]]] = [
    {
        "class": {"od"},
        "expver": {"1"},
        "stream": {"oper", "wave", "mnth", "wamo", "scda", "scwv"},
    },
    {
        "class": {"od"},
        "expver": {"1"},
        "stream": {
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
        },
    },
    {
        "class": {"od"},
        "expver": {"1"},
        "stream": {"eefh", "eehs", "eefo", "weef", "weeh", "wees"},
    },
    {
        "class": {"od"},
        "expver": {"1"},
        "origin": {"ecmf"},
        "stream": {"mmsf", "msmm", "mmsa"},
    },
    {
        "class": {"ai"},
        "expver": {"1"},
    },
]


def _normalize_value(value: Any) -> str:
    """Normalize a value for request matching."""

    return str(value).strip().lower()


def _normalize_request_field_value(key: str, value: Any) -> str:
    """Normalize a request field value for allowlist matching."""

    normalized = _normalize_value(value)
    if key == "expver" and normalized.isdigit():
        return str(int(normalized))
    return normalized


def _is_cc_by_licence(licence_value: Any) -> bool:
    """Check if a licence value is CC-BY or CC-BY-SA."""

    if not isinstance(licence_value, str):
        return False
    return _CC_BY_PATTERN.search(licence_value.strip()) is not None


def _is_missing_licence(licence_value: Any) -> bool:
    """Check if a licence value is missing or empty."""

    if licence_value is None:
        return True
    if isinstance(licence_value, str) and not licence_value.strip():
        return True
    return False


def _extract_licence_value(parsed: dict[str, Any]) -> Any:
    """Extract the top-level British-spelled licence field from a parsed recipe."""

    return parsed.get("licence")


def _is_mars_request(request: dict[str, Any]) -> bool:
    """Check if a dictionary represents a MARS request."""

    return any(key in request for key in ("class", "expver", "stream", "origin"))


def _collect_mars_requests(node: Any, collected: list[dict[str, Any]]) -> None:
    """Recursively collect all MARS requests from a parsed recipe."""

    if isinstance(node, dict):
        if _is_mars_request(node):
            collected.append(node)
        for value in node.values():
            _collect_mars_requests(value, collected)
    elif isinstance(node, list):
        for item in node:
            _collect_mars_requests(item, collected)


def _extract_requests(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all MARS requests from a parsed recipe."""

    extracted: list[dict[str, Any]] = []
    _collect_mars_requests(parsed, extracted)
    return extracted


def _deduplicate_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate MARS requests based on their normalized field values."""

    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for request in requests:
        key = tuple(sorted((str(k), _normalize_value(v)) for k, v in request.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(request)

    return unique


def _input_is_mars_only(parsed: dict[str, Any]) -> bool:
    """Check if the input section contains only MARS requests (no other providers)."""

    input_section = parsed.get("input")
    if not isinstance(input_section, (dict, list)):
        return False

    disallowed_provider_keys = {
        "netcdf",
        "grib",
        "opendap",
        "zarr",
        "fdb",
        "xarray",
        "file",
        "files",
        "url",
    }

    def _contains_disallowed_provider(node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if _normalize_value(key) in disallowed_provider_keys:
                    return True
                if _contains_disallowed_provider(value):
                    return True
            return False
        if isinstance(node, list):
            return any(_contains_disallowed_provider(item) for item in node)
        return False

    return not _contains_disallowed_provider(input_section)


def _normalize_request_for_matching(request: dict[str, Any]) -> dict[str, Any]:
    """Normalize a MARS request for allowlist matching."""

    normalized = dict(request)
    if "stream" not in normalized and _normalize_value(normalized.get("class")) == "od":
        normalized["stream"] = "oper"
    return normalized


def _matches_allowed_pattern(request: dict[str, Any], pattern: dict[str, set[str]]) -> bool:
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


def _validate_parsed_mapping(parsed: dict[str, Any]) -> None:
    """Validate a parsed recipe mapping against MARS CC-BY allowlist policy.

    Raises
    ------
    ValueError
        If the recipe violates the MARS CC-BY allowlist policy.
    """

    licence_value = _extract_licence_value(parsed)
    mars_requests = _deduplicate_requests(_extract_requests(parsed))

    if not mars_requests:
        return

    request_match_flags: list[bool] = []
    for request in mars_requests:
        normalized_request = _normalize_request_for_matching(request)
        request_match_flags.append(
            any(_matches_allowed_pattern(normalized_request, pattern) for pattern in _ALLOWED_PATTERNS)
        )

    if _is_missing_licence(licence_value) and all(request_match_flags) and _input_is_mars_only(parsed):
        raise ValueError(
            "Top-level licence is missing for allowlisted MARS input; expected CC-BY-4.0."
        )

    if not _is_cc_by_licence(licence_value):
        return

    for idx, request in enumerate(mars_requests, start=1):
        if request_match_flags[idx - 1]:
            continue
        raise ValueError(
            "CC-BY licensed MARS request is not in the allowlist "
            f"(request #{idx}: {_format_request(request)})."
        )


def _coerce_recipe_to_mapping(recipe: Any) -> dict[str, Any] | None:
    """Coerce a Recipe object or other type to a dict mapping."""

    if isinstance(recipe, dict):
        return recipe

    model_dump = getattr(recipe, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    return None


def _load_recipe_from_file(file_path: Path) -> dict[str, Any]:
    recipe = _coerce_recipe_to_mapping(loader_recipe_from_yaml(str(file_path)))
    if recipe is None:
        raise ValueError("anemoi recipe did not produce a mapping")
    return recipe


def validate_mars_ccby_allowlist_file(file_path: Path) -> None:
    """Validate a dataset recipe file against MARS CC-BY allowlist policy.

    The recipe must be compatible with anemoi's Recipe model for validation.
    This means it must conform to the anemoi recipe schema for the dataset format.

    Parameters
    ----------
    file_path : Path
        Path to the dataset recipe YAML file.

    Raises
    ------
    ValueError
        If the recipe violates the MARS CC-BY allowlist policy.
    OSError
        If the file cannot be read.
    """

    parsed = _load_recipe_from_file(file_path)
    _validate_parsed_mapping(parsed)


def validate_mars_ccby_allowlist_text(yaml_text: str) -> None:
    """Validate recipe YAML text against MARS CC-BY allowlist policy.

    The recipe must be compatible with anemoi's Recipe model for validation.

    Parameters
    ----------
    yaml_text : str
        YAML text containing a dataset recipe.

    Raises
    ------
    ValueError
        If the YAML cannot be parsed or violates the MARS CC-BY allowlist policy.
    """

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("YAML root must be a mapping")

    recipe = _coerce_recipe_to_mapping(Recipe(**parsed))
    if recipe is None:
        raise ValueError("anemoi recipe did not produce a mapping")

    _validate_parsed_mapping(recipe)
