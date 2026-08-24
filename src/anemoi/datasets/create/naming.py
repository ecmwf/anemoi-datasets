# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import re

from anemoi.utils.config import load_config
from anemoi.utils.dates import frequency_to_string

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout-specific patterns — one per layout, used for both validation and
# layout-from-name inference.  Each pattern's groups are paired with the
# matching ``*_KEYS`` list to build a parsed-fields dictionary.
# ---------------------------------------------------------------------------

# ``...-<purpose>-<labelling>-<source>-<resolution>-<start>-<end>-<freq>-v<n>[-extra]``
GRIDDED_PATTERN = r"^(\w+)-([\w-]+)-(\w+)-([0-9on]\w*)-(\d\d\d\d)-(\d\d\d\d)-" r"(\d+h|\d+m)-v(\d+)-?([a-zA-Z0-9-]+)?$"
GRIDDED_KEYS = [
    "purpose",
    "labelling",
    "source",
    "resolution",
    "start_date",
    "end_date",
    "frequency",
    "version",
    "additional",
]

# ``...-<source>-<resolution>-<start>-<end>-<date-freq>-<step-freq>-v<n>[-extra]``
TRAJECTORIES_PATTERN = (
    r"^(\w+)-([\w-]+)-(\w+)-([0-9on]\w*)-(\d\d\d\d)-(\d\d\d\d)-" r"(\d+h|\d+m)-(\d+h|\d+m)-v(\d+)-?([a-zA-Z0-9-]+)?$"
)
TRAJECTORIES_KEYS = [
    "purpose",
    "labelling",
    "source",
    "resolution",
    "start_date",
    "end_date",
    "frequency",
    "step_frequency",
    "version",
    "additional",
]

# ``...-<source>[-<resolution>]-<start>-<end>-v<n>[-extra]``
# Resolution is optional: tabular datasets may have no meaningful spatial
# resolution (e.g. station observations).
TABULAR_PATTERN = r"^(\w+)-([\w-]+)-(\w+)(?:-([0-9on]\w*))?-(\d\d\d\d)-(\d\d\d\d)-" r"v(\d+)-?([a-zA-Z0-9-]+)?$"
TABULAR_KEYS = [
    "purpose",
    "labelling",
    "source",
    "resolution",
    "start_date",
    "end_date",
    "version",
    "additional",
]

# Layout name -> compiled-able pattern, used by the layout-agnostic path to
# tell whether a name's *shape* matches a given layout.
LAYOUT_PATTERNS = {
    "gridded": GRIDDED_PATTERN,
    "trajectories": TRAJECTORIES_PATTERN,
    "tabular": TABULAR_PATTERN,
}


def _guess_layout_from_name(name: str) -> str | None:
    """Guess the dataset layout from the syntactic form of its name.

    Best-effort only — used to enrich diagnostics, never to decide which layout
    to validate a name against.  Returns ``"gridded"``, ``"trajectories"`` or
    ``"tabular"`` when exactly one layout's pattern matches, and ``None`` when
    the layout cannot be determined — either because the name follows no known
    convention or because it is ambiguous (matches more than one layout).

    Parameters
    ----------
    name : str
        The dataset name (without extension).

    Returns
    -------
    str or None
        The layout name when exactly one pattern matches, otherwise ``None``.
    """
    matches = [layout for layout, pattern in LAYOUT_PATTERNS.items() if re.match(pattern, name)]
    if len(matches) == 1:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_characters(name: str) -> list[str]:
    messages: list[str] = []
    if not name.islower():
        messages.append(f"The dataset name '{name}' should be in lower case.")
    if "_" in name:
        messages.append(f"The dataset name '{name}' should use '-' instead of '_'.")
    for c in name:
        if not c.isalnum() and c not in "-":
            messages.append(f"The dataset name '{name}' should only contain alphanumeric characters and '-'.")
    return messages


def _check_resolution_match(parsed: dict, resolution: str | None, name: str) -> list[str]:
    if resolution is None:
        return []
    messages: list[str] = []
    resolution_str = str(resolution).replace(".", "p").lower()
    if resolution_str not in name:
        messages.append(f"The resolution is '{resolution_str}', but it is missing in '{name}'.")
    if parsed.get("resolution") and parsed["resolution"] != resolution_str:
        messages.append(f"The resolution is '{resolution_str}', but it is '{parsed['resolution']}' in '{name}'.")
    return messages


def _check_frequency_match(
    parsed: dict, frequency: datetime.timedelta | None, key: str, label: str, name: str
) -> list[str]:
    if frequency is None:
        return []
    messages: list[str] = []
    frequency_str = frequency_to_string(frequency)
    if frequency_str not in name:
        messages.append(f"The {label} is '{frequency_str}', but it is missing in '{name}'.")
    if parsed.get(key) and parsed[key] != frequency_str:
        messages.append(f"The {label} is '{frequency_str}', but it is '{parsed[key]}' in '{name}'.")
    return messages


def _check_year_match(parsed: dict, year_date: datetime.date | None, key: str, label: str, name: str) -> list[str]:
    if year_date is None:
        return []
    messages: list[str] = []
    year_str = str(year_date.year)
    if year_str not in name:
        messages.append(f"The {label} is '{year_str}', but it is missing in '{name}'.")
    if parsed.get(key) and parsed[key] != year_str:
        messages.append(f"The {label} is '{year_str}', but it is '{parsed[key]}' in '{name}'.")
    return messages


def _generic_non_conformant(name: str) -> str:
    return (
        f"The dataset name '{name}' does not follow any known naming convention. "
        "See here for details: "
        "https://anemoi-registry.readthedocs.io/en/latest/naming-conventions.html"
    )


def _layout_mismatch_diagnostic(name: str, requested_layout: str) -> list[str]:
    """Diagnose why ``name`` failed the requested layout's regex.

    If the name follows another known convention, point the user at the
    matching layout.  Otherwise return a generic non-conformant message.
    """
    actual = _guess_layout_from_name(name)
    if actual is None:
        # No known convention, or an ambiguous match — fall back to generic.
        return [_generic_non_conformant(name)]
    return [
        f"The dataset name '{name}' follows the {actual} naming convention, "
        f"but the requested layout is {requested_layout}."
    ]


def _format_parsed(name: str, parsed: dict) -> str:
    return f"'{name}' is parsed as: " + "/".join(f"{k}={v}" for k, v in parsed.items()) + "."


# ---------------------------------------------------------------------------
# Per-layout validators
# ---------------------------------------------------------------------------


def _check_gridded_name(
    name: str,
    resolution: str | None,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
    frequency: datetime.timedelta | None,
) -> list[str]:
    messages = _check_characters(name)
    match = re.match(GRIDDED_PATTERN, name)
    if not match:
        messages.extend(_layout_mismatch_diagnostic(name, "gridded"))
        return messages
    parsed = dict(zip(GRIDDED_KEYS, match.groups()))
    messages.extend(_check_resolution_match(parsed, resolution, name))
    messages.extend(_check_frequency_match(parsed, frequency, "frequency", "frequency", name))
    messages.extend(_check_year_match(parsed, start_date, "start_date", "start date", name))
    messages.extend(_check_year_match(parsed, end_date, "end_date", "end date", name))
    if messages:
        messages.append(_format_parsed(name, parsed))
    return messages


def _check_trajectories_name(
    name: str,
    resolution: str | None,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
    frequency: datetime.timedelta | None,
    step_frequency: datetime.timedelta | None,
) -> list[str]:
    messages = _check_characters(name)
    match = re.match(TRAJECTORIES_PATTERN, name)
    if not match:
        messages.extend(_layout_mismatch_diagnostic(name, "trajectories"))
        return messages
    parsed = dict(zip(TRAJECTORIES_KEYS, match.groups()))
    messages.extend(_check_resolution_match(parsed, resolution, name))
    messages.extend(_check_frequency_match(parsed, frequency, "frequency", "frequency", name))
    messages.extend(_check_frequency_match(parsed, step_frequency, "step_frequency", "step frequency", name))
    messages.extend(_check_year_match(parsed, start_date, "start_date", "start date", name))
    messages.extend(_check_year_match(parsed, end_date, "end_date", "end date", name))
    if messages:
        messages.append(_format_parsed(name, parsed))
    return messages


def _check_tabular_name(
    name: str,
    resolution: str | None,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
) -> list[str]:
    messages = _check_characters(name)
    match = re.match(TABULAR_PATTERN, name)
    if not match:
        messages.extend(_layout_mismatch_diagnostic(name, "tabular"))
        return messages
    parsed = dict(zip(TABULAR_KEYS, match.groups()))
    messages.extend(_check_resolution_match(parsed, resolution, name))
    messages.extend(_check_year_match(parsed, start_date, "start_date", "start date", name))
    messages.extend(_check_year_match(parsed, end_date, "end_date", "end date", name))
    if messages:
        messages.append(_format_parsed(name, parsed))
    return messages


def _dispatch(
    layout: str,
    name: str,
    resolution: str | None,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
    frequency: datetime.timedelta | None,
    step_frequency: datetime.timedelta | None,
) -> list[str]:
    """Validate ``name`` against a single, explicit ``layout``."""
    if layout == "gridded":
        return _check_gridded_name(name, resolution, start_date, end_date, frequency)
    if layout == "trajectories":
        return _check_trajectories_name(name, resolution, start_date, end_date, frequency, step_frequency)
    if layout == "tabular":
        return _check_tabular_name(name, resolution, start_date, end_date)
    raise ValueError(f"Unknown layout {layout!r}; expected 'gridded', 'trajectories' or 'tabular'.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_dataset_name(
    name: str,
    resolution: str | None = None,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
    frequency: datetime.timedelta | None = None,
    step_frequency: datetime.timedelta | None = None,
    layout: str | None = None,
) -> list[str]:
    """Check if a dataset name follows the naming conventions.

    Three layout-dependent forms are recognised, each with its own regex:

    - ``layout="gridded"``      → ``...-<source>-<resolution>-<start>-<end>-<freq>-v<n>[-extra]``
      (e.g. ``aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5``).
    - ``layout="trajectories"`` → ``...-<source>-<resolution>-<start>-<end>-<date-freq>-<step-freq>-v<n>[-extra]``
      (e.g. ``aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3``).
    - ``layout="tabular"``      → ``...-<source>[-<resolution>]-<start>-<end>-v<n>[-extra]``
      (e.g. ``aifs-od-fc-oper-0001-mars-o96-2016-2025-v3`` or
      ``dop-ea-ofb-0001-1979-2023-v2-combined-aircraft``).

    The resolution token is mandatory for ``gridded`` and ``trajectories``
    and optional for ``tabular`` (which often has no meaningful spatial
    resolution — e.g. station observations).

    When ``layout`` is given the name is validated against that layout only.
    When ``layout`` is ``None`` the layout is unknown: the name is checked
    against *every* layout and is considered conformant (an empty list is
    returned) as soon as **any** layout accepts it.  Only when no layout
    accepts the name are messages returned — either the value messages of the
    single layout whose regex matched, or a generic non-conformant message.
    The layout is never guessed from the name to *decide* what to validate
    against (see :func:`_guess_layout_from_name`, which is used only to enrich
    diagnostics).

    Parameters
    ----------
    name : str
        The dataset name.
    resolution : str, optional
        Expected resolution.
    start_date : datetime.date, optional
        Expected start date (only the year is checked).
    end_date : datetime.date, optional
        Expected end date.
    frequency : datetime.timedelta, optional
        Expected frequency (date frequency for gridded layouts, base-date
        frequency for trajectory layouts).  Not used for tabular layouts.
    step_frequency : datetime.timedelta, optional
        Expected step frequency.  Only used for trajectory layouts.
    layout : str, optional
        ``"gridded"``, ``"trajectories"``, or ``"tabular"``.  When ``None``,
        the name is accepted if any layout matches (see above).

    Returns
    -------
    list of str
        Validation messages, empty when the name is conformant.
    """
    config = load_config().get("datasets", {})
    if config.get("ignore_naming_conventions", False):
        return []

    if layout is not None:
        return _dispatch(layout, name, resolution, start_date, end_date, frequency, step_frequency)

    # Layout unknown: accept the name if any layout accepts it.
    results: dict[str, list[str]] = {}
    for candidate in ("gridded", "trajectories", "tabular"):
        messages = _dispatch(candidate, name, resolution, start_date, end_date, frequency, step_frequency)
        if not messages:
            return []
        results[candidate] = messages

    # No layout accepted the name.  If exactly one layout's regex matched the
    # name (so it is that layout but some value disagrees), surface its
    # messages.  Otherwise report a generic non-conformant message.
    matched = [c for c in results if re.match(LAYOUT_PATTERNS[c], name)]
    if len(matched) == 1:
        return results[matched[0]]
    return _check_characters(name) + [_generic_non_conformant(name)]
