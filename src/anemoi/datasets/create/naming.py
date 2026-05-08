# (C) Copyright 2024 Anemoi contributors.
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


def guess_layout_from_name(name: str) -> str:
    """Guess the dataset layout from the syntactic form of its name.

    Returns ``"gridded"``, ``"trajectories"`` or ``"tabular"``.  Raises
    :class:`ValueError` if the name does not follow any known convention.

    Currently unused; provided for future call sites that need to recover
    the layout from a dataset name without opening the dataset.

    Parameters
    ----------
    name : str
        The dataset name (without extension).
    """
    if re.match(GRIDDED_PATTERN, name):
        return "gridded"
    if re.match(TRAJECTORIES_PATTERN, name):
        return "trajectories"
    if re.match(TABULAR_PATTERN, name):
        return "tabular"
    raise ValueError(f"Cannot guess layout from name {name!r}; does not follow any known convention.")


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


def _layout_mismatch_diagnostic(name: str, requested_layout: str) -> list[str]:
    """Diagnose why ``name`` failed the requested layout's regex.

    If the name follows another known convention, point the user at the
    matching layout.  Otherwise return a generic-non-conformant message.
    """
    try:
        actual = guess_layout_from_name(name)
    except ValueError:
        return [
            f"The dataset name '{name}' does not follow any known naming convention. "
            "See here for details: "
            "https://anemoi-registry.readthedocs.io/en/latest/naming-conventions.html"
        ]
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

    The function dispatches on ``layout`` to one of three layout-specific
    validators (``_check_gridded_name`` / ``_check_trajectories_name`` /
    ``_check_tabular_name``).  When a name does not match the requested
    layout's regex, the diagnostic checks whether it follows another
    convention (via :func:`guess_layout_from_name`) and reports
    accordingly.

    Tabular is never inferred — callers must opt in via ``layout="tabular"``,
    so a missing or typo'd ``frequency`` argument from another caller is
    not silently classified as a tabular name.  For backward compatibility,
    when ``layout`` is omitted it is inferred as ``trajectories`` if
    ``step_frequency`` is set, otherwise ``gridded``.

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
        inferred from the frequency arguments (see above).

    Returns
    -------
    list of str
        Validation messages, empty when the name is conformant.
    """
    config = load_config().get("datasets", {})
    if config.get("ignore_naming_conventions", False):
        return []

    if layout is None:
        # Tabular is never inferred — callers must opt in explicitly.
        layout = "trajectories" if step_frequency is not None else "gridded"

    if layout == "gridded":
        return _check_gridded_name(name, resolution, start_date, end_date, frequency)
    if layout == "trajectories":
        return _check_trajectories_name(name, resolution, start_date, end_date, frequency, step_frequency)
    if layout == "tabular":
        return _check_tabular_name(name, resolution, start_date, end_date)
    raise ValueError(f"Unknown layout {layout!r}; expected 'gridded', 'trajectories' or 'tabular'.")
