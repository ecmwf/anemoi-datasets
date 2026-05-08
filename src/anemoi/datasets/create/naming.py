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

    Three layout-dependent forms are recognised:

    - Single-frequency form (``layout="gridded"``):
      ``...-<start-year>-<end-year>-<freq>-v<n>[-extra]``,
      e.g. ``aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5``.
    - Two-frequency form (``layout="trajectories"``):
      ``...-<start-year>-<end-year>-<date-freq>-<step-freq>-v<n>[-extra]``,
      e.g. ``aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3``.
    - No-frequency form (``layout="tabular"``):
      ``...-<start-year>-<end-year>-v<n>[-extra]``,
      e.g. ``aifs-od-fc-oper-0001-mars-o96-2016-2025-v3``.

    Each call site passes its ``layout`` explicitly. The validator rejects
    names whose frequency count does not match the layout (e.g. a gridded
    layout name with two frequencies, or a tabular layout name with one).

    The ``layout`` argument is required to validate against the tabular
    no-frequency form: tabular dataset names are not the silent default,
    they must be opted into. For backward compatibility, when ``layout``
    is omitted it is inferred from the frequency arguments:
    ``step_frequency`` set → ``trajectories``; otherwise ``gridded``.

    Parameters
    ----------
    name : str
        The name of the dataset.
    resolution : Optional[str], optional
        The expected resolution of the dataset.
    start_date : Optional[datetime.date], optional
        The expected start date of the dataset.
    end_date : Optional[datetime.date], optional
        The expected end date of the dataset.
    frequency : Optional[datetime.timedelta], optional
        The expected frequency of the dataset (date frequency for gridded
        layouts, base-date frequency for trajectory layouts). Not used for
        tabular layouts.
    step_frequency : Optional[datetime.timedelta], optional
        The expected step frequency. Only meaningful for trajectory layouts.
    layout : Optional[str], optional
        ``"gridded"``, ``"trajectories"``, or ``"tabular"``. When ``None``,
        inferred from the frequency arguments (see above).

    Returns
    -------
    list[str]
        A list of error messages, or an empty list if the name is valid.
    """
    config = load_config().get("datasets", {})
    if config.get("ignore_naming_conventions", False):
        return []

    if layout is None:
        # Tabular is never inferred — callers must opt in explicitly.
        layout = "trajectories" if step_frequency is not None else "gridded"

    if layout not in ("gridded", "trajectories", "tabular"):
        raise ValueError(f"Unknown layout {layout!r}; expected 'gridded', 'trajectories' or 'tabular'.")

    messages = []

    # check_characters
    if not name.islower():
        messages.append(f"The dataset name '{name}' should be in lower case.")
    if "_" in name:
        messages.append(f"The dataset name '{name}' should use '-' instead of '_'.")
    for c in name:
        if not c.isalnum() and c not in "-":
            messages.append(f"The dataset name '{name}' should only contain alphanumeric characters and '-'.")

    # parse — both frequency slots are optional so the same regex matches
    # the no-frequency (tabular), single-frequency (gridded), and
    # two-frequency (trajectories) forms.  The number of captured frequency
    # tokens is then validated against the expected layout.
    pattern = (
        r"^(\w+)-([\w-]+)-(\w+)-(\w+)-(\d\d\d\d)-(\d\d\d\d)"
        r"(?:-(\d+h|\d+m))?(?:-(\d+h|\d+m))?-v(\d+)-?([a-zA-Z0-9-]+)?$"
    )
    match = re.match(pattern, name)
    parsed = {}
    if match:
        keys = [
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
        parsed = {k: v for k, v in zip(keys, match.groups())}

    # check_parsed
    if not parsed:
        messages.append(
            f"The dataset name '{name}' does not follow the naming convention. "
            "See here for details: "
            "https://anemoi-registry.readthedocs.io/en/latest/naming-conventions.html"
        )

    # check layout / form coherence
    n_freqs = sum(1 for k in ("frequency", "step_frequency") if parsed.get(k) is not None)
    expected_n_freqs = {"tabular": 0, "gridded": 1, "trajectories": 2}[layout]
    if parsed and n_freqs != expected_n_freqs:
        forms = {
            0: "no-frequency form '-v<n>'",
            1: "single-frequency form '-<freq>-v<n>'",
            2: "two-frequency form '-<date-freq>-<step-freq>-v<n>'",
        }
        observed = forms[n_freqs]
        expected = forms[expected_n_freqs]
        messages.append(
            f"The dataset name '{name}' uses the {observed}, but {layout} "
            f"layouts require the {expected}."
        )

    # check_resolution
    if parsed.get("resolution") and parsed["resolution"][0] not in "0123456789on":
        messages.append(
            f"The resolution '{parsed['resolution']}' should start "
            f"with a number, 'o', or 'n' in the dataset name '{name}'."
        )
    if resolution is not None:
        resolution_str = str(resolution).replace(".", "p").lower()
        if resolution_str not in name:
            messages.append(f"The resolution is '{resolution_str}', but it is missing in '{name}'.")
        if parsed.get("resolution") and parsed["resolution"] != resolution_str:
            messages.append(f"The resolution is '{resolution_str}', but it is '{parsed['resolution']}' in '{name}'.")

    # check_frequency
    if frequency is not None:
        frequency_str = frequency_to_string(frequency)
        if frequency_str not in name:
            messages.append(f"The frequency is '{frequency_str}', but it is missing in '{name}'.")
        if parsed.get("frequency") and parsed["frequency"] != frequency_str:
            messages.append(f"The frequency is '{frequency_str}', but it is '{parsed['frequency']}' in '{name}'.")

    # check_step_frequency (trajectory layouts only)
    if step_frequency is not None:
        step_frequency_str = frequency_to_string(step_frequency)
        if parsed.get("step_frequency") and parsed["step_frequency"] != step_frequency_str:
            messages.append(
                f"The step frequency is '{step_frequency_str}', but it is "
                f"'{parsed['step_frequency']}' in '{name}'."
            )

    # check_start_date
    if start_date is not None:
        start_date_str = str(start_date.year)
        if start_date_str not in name:
            messages.append(f"The start date is '{start_date_str}', but it is missing in '{name}'.")
        if parsed.get("start_date") and parsed["start_date"] != start_date_str:
            messages.append(f"The start date is '{start_date_str}', but it is '{parsed['start_date']}' in '{name}'.")

    # check_end_date
    if end_date is not None:
        end_date_str = str(end_date.year)
        if end_date_str not in name:
            messages.append(f"The end date is '{end_date_str}', but it is missing in '{name}'.")
        if parsed.get("end_date") and parsed["end_date"] != end_date_str:
            messages.append(f"The end date is '{end_date_str}', but it is '{parsed['end_date']}' in '{name}'.")

    if messages:
        messages.append(f"'{name}' is parsed as: " + "/".join(f"{k}={v}" for k, v in parsed.items()) + ".")

    return messages
