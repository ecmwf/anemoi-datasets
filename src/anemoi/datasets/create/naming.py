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
) -> list[str]:
    """Check if a dataset name follows the naming conventions.

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
        The expected frequency of the dataset.

    Returns
    -------
    list[str]
        A list of error messages, or an empty list if the name is valid.
    """
    config = load_config().get("datasets", {})
    if config.get("ignore_naming_conventions", False):
        return []

    messages = []

    # check_characters
    if not name.islower():
        messages.append(f"The dataset name '{name}' should be in lower case.")
    if "_" in name:
        messages.append(f"The dataset name '{name}' should use '-' instead of '_'.")
    for c in name:
        if not c.isalnum() and c not in "-":
            messages.append(f"The dataset name '{name}' should only contain alphanumeric characters and '-'.")

    # parse
    pattern = r"^(\w+)-([\w-]+)-(\w+)-(\w+)-(\d\d\d\d)-(\d\d\d\d)-(\d+h|\d+m)-v(\d+)-?([a-zA-Z0-9-]+)?$"
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
