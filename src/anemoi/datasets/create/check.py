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
import warnings
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
from anemoi.utils.config import load_config
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


class DatasetName:
    """Validate and parse dataset names according to naming conventions."""

    def __init__(
        self,
        name: str,
        resolution: Optional[str] = None,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        frequency: Optional[datetime.timedelta] = None,
    ):
        """Initialize a DatasetName instance.

        Parameters
        ----------
        name : str
            The name of the dataset.
        resolution : Optional[str], optional
            The resolution of the dataset.
        start_date : Optional[datetime.date], optional
            The start date of the dataset.
        end_date : Optional[datetime.date], optional
            The end date of the dataset.
        frequency : Optional[datetime.timedelta], optional
            The frequency of the dataset.
        """
        self.name = name
        self.parsed = self._parse(name)
        print("---------------")
        print(self.parsed)
        print("---------------")

        self.messages = []

        config = load_config().get("datasets", {})

        if config.get("ignore_naming_conventions", False):
            # setting the env variable ANEMOI_CONFIG_DATASETS_IGNORE_NAMING_CONVENTIONS=1
            # will ignore the naming conventions
            return

        self.check_characters()
        self.check_parsed()
        self.check_resolution(resolution)
        self.check_frequency(frequency)
        self.check_start_date(start_date)
        self.check_end_date(end_date)

        if self.messages:
            self.messages.append(f"{self} is parsed as :" + "/".join(f"{k}={v}" for k, v in self.parsed.items()))

    @property
    def error_message(self) -> str:
        """Generate an error message based on the collected messages."""
        out = " And ".join(self.messages)
        if out:
            out[0].upper() + out[1:]
        return out

    def raise_if_not_valid(self, print: Callable = print) -> None:
        """Raise a ValueError if the dataset name is not valid.

        Parameters
        ----------
        print : Callable
            The function to use for printing messages.
        """
        if self.messages:
            for m in self.messages:
                print(m)
            raise ValueError(self.error_message)

    def _parse(self, name: str) -> dict:
        """Parse the dataset name into its components.

        Parameters
        ----------
        name : str
            The name of the dataset.

        Returns
        -------
        dict
            The parsed components of the dataset name.
        """
        pattern = r"^(\w+)-([\w-]+)-(\w+)-(\w+)-(\d\d\d\d)-(\d\d\d\d)-(\d+h|\d+m)-v(\d+)-?([a-zA-Z0-9-]+)?$"
        match = re.match(pattern, name)

        if not match:
            raise ValueError(f"the dataset name '{name}' does not follow naming convention. Does not match {pattern}")

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

        return parsed

    def __str__(self) -> str:
        """Return the string representation of the dataset name."""
        return self.name

    def check_parsed(self) -> None:
        """Check if the dataset name was parsed correctly."""
        if not self.parsed:
            self.messages.append(
                f"the dataset name {self} does not follow naming convention. "
                "See here for details: "
                "https://anemoi-registry.readthedocs.io/en/latest/naming-conventions.html"
            )

    def check_resolution(self, resolution: Optional[str]) -> None:
        """Check if the resolution matches the expected format.

        Parameters
        ----------
        resolution : str or None
            The expected resolution.
        """
        if self.parsed.get("resolution") and self.parsed["resolution"][0] not in "0123456789on":
            self.messages.append(
                f"the resolution {self.parsed['resolution'] } should start "
                f"with a number or 'o' or 'n' in the dataset name {self}."
            )

        if resolution is None:
            return
        resolution_str = str(resolution).replace(".", "p").lower()
        self._check_missing("resolution", resolution_str)
        self._check_mismatch("resolution", resolution_str)

    def check_characters(self) -> None:
        if not self.name.islower():
            self.messages.append(f"the {self.name} should be in lower case.")
        if "_" in self.name:
            self.messages.append(f"the {self.name} should use '-' instead of '_'.")
        for c in self.name:
            if not c.isalnum() and c not in "-":
                self.messages.append(f"the {self.name} should only contain alphanumeric characters and '-'.")

    def check_frequency(self, frequency: Optional[datetime.timedelta]) -> None:
        """Check if the frequency matches the expected format.

        Parameters
        ----------
        frequency : datetime.timedelta or None
            The expected frequency.
        """
        if frequency is None:
            return
        frequency_str = frequency_to_string(frequency)
        self._check_missing("frequency", frequency_str)
        self._check_mismatch("frequency", frequency_str)

    def check_start_date(self, start_date: Optional[datetime.date]) -> None:
        """Check if the start date matches the expected format.

        Parameters
        ----------
        start_date : datetime.date or None
            The expected start date.
        """
        if start_date is None:
            return
        start_date_str = str(start_date.year)
        self._check_missing("start_date", start_date_str)
        self._check_mismatch("start_date", start_date_str)

    def check_end_date(self, end_date: Optional[datetime.date]) -> None:
        """Check if the end date matches the expected format.

        Parameters
        ----------
        end_date : datetime.date or None
            The expected end date.
        """
        if end_date is None:
            return
        end_date_str = str(end_date.year)
        self._check_missing("end_date", end_date_str)
        self._check_mismatch("end_date", end_date_str)

    def _check_missing(self, key: str, value: str) -> None:
        """Check if a component is missing from the dataset name.

        Parameters
        ----------
        key : str
            The component key.
        value : str
            The expected value.
        """
        if value not in self.name:
            self.messages.append(f"the {key} is {value}, but is missing in {self.name}.")

    def _check_mismatch(self, key: str, value: str) -> None:
        """Check if a component value mismatches the expected value.

        Parameters
        ----------
        key : str
            The component key.
        value : str
            The expected value.
        """
        if self.parsed.get(key) and self.parsed[key] != value:
            self.messages.append(f"the {key} is {value}, but is {self.parsed[key]} in {self.name}.")


class StatisticsValueError(ValueError):
    """Custom error for statistics value issues."""

    pass


def check_data_values(
    arr: NDArray[Any], *, name: str, log: list = [], allow_nans: Union[bool, list, set, tuple, dict] = False
) -> None:
    """Check the values in the data array for validity.

    Parameters
    ----------
    arr : NDArray[Any]
        The data array to check.
    name : str
        The name of the data array.
    log : list, optional
        A list to log messages.
    allow_nans : bool or list or set or tuple or dict, optional
        Whether to allow NaNs in the data array.
    """
    shape = arr.shape

    if (isinstance(allow_nans, (set, list, tuple, dict)) and name in allow_nans) or allow_nans:
        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        warnings.warn(f"Empty array for {name} ({shape})")
        return

    assert arr.size > 0, (name, *log)

    min, max = arr.min(), arr.max()
    assert not (np.isnan(arr).any()), (name, min, max, *log)

    if min == 9999.0:
        warnings.warn(f"Min value 9999 for {name}")

    if max == 9999.0:
        warnings.warn(f"Max value 9999 for {name}")

    in_minus_1_plus_1 = dict(minimum=-1, maximum=1)
    limits = {
        "cos_latitude": in_minus_1_plus_1,
        "sin_latitude": in_minus_1_plus_1,
        "cos_longitude": in_minus_1_plus_1,
        "sin_longitude": in_minus_1_plus_1,
    }

    if name in limits:
        if min < limits[name]["minimum"]:
            warnings.warn(
                f"For {name}: minimum value in the data is {min}. "
                "Not in acceptable range [{limits[name]['minimum']} ; {limits[name]['maximum']}]"
            )
        if max > limits[name]["maximum"]:
            warnings.warn(
                f"For {name}: maximum value in the data is {max}. "
                "Not in acceptable range [{limits[name]['minimum']} ; {limits[name]['maximum']}]"
            )


def check_stats(minimum: float, maximum: float, mean: float, msg: str, **kwargs: Any) -> None:
    """Check if the mean value is within the min/max interval.

    Parameters
    ----------
    minimum : float
        The minimum value.
    maximum : float
        The maximum value.
    mean : float
        The mean value.
    msg : str
        The message to include in the error.
    **kwargs : Any
        Additional keyword arguments.
    """
    tolerance = (abs(minimum) + abs(maximum)) * 0.01
    if (mean - minimum < -tolerance) or (mean - minimum < -tolerance):
        raise StatisticsValueError(
            f"Mean is not in min/max interval{msg} : we should have {minimum} <= {mean} <= {maximum}"
        )
