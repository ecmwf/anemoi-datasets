# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import re
import warnings

import numpy as np
from anemoi.utils.dates import frequency_to_string

LOG = logging.getLogger(__name__)


class DatasetName:
    def __init__(
        self,
        name,
        resolution=None,
        start_date=None,
        end_date=None,
        frequency=None,
    ):
        self.name = name
        self.parsed = self._parse(name)
        print("---------------")
        print(self.parsed)
        print("---------------")

        self.messages = []

        self.check_parsed()
        self.check_resolution(resolution)
        self.check_frequency(frequency)
        self.check_start_date(start_date)
        self.check_end_date(end_date)

        if self.messages:
            self.messages.append(f"{self} is parsed as :" + "/".join(f"{k}={v}" for k, v in self.parsed.items()))

    @property
    def error_message(self):
        out = " And ".join(self.messages)
        if out:
            out = out[0].upper() + out[1:]
        return out

    def raise_if_not_valid(self, print=print):
        if self.messages:
            for m in self.messages:
                print(m)
            raise ValueError(self.error_message)

    def _parse(self, name):
        pattern = r"^(\w+)-([\w-]+)-(\w+)-(\w+)-(\d\d\d\d)-(\d\d\d\d)-(\d+h)-v(\d+)-?([a-zA-Z0-9-]+)?$"
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

    def __str__(self):
        return self.name

    def check_parsed(self):
        if not self.parsed:
            self.messages.append(
                f"the dataset name {self} does not follow naming convention. "
                "See here for details: "
                "https://confluence.ecmwf.int/display/DWF/Datasets+available+as+zarr"
            )

    def check_resolution(self, resolution):
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

    def check_frequency(self, frequency):
        if frequency is None:
            return
        frequency_str = frequency_to_string(frequency)
        self._check_missing("frequency", frequency_str)
        self._check_mismatch("frequency", frequency_str)

    def check_start_date(self, start_date):
        if start_date is None:
            return
        start_date_str = str(start_date.year)
        self._check_missing("start_date", start_date_str)
        self._check_mismatch("start_date", start_date_str)

    def check_end_date(self, end_date):
        if end_date is None:
            return
        end_date_str = str(end_date.year)
        self._check_missing("end_date", end_date_str)
        self._check_mismatch("end_date", end_date_str)

    def _check_missing(self, key, value):
        if value not in self.name:
            self.messages.append(f"the {key} is {value}, but is missing in {self.name}.")

    def _check_mismatch(self, key, value):
        if self.parsed.get(key) and self.parsed[key] != value:
            self.messages.append(f"the {key} is {value}, but is {self.parsed[key]} in {self.name}.")


class StatisticsValueError(ValueError):
    pass


def check_data_values(arr, *, name: str, log=[], allow_nans=False):

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


def check_stats(minimum, maximum, mean, msg, **kwargs):
    tolerance = (abs(minimum) + abs(maximum)) * 0.01
    if (mean - minimum < -tolerance) or (mean - minimum < -tolerance):
        raise StatisticsValueError(
            f"Mean is not in min/max interval{msg} : we should have {minimum} <= {mean} <= {maximum}"
        )
