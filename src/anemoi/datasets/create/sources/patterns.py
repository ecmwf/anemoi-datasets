# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import glob
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple

from earthkit.data.utils.patterns import Pattern


def _expand(paths: List[str]) -> Generator[str, None, None]:
    """Expand the given paths to include all matching file paths.

    Parameters
    ----------
    paths : List[str]
        List of paths to expand.

    Returns
    -------
    Generator[str]
        Expanded file paths.
    """
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if path.startswith("file://"):
            path = path[7:]

        if path.startswith("http://"):
            yield path
            continue

        if path.startswith("https://"):
            yield path
            continue

        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


def iterate_patterns(
    path: str, dates: List[datetime.datetime], **kwargs: Any
) -> Generator[Tuple[str, List[str]], None, None]:
    """Iterate over patterns and expand them with given dates and additional keyword arguments.

    Parameters
    ----------
    path : str
        The pattern path to iterate over.
    dates : List[datetime.datetime]
        List of datetime objects to substitute in the pattern.
    **kwargs : Any
        Additional keyword arguments to substitute in the pattern.

    Returns
    -------
    Generator[Tuple[str, List[str]]]
        The expanded path and list of ISO formatted dates.
    """
    given_paths = path if isinstance(path, list) else [path]

    dates = [d.isoformat() for d in dates]
    if len(dates) > 0:
        kwargs["date"] = dates

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(**kwargs)
        for path in _expand(paths):
            yield path, dates
