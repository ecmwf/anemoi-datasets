# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from earthkit.data import from_source

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source


@legacy_source(__file__)
def source(context: Optional[Any], dates: List[datetime], **kwargs: Any) -> Any:
    """Generates a source based on the provided context, dates, and additional keyword arguments.

    Parameters
    ----------
    context : Optional[Any]
        The context in which the source is generated.
    dates : List[datetime]
        A list of datetime objects representing the dates.
    **kwargs : Any
        Additional keyword arguments for the source generation.

    Returns
    -------
    Any
        The generated source.
    """
    name = kwargs.pop("name")
    context.trace("✅", f"from_source({name}, {dates}, {kwargs}")
    if kwargs["date"] == "$from_dates":
        kwargs["date"] = list({d.strftime("%Y%m%d") for d in dates})
    if kwargs["time"] == "$from_dates":
        kwargs["time"] = list({d.strftime("%H%M") for d in dates})
    return from_source(name, **kwargs)


execute = source

if __name__ == "__main__":
    import yaml

    config: Dict[str, Any] = yaml.safe_load(
        """
      name: mars
      class: ea
      expver: '0001'
      grid: 20.0/20.0
      levtype: sfc
      param: [2t]
      number: [0, 1]
      date: $from_dates
      time: $from_dates
    """
    )
    dates: List[str] = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    for f in source(None, dates, **config):
        print(f, f.to_numpy().mean())
