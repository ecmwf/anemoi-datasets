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
from typing import Any

import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.create.sources import source_registry

from .accumulation_utils.intervals import Link
from .accumulation_utils.intervals import LinksCollection
from .accumulation_utils.intervals import Vector
from .accumulation_utils.intervals import build_catalogue
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


# flow of information in accumulation:
#
# main accumulate source creates Accumulator objects for each param/member/valid_date
#    Accumulator will be a subclass of Reducer (as Averager or Maxer would also be)
#
# first, the Accumulators should create a list of requests to get all the needed data from the source
#    this is done by looping on the accumulators and getting their requests
#    each accumulator has a Cataloguer object that knows which intervals are available for the source
#      each accumulator chat with their Cataloguer to get the intervals they need : intervals = self.cataloguer.get_intervals(date_start, date_end, source_parameters, extra1=, extra2=..)
#      each interval has the full request needed to give to the source to retrieve the data for this interval
#      each interval also knows how to find its data in the source response
#
# once all requests are built, the source is queried and data retrieved
#
# then, each field from the source is sent to each accumulator
#    the accumulator checks if the field is needed, asking its intervals if some intervals need this field
#    if yes, the accumulator accumulated the values (asking the interval how to do it)
#    once all intervals are done, the accumulator assembles the final output field and writes in the result


class Accumulator:
    values: NDArray | None = None
    completed: bool = False

    def __init__(self, vector: Vector, key: dict[str, Any], coverage):
        """Accumulator object for a given param/member/valid_date"""
        self.vector = vector
        self.coverage = coverage
        self.todo = [v for v in coverage]

        self.period = vector.max - vector.min
        self.valid_date = vector.max

        assert isinstance(key, dict)
        self.key = key
        for k in ["date", "time", "step"]:
            if k in self.key:
                raise ValueError(f"Cannot use {k} for accumulation")

        self.values = None  # will hold accumulated values array

        # The accumulator only accumulates fields matching its metadata
        # and does not know about the rest

        # key contains the mars request parameters except the one related to the time
        # A mars request is a dictionary with three categories of keys:
        #   - the ones related to the time (date, time, step)
        #   - the ones related to the data (param, stream, levtype, expver, number, ...)
        #   - the ones related to the processing to be done (grid, area, ...)

    def is_field_needed(self, field: Any):
        """Check whether the given field is needed by the accumulator (correct param, etc...)"""
        for k, v in self.key.items():
            metadata = field.metadata(k)
            if k == "number":  # bug in eccodes has number=None randomly
                if metadata is None:
                    metadata = 0
                if v is None:
                    v = 0
            if metadata != v:
                LOG.debug(f"{self} does not need field {field} because of {k}={metadata} not {v}")
                return False
        return True

    def is_complete(self, **kwargs) -> bool:
        """Check whether the accumulation is complete (all intervals have been processed)"""
        return not self.todo

    def compute(self, field: Any, values: NDArray, link: Link) -> None:
        """Verify the field time metadata, find the associated interval
        and perform accumulation with the values array on this interval.
        Note: values have been extracted from field before the call to `compute`,
        so values are read from field only once.

        Parameters:
        ----------
        field: Any
            An earthkit-data-like field
        values: NDArray
            Values from the field, will be added to the held values array

        Return
        ------
        None
        """
        vector = link.vector
        if self.completed:
            raise ValueError(f"Accumulator {self} already completed, cannot process vector {vector}")

        # actual accumulation computation
        if self.values is None:
            self.values = values
        else:
            self.values += values

        if vector not in self.todo:
            raise ValueError(f"Vector {vector} not in todo list of accumulator {self}")
        self.todo.remove(vector)

        if self.is_complete():
            # all intervals for accumulation have been processed
            # final list of outputs is ready to be updated
            self.write(field)  # field is used as a template
            self.completed = True

    def field_to_vector(self, field: Any):
        valid_date = field.metadata("valid_date")
        step = field.metadata("step")
        date = valid_date - datetime.timedelta(hours=step)
        return Vector(date=date, step=step)

    def write(self, field: Any) -> None:
        """Writing output inside a new field at the end of the accumulation.
        The field is simply used as a template.

        Parameters:
        ----------
        field: Any
            An earthkit-data-like field

        Return
        ------
        None
        """
        assert self.is_complete(), self.todo

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        if np.any(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )

        startStep = datetime.timedelta(hours=0)
        endStep = self.vector.max - self.vector.min

        accumfield = new_field_from_numpy(
            self.values, template=field, startStep=startStep, endStep=endStep, stepType="accum"
        )
        self.out = new_field_with_valid_datetime(accumfield, self.valid_date)

    def __repr__(self):
        key = ", ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.__class__.__name__}({self.vector}, {key}, {len(self.coverage)-len(self.todo)}/{len(self.coverage)} already accumulated)"


def _compute_accumulations(
    context: Any,
    dates: list[datetime.datetime],
    period: datetime.timedelta,
    source: Any,
    hints: dict[str, Any] | None = None,
) -> Any:
    """Concrete accumulation logic.

    - identify the needed intervals for each date/parameter/member defined in recipe
    - fetch the source data via a database (mars or grib-index)
    - create Accumulator objects and fill them will accumulated values from source data
    - return the datasource with accumulated values

    Parameters:
    ----------
    context: Any,
        The dataset building context (will be updated with trace of accumulation)
    dates: list[datetime.datetime]
        The list of valid dates on which to perform accumulations.
    source: Any,
    period: datetime.timedelta,
        The interval over which to accumulate (user-defined)

    Return
    ------
    The accumulated datasource for all dates, parameters, members.

    """

    print("💬 source for accumulations:", source)

    cataloguer = build_catalogue(context, hints, source)

    # building accumulators :
    # one accumulator per valid date, per output parm and ensemble member
    # i.e. one accumulator per output field
    accumulators = []
    for valid_date in dates:
        vector = Vector.from_valid_date_and_period(valid_date, period)
        coverage = cataloguer.covering_vectors(vector)
        for key in cataloguer.get_all_keys():
            accumulators.append(Accumulator(vector, key=key, coverage=coverage))

    links = LinksCollection()
    for accumulator in accumulators:
        print(f"💬 {accumulator} will need:")
        for v in accumulator.coverage:
            link = Link(vector=v, accumulator=accumulator, catalogue=cataloguer)
            print("  💬 ", link)
            links.append(link)

    for field, values, link in cataloguer.retrieve_fields(links):
        link.accumulator.compute(field, values, link)

    for a in accumulators:
        if not a.is_complete():
            a.is_complete(debug=True)
            LOG.error(f"Accumulator incomplete: {a}")
            LOG.error(f"{len(a.done)} Vectors received:")
            for v in a.done:
                LOG.error(f"  {v}")
            raise AssertionError("missing periods for accumulator, stopping.")

    # final data source
    ds = new_fieldlist_from_list([a.out for a in accumulators])

    # the resulting datasource has one field per valid date, parameter and ensemble member
    keys = list(cataloguer.get_all_keys())
    assert len(ds) / len(keys) == len(dates), (len(ds), len(keys), len(dates))

    return ds


@source_registry.register("accumulate")
class Accumulations2Source(LegacySource):

    @staticmethod
    def _execute(
        context: Any,
        dates: list[datetime.datetime],
        source: Any,
        period,
        hints=None,
        data_accumulation_period=None,
    ) -> Any:
        """Accumulation source callable function.
        Read the recipe for accumulation in the request dictionary, check main arguments and call computation.

        Parameters:
        ----------
        context: Any,
            The dataset building context (will be updated with trace of accumulation)
        dates: list[datetime.datetime]
            The list of valid dates on which to perform accumulations.
        source: Any,
            The accumulation source
        request: dict[str,Any]
            The parameters from the accumulation recipe

        Return
        ------
        The accumulated data source.

        """
        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")
        period = frequency_to_timedelta(period)
        if hints is None:
            hints = {}

        if data_accumulation_period is not None:
            data_accumulation_period = frequency_to_timedelta(data_accumulation_period)
            hints["accumulation_period"] = data_accumulation_period

        return _compute_accumulations(
            context,
            dates,
            source=source,
            period=period,
            hints=hints,
        )
