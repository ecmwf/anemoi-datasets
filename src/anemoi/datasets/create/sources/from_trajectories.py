# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""FromTrajectoriesSource — build a gridded dataset from forecast (basetime, step) pairs.

Wraps any single forecast-aware source (``mars:``, ``fdb:``, …) and, for each
validity time the pipeline asks for, picks a ``(basetime, step)`` pair that
produces it.  The inner source is then invoked with a typed ``ForecastDates``
argument and dispatches through ``@for_forecast_dates`` — no step injection
into the config, no post-retrieval filtering.

Recipe example::

    input:
      from-trajectories:
        - bases: "????-??-?? 00:00:00"   # only 00Z base times
          steps: 6/to/24/by/6            # steps 6, 12, 18, 24 h
        - mars:
            type: fc
            class: od
            expver: 0001
            param: [t, q]

The ``bases`` pattern uses ``fnmatch`` wildcard syntax against
``"%Y-%m-%d %H:%M:%S"`` formatted basetimes (``?`` matches any single
character, ``*`` matches any sequence).  Omit ``bases`` to accept any
basetime.  Omit ``steps`` to default to step 0 (analysis-like).
"""

import datetime
import fnmatch
import logging
from typing import Any

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source

from . import source_registry
from .mars.retrieval import expand_to_by

LOG = logging.getLogger(__name__)


@source_registry.register("from-trajectories")
class FromTrajectoriesSource(Source):
    """Build a gridded dataset from forecast (basetime, step) pairs.

    For each requested validity time, chooses a ``(basetime, step)`` pair that
    matches the configured ``bases`` pattern and whose step is in the
    configured ``steps`` list.  The resulting ``ForecastDates`` is passed to
    the inner source, which is expected to dispatch on
    ``@for_forecast_dates`` (as ``MarsSource`` does).

    Parameters
    ----------
    context : Any
        The build context.
    filter_config : dict
        A dict with optional keys ``bases`` (fnmatch pattern for basetime
        strings in ``"%Y-%m-%d %H:%M:%S"`` format) and ``steps`` (MARS step
        syntax, e.g. ``"6/to/24/by/6"`` or ``[6, 12, 18, 24]``).
    source_config : dict
        Inner source config in registry form, e.g.
        ``{"mars": {"class": "od", "param": ["t"]}}``.
    """

    def __init__(self, context: Any, filter_config: dict, source_config: dict) -> None:
        super().__init__(context)

        steps_raw = filter_config.get("steps", 0)
        expanded = expand_to_by(steps_raw)
        if not isinstance(expanded, list):
            expanded = [expanded]
        self.steps_hours: list[int] = [int(s) for s in expanded]

        self.bases_pattern: str | None = filter_config.get("bases")

        LOG.debug(
            "from-trajectories: steps=%s, bases=%r",
            self.steps_hours,
            self.bases_pattern,
        )

        if len(source_config) != 1:
            raise ValueError(
                f"from-trajectories: 'source' must be a single-key dict, "
                f"got keys: {list(source_config.keys())}"
            )

        from anemoi.datasets.create.sources import create_source

        self.inner = create_source(context, source_config)

    def _basetime_matches(self, basetime: datetime.datetime) -> bool:
        """Return ``True`` if *basetime* matches the ``bases`` pattern.

        Parameters
        ----------
        basetime : datetime.datetime
            The forecast base time to test.
        """
        if self.bases_pattern is None:
            return True
        return fnmatch.fnmatch(basetime.strftime("%Y-%m-%d %H:%M:%S"), self.bases_pattern)

    def _pick_basetime(self, valid_time: datetime.datetime) -> datetime.datetime:
        """Pick the first candidate basetime for *valid_time* satisfying the filters.

        Candidate basetimes are ``valid_time - step`` for each configured step
        (smallest step first).  The first candidate matching ``bases_pattern``
        wins.  Raises ``ValueError`` if none match.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time to resolve.
        """
        for step in self.steps_hours:
            candidate = valid_time - datetime.timedelta(hours=step)
            if self._basetime_matches(candidate):
                return candidate
        raise ValueError(
            f"from-trajectories: no basetime matches bases={self.bases_pattern!r} "
            f"for valid_time={valid_time.isoformat()} with steps={self.steps_hours}"
        )

    def _as_forecast_dates(self, argument: Any) -> ForecastDates:
        """Convert a ``ValidDates``-like argument into a ``ForecastDates``."""

        if isinstance(argument, ValidDates):
            valid_times = list(argument.dates)
        elif hasattr(argument, "dates"):
            valid_times = list(argument.dates)
        else:
            valid_times = list(argument)

        items = [(vt, self._pick_basetime(vt)) for vt in valid_times]
        return ForecastDates(items)

    def execute(self, dates: Any) -> Any:
        """Resolve ``dates`` into a ``ForecastDates`` and delegate to the inner source.

        Parameters
        ----------
        dates : Any
            Validity-time argument from the pipeline (``ValidDates`` or
            ``GroupOfDates``-like).
        """
        forecast_dates = self._as_forecast_dates(dates)
        LOG.debug(
            "from-trajectories: %d validity time(s) → %d forecast item(s)",
            len(forecast_dates.items),
            len(forecast_dates.items),
        )
        return self.inner.execute(forecast_dates)
