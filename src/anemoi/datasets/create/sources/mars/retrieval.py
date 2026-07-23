# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Low-level MARS retrieval primitives.

These functions build and fire MARS (or CDS) requests directly, with no
dependency on the source class hierarchy.  They are the stable interface used
by sources that need to call MARS outside the normal ``source.execute()``
pipeline (e.g. hindcasts, recentre).
"""

import datetime
import re
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from anemoi.utils.humanize import did_you_mean
from earthkit.data import from_source
from earthkit.data.utils.availability import Availability

# ---------------------------------------------------------------------------
# MARS key whitelist
# ---------------------------------------------------------------------------

MARS_KEYS = [
    "accuracy",
    "activity",
    "anoffset",
    "area",
    "bitmap",
    "channel",
    "class",
    "database",
    "dataset",
    "date",
    "diagnostic",
    "direction",
    "domain",
    "expect",
    "experiment",
    "expver",
    "fcmonth",
    "fcperiod",
    "fieldset",
    "filter",
    "format",
    "frame",
    "frequency",
    "gaussian",
    "generation",
    "grid",
    "hdate",
    "ident",
    "instrument",
    "interpolation",
    "intgrid",
    "iteration",
    "level",
    "levelist",
    "levtype",
    "method",
    "model",
    "month",
    "number",
    "obsgroup",
    "obstype",
    "offsetdate",
    "offsettime",
    "optimise",
    "origin",
    "packing",
    "padding",
    "param",
    "quantile",
    "realization",
    "reference",
    "reportype",
    "repres",
    "resol",
    "resolution",
    "rotation",
    "step",
    "stream",
    "system",
    "target",
    "time",
    "truncation",
    "type",
    "year",
    "paramtype",
    "timespan",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def to_list(x: list | tuple | Any) -> list:
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _date_to_datetime(
    d: datetime.datetime | list | tuple | str,
) -> datetime.datetime | list[datetime.datetime]:
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, (list, tuple)):
        return [_date_to_datetime(x) for x in d]
    return datetime.datetime.fromisoformat(d)


def expand_to_by(x: str | int | list) -> str | int | list:
    if isinstance(x, (str, int)):
        return expand_to_by(str(x).split("/"))
    if len(x) == 3 and x[1] == "to":
        start = int(x[0])
        end = int(x[2])
        return list(range(start, end + 1))
    if len(x) == 5 and x[1] == "to" and x[3] == "by":
        start = int(x[0])
        end = int(x[2])
        by = int(x[4])
        return list(range(start, end + 1, by))
    return x


def _normalise_time(t: int | str) -> str:
    t = int(t)
    if t < 100:
        t * 100
    return f"{t:04d}"


# ---------------------------------------------------------------------------
# RequestFilter — per-step filter on computed base date / base time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestFilter:
    """Per-step filter applied to the *computed* MARS base date / base time.

    The MARS request dict carries data that is sent to MARS verbatim.  This
    object owns the orthogonal concept: predicates that decide which expanded
    requests to keep based on the base date and base time computed during
    expansion (``base = valid_time - step``).

    The YAML surface is the wildcard shorthand on ``date`` / ``time``:
    when ``date`` is a string containing ``?`` (e.g. ``"????-??-01"``), it
    is treated as a base-date pattern rather than a MARS ``date`` value, and
    the accompanying ``time`` (if any) is treated as a base-time filter.
    The internal keys ``user_date`` / ``user_time`` are not accepted in
    user configuration.

    Attributes
    ----------
    date :
        Compiled regex matching the base date string ``YYYYMMDD``, or ``None``
        for no date filter.
    time :
        Frozen set of normalised ``HHMM`` strings, or ``None`` for no time
        filter.
    """

    date: re.Pattern[str] | None = None
    time: frozenset[str] | None = None

    @classmethod
    def extract(cls, request: dict[str, Any]) -> tuple["RequestFilter", dict[str, Any]]:
        """Build a filter from a request dict and return the cleaned request.

        If ``date`` is a wildcard string (contains ``?``), pops it and any
        accompanying ``time`` into the filter.  The returned ``cleaned``
        dict contains only payload keys safe to forward to MARS (subject to
        the usual ``MARS_KEYS`` whitelist downstream).

        The legacy keys ``user_date`` / ``user_time`` are rejected with a
        clear ``ValueError`` so configs that still rely on them fail loudly
        rather than silently bypassing the filter logic.

        Parameters
        ----------
        request :
            A MARS request template.

        Returns
        -------
        tuple
            ``(filter, cleaned_request)``.
        """
        for legacy in ("user_date", "user_time"):
            if legacy in request:
                raise ValueError(
                    f"{legacy!r} is not a supported MARS request key. "
                    f"Use the wildcard shorthand instead: "
                    f"date: '????-??-01' (and optionally time: 0)."
                )

        cleaned = dict(request)

        raw_date = cleaned.get("date")
        if isinstance(raw_date, str) and "?" in raw_date:
            date_value = cleaned.pop("date")
            time_value = cleaned.pop("time", None)
            return (
                cls(
                    date=cls._compile_date(date_value),
                    time=cls._compile_time(time_value),
                ),
                cleaned,
            )

        return cls(), cleaned

    @staticmethod
    def _compile_date(value: Any) -> re.Pattern[str] | None:
        if value is None:
            return None
        if isinstance(value, int):
            value = str(value)
        elif isinstance(value, datetime.datetime):
            value = value.strftime("%Y%m%d")
        elif isinstance(value, str):
            pass
        else:
            raise ValueError(f"Invalid type for wildcard date: {value!r}")
        pattern = value.replace("-", "").replace("?", ".")
        return re.compile(f"^{pattern}$")

    @staticmethod
    def _compile_time(value: Any) -> frozenset[str] | None:
        if value is None:
            return None
        return frozenset(_normalise_time(t) for t in to_list(value))

    @property
    def is_empty(self) -> bool:
        return self.date is None and self.time is None

    def keep(self, base_date: str, base_time: str) -> bool:
        """Return True iff a request with this base_date/base_time passes the filter."""
        if self.date is not None and not self.date.match(base_date):
            return False
        if self.time is not None and base_time not in self.time:
            return False
        return True


def _expand_mars_request(
    request: dict[str, Any],
    valid_date: datetime.datetime,
) -> list[dict[str, Any]]:
    """Expand a MARS request template into concrete per-step request dicts.

    For each step, the MARS base date/time is computed as
    ``valid_date - step`` and written into the returned dicts as
    ``date`` and ``time``.  If the incoming request uses the wildcard-date
    shorthand (``date`` is a string containing ``?``), a ``RequestFilter``
    is extracted and applied to drop non-matching expansions.

    Parameters
    ----------
    request :
        MARS request template.
    valid_date :
        The validity datetime for which data is requested.  The MARS base
        date/time is derived from this by subtracting the step.

    Returns
    -------
    list[dict[str, Any]]
        The concrete per-step request dicts.
    """
    filter_, cleaned = RequestFilter.extract(request)

    user_step = to_list(expand_to_by(cleaned.get("step", [0])))

    requests = []
    for step in user_step:
        r = cleaned.copy()

        if isinstance(step, str) and "-" in step:
            assert step.count("-") == 1, step

        hours = int(str(step).split("-")[-1])
        base = valid_date - datetime.timedelta(hours=hours)  # MARS base date/time
        r.update(
            {
                "date": base.strftime("%Y%m%d"),
                "time": base.strftime("%H%M"),
                "step": step,
            }
        )
        for grid_key in ("grid", "rotation", "frame", "area", "bitmap", "resol"):
            if grid_key in r:
                if isinstance(r[grid_key], (list, tuple)):
                    r[grid_key] = "/".join(str(x) for x in r[grid_key])

        if not filter_.keep(r["date"], r["time"]):
            continue

        # SCDA stream auto-selection for ECMWF operational data:
        # the 06 and 18 UTC runs use stream "scda" instead of "oper"
        if r.get("class") == "od" and r.get("stream") == "oper":
            if int(r.get("time", 0)) in (600, 1800):
                r["stream"] = "scda"

        requests.append(r)

    return requests


def use_grib_paramid(r: dict[str, Any]) -> dict[str, Any]:
    """Convert parameter short names to GRIB parameter IDs."""
    from anemoi.utils.grib import shortname_to_paramid

    params = r["param"]
    if isinstance(params, str):
        params = params.split("/")
    assert isinstance(params, (list, tuple)), params
    params = [shortname_to_paramid(p) for p in params]
    r["param"] = "/".join(str(p) for p in params)
    return r


def _validate_params(requests: tuple) -> None:
    """Raise ValueError if any 'param' value is a YAML boolean artefact."""
    for r in requests:
        param = r.get("param", [])
        if not isinstance(param, (list, tuple)):
            param = [param]
        for p in param:
            if p is False:
                raise ValueError(
                    "'param' cannot be 'False'. If you wrote 'param: no' or 'param: off' in yaml, "
                    "you may want to use quotes?"
                )
            if p is None:
                raise ValueError(
                    "'param' cannot be 'None'. If you wrote 'param: no' in yaml, you may want to use quotes?"
                )
            if p is True:
                raise ValueError(
                    "'param' cannot be 'True'. If you wrote 'param: on' in yaml, you may want to use quotes?"
                )


def _validate_keys(requests: tuple) -> None:
    """Raise a clear ValueError for any key not in ``MARS_KEYS``.

    Runs on the raw request templates *before* factorisation, so that unknown
    keys -- including dict-valued ones, which would otherwise crash request
    factorisation with an opaque ``unhashable type: 'dict'`` error -- fail with
    a helpful "did you mean" message instead.
    """
    for r in requests:
        if not isinstance(r, dict):
            continue
        # Grouping wrapper ({"requests": [...]}): validate the nested requests.
        if "requests" in r:
            _validate_keys(r["requests"])
            continue
        for k, v in r.items():
            if k not in MARS_KEYS:
                raise ValueError(
                    f"⚠️ Unknown key {k}={v} in MARS request. Did you mean '{did_you_mean(k, MARS_KEYS)}' ?"
                )


def _fire_requests(context: Any, requests: list, use_cdsapi_dataset: str | None) -> Any:
    """Send a list of ready-to-fire request dicts to MARS or CDS."""
    ds = from_source("empty")
    for r in requests:
        r = {k: v for k, v in r.items() if v != ("-",)}

        if context.use_grib_paramid and "param" in r:
            r = use_grib_paramid(r)

        for k, v in r.items():
            if k not in MARS_KEYS:
                raise ValueError(
                    f"⚠️ Unknown key {k}={v} in MARS request. Did you mean '{did_you_mean(k, MARS_KEYS)}' ?"
                )
        try:
            if use_cdsapi_dataset:
                ds = ds + from_source("cds", use_cdsapi_dataset, r)
            else:
                ds = ds + from_source("mars", **r)
        except Exception as e:
            if "File is empty:" not in str(e):
                raise
    return ds


# ---------------------------------------------------------------------------
# Reusable primitives below:
# These functions are used by multiple sources, and are the interface for
# building and firing MARS/CDS requests.
# They are not public API though.
# ---------------------------------------------------------------------------


def factorise_requests(
    dates: list[datetime.datetime],
    *requests: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Factorise a set of per-validity-date MARS requests into a compressed request list."""
    if isinstance(requests, tuple) and len(requests) == 1 and "requests" in requests[0]:
        requests = requests[0]["requests"]

    # Extract filters once per request: the cleaned request is what the
    # literal-equality pre-filter below should reason about (so that a
    # wildcard 'date' in the original is not treated as a literal value).
    prepared = [(req, *RequestFilter.extract(req)) for req in requests]

    updates = []
    for d in sorted(dates):
        for original, _filter, cleaned in prepared:
            if (
                ("date" in cleaned)
                and ("time" in cleaned)
                and d.strftime("%Y%m%d%H%M") != (str(cleaned["date"]) + str(cleaned["time"]).zfill(4))
            ):
                continue
            # Pass the *original* request (with filter keys still attached) so
            # that _expand_mars_request can re-extract and apply the filter.
            updates += _expand_mars_request(original, valid_date=d)

    if not updates:
        return
    compressed = Availability(updates)
    for r in compressed.iterate():
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                r[k] = v[0]
        yield r


def compress_prebuilt_requests(
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise grid/area fields and compress pre-built requests via Availability.

    Unlike ``factorise_requests``, this function expects ``date``, ``time``, and
    ``step`` to already be set in each request.  It does not expand validity
    dates into base-time/step pairs.  Use it (or ``fire_prebuilt_requests``) for
    forecast-date, interval, hindcast, and FDB request lists.
    """
    normalised = []
    for r in requests:
        r = r.copy()
        for grid_key in ("grid", "rotation", "frame", "area", "bitmap", "resol"):
            if grid_key in r and isinstance(r[grid_key], (list, tuple)):
                r[grid_key] = "/".join(str(x) for x in r[grid_key])
        if r.get("class") == "od" and r.get("stream") == "oper":
            if int(r.get("time", 0)) in (600, 1800):
                r["stream"] = "scda"
        normalised.append(r)

    compressed = Availability(normalised)
    result = []
    for r in compressed.iterate():
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                r[k] = v[0]
        result.append(r)
    return result


def fire_prebuilt_requests(
    context: Any,
    requests: list[dict[str, Any]],
    use_cdsapi_dataset: str | None = None,
) -> Any:
    """Validate, normalise, compress, and fire pre-built MARS/CDS requests.

    Use instead of ``execute_mars_request`` when ``date``, ``time``, and
    ``step`` are already set in each request dict (forecast dates, accumulation
    intervals, hindcasts).
    """
    _validate_params(requests)
    _validate_keys(requests)
    compressed = compress_prebuilt_requests(requests)
    context.trace("\u2705", f"Will run {len(compressed)} prebuilt requests")
    for r in compressed:
        context.trace("\u2705", f"mars {r}")
    return _fire_requests(context, compressed, use_cdsapi_dataset)


def execute_mars_request(
    context: Any,
    dates: list[datetime.datetime],
    *requests: dict[str, Any],
    use_cdsapi_dataset: str | None = None,
    **kwargs: Any,
) -> Any:
    """Execute MARS requests for a list of validity dates.

    Parameters
    ----------
    context :
        Build context (provides ``trace``, ``use_grib_paramid``).
    dates :
        Validity times to retrieve data for.
    requests :
        Base MARS request dicts.  If empty, ``kwargs`` is used as the
        single request.
    use_cdsapi_dataset :
        If set, use the CDS API with this dataset name instead of MARS.
    kwargs :
        Used as the single request when ``requests`` is empty.

    Returns
    -------
    Any
        The retrieved data.
    """
    if not requests:
        requests = [kwargs]

    _validate_params(requests)
    _validate_keys(requests)

    requests = list(factorise_requests(dates, *requests))

    context.trace("✅", f"{[str(d) for d in dates]}, {len(dates)}")
    context.trace("✅", f"Will run {len(requests)} requests")
    for r in requests:
        context.trace("✅", f"mars {r}")

    return _fire_requests(context, requests, use_cdsapi_dataset)
