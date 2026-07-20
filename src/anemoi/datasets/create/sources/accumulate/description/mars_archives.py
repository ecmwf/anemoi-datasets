# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Factorised descriptions of well-known MARS archives (an omitted ``from:``)."""

from __future__ import annotations

import logging

from .union import FromTrajectories

LOG = logging.getLogger(__name__)


def _mars_archive_description(_class: str, _stream: str | None = None, _origin: str | None = None) -> dict:
    """Return the factorised archive description for a well-known MARS archive.

    Parameters
    ----------
    _class
        MARS class (e.g., 'ea', 'od', 'rr', 'l5').
    _stream
        MARS stream (e.g., 'oper', 'enda', 'elda', 'enfo'). Defaults to 'oper'.
    _origin
        MARS origin (e.g., 'se-al-ec', 'fr-ms-ec'). Defaults to None.

    Returns
    -------
    dict
        A ``from: {type: trajectories}`` payload.

    Raises
    ------
    NotImplementedError
        If the combination is not yet implemented.
    ValueError
        If the combination is unknown.
    """
    _stream = _stream or "oper"

    match (_class, _stream, _origin):
        case ("ea", "oper", _) | ("e6", "oper", _) | ("e6", "enda", _):
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "1h", "end": "18h", "frequency": "1h"},
                "accumulation": "1h",
            }
        case ("ea", "enda", _):
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "3h", "end": "18h", "frequency": "3h"},
                "accumulation": "3h",
            }
        case ("od", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
            return {
                "base_dates": {"times": [0, 12]},
                "steps": {"start": "1h", "end": "90h", "frequency": "1h"},
                "accumulation": "from-zero",
            }
        case ("od", "elda", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=elda&levtype=sfc&time=06%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-31&type=fc&class=od
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "1h", "end": "12h", "frequency": "1h"},
                "accumulation": "from-zero",
            }
        case ("od", "enfo", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=od&stream=enfo&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-31&time=06:00:00
            raise NotImplementedError("od-enfo archive description not implemented yet")

        # CERRA regional reanalysis
        case ("rr", _, "se-al-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?class=rr&expver=prod&origin=se-al-ec&stream=oper&type=fc&year=2020&month=aug&levtype=sfc
            # Irregular from-zero grid (hourly to 6h, then 3-hourly to 30h);
            # an explicit pair list is the only form that expresses it.
            return {
                "base_dates": {"times": [0]},
                "steps": [f"0-{s}" for s in (1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30)],
            }
        # CERRA regional reanalysis
        case ("rr", _, "fr-ms-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
            return {
                "base_dates": {"times": [0]},
                "steps": {"start": "1h", "end": "19h", "frequency": "3h"},
                "accumulation": "from-zero",
            }

        case ("l5", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=l5&stream=oper&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-25&time=00:00:00
            return {
                "base_dates": {"times": [0]},
                "steps": {"start": "1h", "end": "24h", "frequency": "1h"},
                "accumulation": "1h",
            }

        case _:
            raise ValueError(f"Unknown MARS configuration: class={_class}, stream={_stream}, origin={_origin}")


def infer_from_trajectories(source_name: str | None, source: dict | None) -> FromTrajectories:
    """Recognise a trajectory description from a MARS source config (an omitted ``from:``)."""
    assert None not in (source_name, source), "Source must be specified to recognise the description"
    if source_name != "mars":
        raise ValueError(
            "recognising the source data from the source (an omitted 'from:') is only supported "
            "for the 'mars' source; write the description explicitly for other sources"
        )

    _class, _stream, _origin = source.get("class"), source.get("stream"), source.get("origin")

    if _class is None:
        raise ValueError("accumulate: the archive description is taken from the mars source, but it has no 'class'")

    if (_stream is None) or (_origin is None):
        LOG.warning(
            f"Stream and/or origin unspecified for class {_class}, " f"stream and/or origin will be set as defaults.",
        )

    return FromTrajectories.model_validate(_mars_archive_description(_class, _stream, _origin))
