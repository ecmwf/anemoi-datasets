# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Annotated
from typing import Any
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import create_source
from anemoi.datasets.create.sources import source_registry

from .bufr_support.bufr_reader import BUFRReader
from .bufr_support.bufr_to_df import BUFRToDataFrame
from .bufr_support.bufr_to_df import bufr_to_dataframe_parallel

LOG = logging.getLogger(__name__)


BUFRCondition = Annotated[list[Any], Field(min_length=2, max_length=2)]


class BUFRExtractSchema(BaseModel):
    """Validation schema for BUFR extraction options."""

    model_config = ConfigDict(extra="forbid")

    latitude: str = "latitude"
    longitude: str = "longitude"
    per_report: dict[str, str] = Field(default_factory=dict)
    preselect_msg_header: dict[str, BUFRCondition] | None = None
    preselect_msg_data: dict[str, BUFRCondition] | None = None
    datetime_position_prefix: str = ""
    per_datum: dict[str, dict[str, str | int]] | None = None
    per_datum_format: Literal["long", "wide"] = "wide"


class BUFRSourceSchema(BaseModel):
    """Validation schema for the `bufr` source in recipes."""

    model_config = ConfigDict(extra="forbid")

    source: dict[Literal["mars"], dict[str, Any]] = Field(..., min_length=1, max_length=1)
    extract: BUFRExtractSchema
    num_processes: int = Field(default=1, ge=1)


@source_registry.register("bufr")
class BUFRSource(Source):
    """Read data from BUFR files and converts to a DataFrame.

    This source retrieves BUFR data (currently via MARS), unpacks the messages,
    and extracts per-report and per-datum variables into a tabular format.
    Messages can optionally be filtered before or after unpacking using
    header and data section conditions.

    Parameters:
    ----------
    context: Any
        The context object provided by the pipeline.
    source : dict
        A single-entry dictionary specifying the data source. Currently only
        ``mars`` is supported. The nested dictionary is forwarded to the MARS
        retrieval (i.e. uses the same specfication as the MARS source).
    extract : dict
        Configuration forwarded to :class:`BUFRToDataFrame`. Accepted keys:

        latitude : str, optional
            BUFR key to use for the canonical ``latitude`` output column.
            Defaults to ``"latitude"``.
        longitude : str, optional
            BUFR key to use for the canonical ``longitude`` output column.
            Defaults to ``"longitude"``.
        per_report : dict
            Mapping of output column names to BUFR key names for values that
            occur once per observation report (subset). ``latitude`` and
            ``longitude`` must not be used as keys here.
        per_datum : dict, optional
            Variables with multiple values per report. Structure depends on
            ``per_datum_format``:

            * ``"wide"`` – ``{bufr_key: {col_name: slice_str, ...}, ...}``
              where ``slice_str`` is a slice expression (e.g. ``"0:20:1"``)
              applied to the reshaped 2-D array.
            * ``"long"`` – ``{bufr_key: {name: col_name, start_index: int}, ...}``
              where each BUFR key is unpacked into a single column and the
              per-report rows are repeated accordingly.
        per_datum_format : str, optional
            Either ``"wide"`` (default) or ``"long"``. Controls how
            ``per_datum`` columns are arranged in the resulting DataFrame.
        preselect_msg_header : dict, optional
            Header-section selection applied _before_ the message is unpacked.
            Each entry maps a BUFR key to a ``[operator, value]`` pair.
            Supported operators: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``,
            ``in``, ``not in``.
        preselect_msg_data : dict, optional
            Data-section selection applied _after_ the message is unpacked.
            Same format as ``preselect_msg_header``.
        datetime_position_prefix : str, optional
            Prefix prepended to datetime key names (``year``, ``month``,
            ``day``, ``hour``, ``minute``, ``second``). Defaults to an empty
            string.
    num_processes : int, optional
        Number of parallel worker processes for reading BUFR messages.
        Defaults to ``1``.

    Examples
    --------
    Basic satellite radiance extraction (per-report only):

    .. code-block:: yaml

        input:
          bufr:
            num_processes: 1
            source:
              mars:
                class: od
                expver: 0001
                stream: "DCDA/LWDA"
                type: ai
                obstype: atov
                time: "00"
            extract:
              prefilter_msg_header:
                satelliteID: ["==", 223]
              prefilter_msg_data:
                satelliteSensorIndicator: ["==", 11]
              latitude: latitude
              longitude: longitude
              per_report:
                azimuth: bearingOrAzimuth
                zenith: satelliteZenithAngle
                obsvalue_rawbt_1: "#1#brightnessTemperature"
                obsvalue_rawbt_2: "#2#brightnessTemperature"

    Wide format – extracting sliced data as separate columns:

    .. code-block:: yaml

        input:
          bufr:
            num_processes: 1
            source:
              mars:
                class: od
                expver: 0001
                stream: DCDA/LWDA
                type: ai
                obstype: iasi
            extract:
              latitude: latitude
              longitude: longitude
              per_report:
                zenith: satelliteZenithAngle
              per_datum:
                nonNormalizedPrincipalComponentScore:
                  obsvalue_pc_lw: "0:20:1"
                  obsvalue_pc_mw: "90:110:1"

    Long format – repeating per-report rows for each datum:

    .. code-block:: yaml

        input:
          bufr:
            num_processes: 1
            source:
              mars:
                class: od
                expver: 0001
                stream: DCDA/LWDA
                type: ai
                obstype: gpsro
                times: "00/06/12/18"
            extract:
              latitude: latitude
              longitude: longitude
              per_report:
                satellite_id: satelliteIdentifier
                radcurv: earthLocalRadiusOfCurvature
              per_datum_format: "long"
              per_datum:
                latitude:
                  name: latitude
                  start_index: 1
                longitude:
                  name: longitude
                  start_index: 1
                bendingAngle:
                  name: obsvalue_bend_angle_0
                impactParameter:
                  name: vertco_reference_1
    """

    schema = BUFRSourceSchema

    def __init__(self, context, *, source: dict, extract: dict, num_processes: int = 1):
        super().__init__(context)
        self.source = self._create_source(source)
        extract_cfg = BUFRExtractSchema.model_validate(extract)
        bufr_to_df_args = extract_cfg.model_dump()
        self.bufr_to_df = BUFRToDataFrame(**bufr_to_df_args)
        self.num_processes = num_processes

    def _create_source(self, source_config: dict):
        if len(source_config) != 1:
            raise ValueError("Exactly one source must be specified")
        name, config = next(iter(source_config.items()))
        if name != "mars":
            raise ValueError(f"Invalid source name: {name}, must be 'mars'")
        return create_source(self.context, {name: config})

    def execute(self, dates) -> pd.DataFrame:
        # Return an empty frame that still carries the declared output columns
        # (known from config alone) so a data-less slot keeps a valid schema and
        # downstream filters / metadata collection do not crash on missing columns.
        empty = self.bufr_to_df.empty_frame()

        ekd_ds = self.source.execute(dates)
        if ekd_ds is None:
            LOG.debug(f"No BUFR data returned for date range {dates.start_range} - {dates.end_range}")
            return empty

        bufr_reader = BUFRReader(ekd_ds.path)
        df = bufr_to_dataframe_parallel(bufr_reader, self.bufr_to_df, self.num_processes)
        LOG.debug(f"Extracted {len(df)} BUFR rows for date range {dates.start_range} - {dates.end_range}")

        if len(df) == 0:
            return empty

        return df
