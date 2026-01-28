import datetime
from typing import Any

from earthkit.data import from_source
from anemoi.datasets.create.sources import source_registry

from .legacy import LegacySource

def _split_dates(dates: list[datetime.datetime]) -> list[list[str]]:
    """
    CDS wants lists of strings for `year`, `month`, `day`, `time`.
    """
    year = list({d.strftime("%Y") for d in dates})
    month = list({d.strftime("%m") for d in dates})
    day = list({d.strftime("%d") for d in dates})
    time = list({d.strftime("%H:%M") for d in dates})

    return year, month, day, time

@source_registry.register("cds")
class CdsSource(LegacySource):       

    @staticmethod
    def _execute(
        context: Any,
        dates: list[datetime.datetime],
        *requests: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Executes CDS requests based on the given context, dates, and other parameters.

        Parameters
        ----------
        context : Any
            The context for the requests.
        dates : List[datetime.datetime]
            The list of dates to be used in the requests.
        requests : Dict[str, Any]
            The input MARS requests to be executed.
        kwargs : Any
            Additional keyword arguments for the requests.

        Returns
        -------
        Any
            The resulting dataset.
        """

        if not requests:
            requests = [kwargs]
        
        year, month, day, time = _split_dates(dates)

        ds = from_source("empty")
        context.trace("✅", f"{[str(d) for d in dates]}")
        context.trace("✅", f"Will run {len(requests)} requests")
        for r in requests:
            dataset_id = r.pop("dataset_id", None)
            if not dataset_id:
                raise ValueError ("CDS requests require a `dataset_id`.")
            
            r.update({
                "year": year,
                "month": month,
                # "day": day,
                # "time": time,
                })
            
            ds = ds + ds = from_source("cds", dataset_id, r)
            # TODO: May need to clean up end dates/times, e.g. if last time is 12.00, may need to delete 18.00 from last date
        
        return ds
