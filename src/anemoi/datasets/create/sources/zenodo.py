# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict
from typing import List

import earthkit.data as ekd
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.sources.url import download_and_cache

from .legacy import legacy_source
from .patterns import iterate_patterns
from .xarray import load_one


@legacy_source(__file__)
def execute(context: Any, dates: Any, record_id: str, file_key: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """Executes the download and processing of files from Zenodo.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    dates : Any
        The dates for which the data is required.
    record_id : str
        The Zenodo record ID.
    file_key : str
        The key to identify the file.
    *args : Any
        Additional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        A list of fields loaded from the downloaded files.
    """
    import requests

    result: List[Any] = []

    URLPATTERN = "https://zenodo.org/api/records/{record_id}"
    url = URLPATTERN.format(record_id=record_id)
    r = requests.get(url)
    r.raise_for_status()
    record: Dict[str, Any] = r.json()

    urls: Dict[str, str] = {}
    for file in record["files"]:
        urls[file["key"]] = file["links"]["self"]

    for url, dates in iterate_patterns(file_key, dates, **kwargs):
        if url not in urls:
            continue

        path = download_and_cache(urls[url])
        result.append(load_one("?", context, dates, path, options={}, flavour=None, **kwargs))

    return MultiFieldList(result)
