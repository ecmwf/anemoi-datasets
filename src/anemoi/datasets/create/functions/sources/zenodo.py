# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.sources.url import download_and_cache

from . import iterate_patterns
from .xarray import load_one


def execute(context, dates, record_id, file_key, *args, **kwargs):
    import requests

    result = []

    URLPATTERN = "https://zenodo.org/api/records/{record_id}"
    url = URLPATTERN.format(record_id=record_id)
    r = requests.get(url)
    r.raise_for_status()
    record = r.json()

    urls = {}
    for file in record["files"]:
        urls[file["key"]] = file["links"]["self"]

    for url, dates in iterate_patterns(file_key, dates, **kwargs):
        if url not in urls:
            continue

        path = download_and_cache(urls[url])
        result.append(load_one("?", context, dates, path, options={}, flavour=None, **kwargs))

    return MultiFieldList(result)
