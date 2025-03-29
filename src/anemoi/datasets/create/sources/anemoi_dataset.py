# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from .legacy import legacy_source


@legacy_source(__file__)
def execute(context, dates, params=None, **kwargs):
    import earthkit.data as ekd

    from anemoi.datasets import open_dataset

    ds = open_dataset(**kwargs)
    # dates_to_index = {date: i for i, date in enumerate(ds.dates)}

    indices = []
    for date in dates:
        idx = np.where(ds.dates == date)[0]
        if len(idx) == 0:
            continue
        indices.append((int(idx[0]), date))

    vars = ds.variables
    if params is None:
        params = vars

    if not isinstance(params, (list, tuple, set)):
        params = [params]

    params = set(params)
    results = []

    ensemble = ds.shape[2] > 1
    latitudes = ds.latitudes
    longitudes = ds.longitudes

    for idx, date in indices:

        metadata = dict(valid_datetime=date, latitudes=latitudes, longitudes=longitudes)

        for j, y in enumerate(ds[idx]):

            param = vars[j]
            if param not in params:
                continue

            # metadata['name'] = param
            # metadata['param_level'] = param
            metadata["param"] = param

            for k, e in enumerate(y):
                if ensemble:
                    metadata["number"] = k + 1

                metadata["values"] = e

                results.append(metadata.copy())

    print(results[0].keys())

    # "list-of-dicts" does support resolution
    results = ekd.from_source("list-of-dicts", results)

    # return new_fieldlist_from_list([new_field_from_latitudes_longitudes(x, latitudes, longitudes) for x in results])
    return results
