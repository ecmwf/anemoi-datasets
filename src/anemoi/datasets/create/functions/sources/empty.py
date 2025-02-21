# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd


def execute(context, dates, **kwargs):
    """Executes the loading of an empty data source.

    Parameters
    ----------
    context : object
        The context in which the function is executed.
    dates : list
        List of dates for which data is to be loaded.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    object
        Loaded empty data source.
    """
    return ekd.from_source("empty")
