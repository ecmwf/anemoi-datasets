# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

import earthkit.data as ekd

from . import source_registry
from .legacy import LegacySource
from .xarray import load_many


@source_registry.register("opendap")
class OpenDAPSource(LegacySource):

    @staticmethod
    def _execute(context: dict[str, Any], dates: list[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
        """Execute the data loading process from an OpenDAP source.

        Parameters
        ----------
        context : dict
            The context in which the function is executed.
        dates : list
            List of dates for which data is to be loaded.
        url : str
            The URL of the OpenDAP source.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        xarray.Dataset
            The loaded dataset.
        """
        return load_many("🌐", context, dates, url, *args, **kwargs)
