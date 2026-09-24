# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import hashlib
import json
import os

from earthkit.data import from_source as original_from_source


class LoadSource:
    """Class to load data sources and handle mockup data."""

    def __init__(self, get_test_data_func, *, fetch_missing: bool = False) -> None:
        self._get_test_data = get_test_data_func

        #: Whether a request with no test data may be retrieved from the live archive.
        #: Off by default: a missing fixture usually means the recipe changed what it
        #: asks for, and fetching would spend a real retrieval on a change nobody has
        #: looked at. Recording the change is what turns this on.
        self._fetch_missing = fetch_missing

        #: Every MARS request this instance was asked for, in order, as
        #: ``{"md5": ..., "request": [args, kwargs]}``.  The fixtures are keyed by
        #: that md5, so this is the record of what the recipe asked the archive
        #: for -- see ``test_create._check_mars_requests``.
        self.requests: list[dict] = []


    def filename(self, args: tuple, kwargs: dict) -> str:
        """Generate a filename based on the arguments and keyword arguments.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        str
            The generated filename.
        """
        string = json.dumps([args, kwargs], sort_keys=True, default=str)
        h = hashlib.md5(string.encode("utf8")).hexdigest()
        self.requests.append({"md5": h, "request": json.loads(string)})
        return h + ".grib"

    def get_data(self, args: tuple, kwargs: dict, path: str) -> None:
        """Retrieve data and save it to the specified path.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.
        path : str
            The path to save the data.

        Raises
        ------
        ValueError
            If the test data is missing.
        """
        upload_path = os.path.realpath(path + ".to_upload")
        ds = original_from_source("mars", *args, **kwargs)
        ds.save(upload_path)
        print(f"Mockup: Saving to {upload_path} for {args}, {kwargs}")
        print()
        print("⚠️ To upload the test data, run this:")
        path = os.path.relpath(upload_path, os.getcwd())
        name = os.path.basename(upload_path).replace(".to_upload", "")
        print(f"scp {path} data@anemoi.ecmwf.int:public/anemoi-datasets/create/mock-mars/{name}")
        print()
        raise ValueError("Test data is missing")

    def mars(self, args: tuple, kwargs: dict) -> object:
        """Load data from the MARS archive.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        object
            The loaded data source.
        """

        name = self.filename(args, kwargs)

        try:
            return original_from_source("file", self._get_test_data(f"anemoi-datasets/create/mock-mars/{name}"))
        except RuntimeError:
            raise  # If offline
        except Exception:
            # A miss almost always means the recipe now asks for different fields --
            # which, while refactoring the covering or the search, is the defect we are
            # looking for. Refusing to fetch keeps a changed request from costing a real
            # retrieval and leaving a file inviting you to bless it unseen. Recording
            # the change is what authorises the download.
            if not self._fetch_missing:
                raise AssertionError(
                    "no test data for this MARS request, and fetching is off:\n"
                    f"  {json.dumps([args, kwargs], sort_keys=True, default=str)}\n"
                    "If the recipe now retrieves different fields, that is the change to "
                    "look at; the recorded requests are in tests/create/requests/.\n"
                    "To record it and fetch the data, re-run with ANEMOI_UPDATE_MARS_REQUESTS=1"
                ) from None
            self.get_data(args, kwargs, name)

    def __call__(self, name: str, *args: tuple, **kwargs: dict) -> object:
        """Call the appropriate method based on the data source name.

        Parameters
        ----------
        name : str
            The name of the data source.
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        object
            The loaded data source.
        """
        if name == "mars":
            return self.mars(args, kwargs)

        return original_from_source(name, *args, **kwargs)
