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

    def __init__(self, get_test_data_func) -> None:
        self._get_test_data = get_test_data_func

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
        exit(1)
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
