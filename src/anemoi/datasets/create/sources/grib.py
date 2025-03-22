# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import glob
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd
from earthkit.data import from_source
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.utils.patterns import Pattern

from .legacy import legacy_source

LOG = logging.getLogger(__name__)


def _load(context: Any, name: str, record: Dict[str, Any]) -> tuple:
    """Load data from a given source.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    name : str
        The name of the data source.
    record : dict of str to Any
        The record containing source information.

    Returns
    -------
    tuple
        A tuple containing the data as a numpy array and the UUID of the HGrid.
    """
    ds = None

    param = record["param"]

    if "path" in record:
        context.info(f"Using {name} from {record['path']} (param={param})")
        ds = from_source("file", record["path"])

    if "url" in record:
        context.info(f"Using {name} from {record['url']} (param={param})")
        ds = from_source("url", record["url"])

    ds = ds.sel(param=param)

    assert len(ds) == 1, f"{name} {param}, expected one field, got {len(ds)}"
    ds = ds[0]

    return ds.to_numpy(flatten=True), ds.metadata("uuidOfHGrid")


class Geography:
    """This class retrieves the latitudes and longitudes of unstructured grids,
    and checks if the fields are compatible with the grid.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    latitudes : dict of str to Any
        Latitude information.
    longitudes : dict of str to Any
        Longitude information.
    """

    def __init__(self, context: Any, latitudes: Dict[str, Any], longitudes: Dict[str, Any]) -> None:
        """Initialize the Geography class.

        Parameters
        ----------
        context : Any
            The context in which the function is executed.
        latitudes : dict of str to Any
            Latitude information.
        longitudes : dict of str to Any
            Longitude information.
        """
        latitudes, uuidOfHGrid_lat = _load(context, "latitudes", latitudes)
        longitudes, uuidOfHGrid_lon = _load(context, "longitudes", longitudes)

        assert (
            uuidOfHGrid_lat == uuidOfHGrid_lon
        ), f"uuidOfHGrid mismatch: lat={uuidOfHGrid_lat} != lon={uuidOfHGrid_lon}"

        context.info(f"Latitudes: {len(latitudes)}, Longitudes: {len(longitudes)}")
        assert len(latitudes) == len(longitudes)

        self.uuidOfHGrid = uuidOfHGrid_lat
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.first = True

    def check(self, field: Any) -> None:
        """Check if the field is compatible with the grid.

        Parameters
        ----------
        field : Any
            The field to check.
        """
        if self.first:
            # We only check the first field, for performance reasons
            assert (
                field.metadata("uuidOfHGrid") == self.uuidOfHGrid
            ), f"uuidOfHGrid mismatch: {field.metadata('uuidOfHGrid')} != {self.uuidOfHGrid}"
            self.first = False


class AddGrid:
    """An earth-kit.data.Field wrapper that adds grid information.

    Parameters
    ----------
    field : Any
        The field to wrap.
    geography : Geography
        The geography information.
    """

    def __init__(self, field: Any, geography: Geography) -> None:
        """Initialize the AddGrid class.

        Parameters
        ----------
        field : Any
            The field to wrap.
        geography : Geography
            The geography information.
        """
        self._field = field

        geography.check(field)

        self._latitudes = geography.latitudes
        self._longitudes = geography.longitudes

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped field.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self._field, name)

    def __repr__(self) -> str:
        """Get the string representation of the wrapped field.

        Returns
        -------
        str
            The string representation.
        """
        return repr(self._field)

    def grid_points(self) -> tuple:
        """Get the grid points (latitudes and longitudes).

        Returns
        -------
        tuple
            The latitudes and longitudes.
        """
        return self._latitudes, self._longitudes

    @property
    def resolution(self) -> str:
        """Get the resolution of the grid."""
        return "unknown"


def check(ds: Any, paths: List[str], **kwargs: Any) -> None:
    """Check if the dataset matches the expected number of fields.

    Parameters
    ----------
    ds : Any
        The dataset to check.
    paths : list of str
        List of paths to the GRIB files.
    **kwargs : Any
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If the number of fields does not match the expected count.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})")


def _expand(paths: List[str]) -> Any:
    """Expand the given paths using glob.

    Parameters
    ----------
    paths : list of str
        List of paths to expand.

    Returns
    -------
    Any
        The expanded paths.
    """
    for path in paths:
        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


@legacy_source(__file__)
def execute(
    context: Any,
    dates: List[Any],
    path: Union[str, List[str]],
    latitudes: Optional[Dict[str, Any]] = None,
    longitudes: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> ekd.FieldList:
    """Execute the function to load data from GRIB files.

    Args:
        context (Any): The context in which the function is executed.
        dates (List[Any]): List of dates.
        path (Union[str, List[str]]): Path or list of paths to the GRIB files.
        latitudes (Optional[Dict[str, Any]], optional): Latitude information. Defaults to None.
        longitudes (Optional[Dict[str, Any]], optional): Longitude information. Defaults to None.
        *args (Any): Additional arguments.
        **kwargs (Any): Additional keyword arguments.

    Returns
    -------
    Any
        The loaded dataset.
    """
    given_paths = path if isinstance(path, list) else [path]

    geography = None
    if latitudes is not None and longitudes is not None:
        geography = Geography(context, latitudes, longitudes)

    ds = from_source("empty")
    dates = [d.isoformat() for d in dates]

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(*args, date=dates, **kwargs)

        for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
            if name in kwargs:
                raise ValueError(f"MARS interpolation parameter '{name}' not supported")

        for path in _expand(paths):
            context.trace("📁", "PATH", path)
            s = from_source("file", path)
            s = s.sel(valid_datetime=dates, **kwargs)
            ds = ds + s

    if kwargs and not context.partial_ok:
        check(ds, given_paths, valid_datetime=dates, **kwargs)

    if geography is not None:
        ds = FieldArray([AddGrid(_, geography) for _ in ds])

    if len(ds) == 0:
        LOG.warning(f"No fields found for {dates} in {given_paths} (kwargs={kwargs})")

    return ds
