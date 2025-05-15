# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Set

# from .dataset import FullIndex
# from .dataset import Shape
# from .dataset import TupleIndex
from .misc import _open_dataset
from .misc import _save_dataset
from .misc import add_dataset_path
from .misc import add_named_dataset

if TYPE_CHECKING:
    from .dataset import Dataset

LOG = logging.getLogger(__name__)

__all__ = [
    "open_dataset",
    "MissingDateError",
    "add_dataset_path",
    "add_named_dataset",
]


class MissingDateError(Exception):
    pass


def _convert(x: Any) -> Any:
    """Convert OmegaConf objects to standard Python containers.

    Parameters
    ----------
    x : Any
        The object to convert.

    Returns
    -------
    Any
        The converted object.
    """
    if isinstance(x, list):
        return [_convert(a) for a in x]

    if isinstance(x, tuple):
        return tuple(_convert(a) for a in x)

    if isinstance(x, dict):
        return {k: _convert(v) for k, v in x.items()}

    if x.__class__.__name__ in ("DictConfig", "ListConfig"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(x, resolve=True)

    return x


def open_dataset(*args: Any, **kwargs: Any) -> "Dataset":
    """Open a dataset.

    Parameters
    ----------
    *args : Any
        Positional arguments.
    **kwargs : Any
        Keyword arguments.

    Returns
    -------
    Dataset
        The opened dataset.
    """
    # That will get rid of OmegaConf objects

    args, kwargs = _convert(args), _convert(kwargs)

    ds = _open_dataset(*args, **kwargs)
    ds = ds.mutate()
    ds.arguments = {"args": args, "kwargs": kwargs}
    ds._check()
    return ds


def save_dataset(recipe: dict, zarr_path: str, n_workers: int = 1) -> None:
    """Open a dataset and save it to disk.

    Parameters
    ----------
    recipe : dict
        Recipe used with open_dataset (not a dataset creation recipe).
    zarr_path : str
        Path to store the obtained anemoi dataset to disk.
    n_workers : int
        Number of workers to use for parallel processing. If none, sequential processing will be performed.
    """
    _save_dataset(recipe, zarr_path, n_workers)


def list_dataset_names(*args: Any, **kwargs: Any) -> list[str]:
    """List the names of datasets.

    Parameters
    ----------
    *args : Any
        Positional arguments.
    **kwargs : Any
        Keyword arguments.

    Returns
    -------
    list of str
        The list of dataset names.
    """
    ds = _open_dataset(*args, **kwargs)
    names: Set[str] = set()
    ds.get_dataset_names(names)
    return sorted(names)
