# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

# from .dataset import FullIndex
# from .dataset import Shape
# from .dataset import TupleIndex
from anemoi.datasets.usage.misc import _open_dataset
from anemoi.datasets.usage.misc import _save_dataset
from anemoi.datasets.usage.misc import add_dataset_path
from anemoi.datasets.usage.misc import add_named_dataset

if TYPE_CHECKING:
    from anemoi.datasets.usage.dataset import Dataset

LOG = logging.getLogger(__name__)

__all__ = [
    "MissingDateError",
    "add_dataset_path",
    "add_named_dataset",
    "open_dataset",
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
        The opened dataset. When ``sharding=N`` is given (tabular datasets
        only), a ``ShardedTabular`` collection of ``N`` shard datasets is
        returned instead: each shard behaves like the dataset but every window
        yields only ``1/N`` of the rows, with boundaries adjusted accordingly.
        With ``sharding=N, shard=i`` a single shard ``i`` is returned.
    """

    trace = int(os.environ.get("ANEMOI_DATASETS_TRACE", 0))

    # That will get rid of OmegaConf objects

    args, kwargs = _convert(args), _convert(kwargs)

    # `sharding`/`shard` are handled here, after the dataset is fully built,
    # because they turn a single dataset into a collection of shards (or pick
    # one). They are not `_subset` options.
    sharding = kwargs.pop("sharding", None)
    shard = kwargs.pop("shard", None)

    if shard is not None and sharding is None:
        raise ValueError("`shard` requires `sharding` to be set")

    ds = _open_dataset(*args, **kwargs)
    ds = ds.mutate()
    ds.arguments = {"args": args, "kwargs": kwargs}
    ds._check()

    if sharding is not None:
        from anemoi.datasets.usage.tabular.sharding import shard_dataset
        from anemoi.datasets.usage.tabular.sharding import single_shard

        wrap = None
        if trace:
            from anemoi.datasets.testing import Trace

            wrap = Trace

        if shard is not None:
            one = single_shard(ds, sharding, shard, wrap=wrap)
            one.arguments = {"args": args, "kwargs": {**kwargs, "sharding": sharding, "shard": shard}}
            one._check()
            return one

        shards = shard_dataset(ds, sharding, wrap=wrap)
        for s in shards:
            s.arguments = {"args": args, "kwargs": {**kwargs, "sharding": sharding}}
            s._check()
        return shards

    if trace:
        from anemoi.datasets.testing import Trace

        ds = Trace(ds)

    return ds


def save_dataset(dataset: "Dataset", zarr_path: str, n_workers: int = 1) -> None:
    """Open a dataset and save it to disk.

    Parameters
    ----------
    dataset : Dataset
        anemoi-dataset opened from python to save to Zarr store.
    zarr_path : str
        Path to store the obtained anemoi dataset to disk.
    n_workers : int
        Number of workers to use for parallel processing. If none, sequential processing will be performed.
    """
    _save_dataset(dataset, zarr_path, n_workers)


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
    names: set[str] = set()
    ds.get_dataset_names(names)
    return sorted(names)
