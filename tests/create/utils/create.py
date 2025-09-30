# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import tempfile
from typing import Any

import yaml

from anemoi.datasets.create.fields.tasks import task_factory


class TestingContext:
    pass


def create_dataset(
    *,
    config: str | dict[str, Any],
    output: str | None,
    delta: list[str] | None = None,
    is_test: bool = False,
) -> str:
    """Create a dataset based on the provided configuration.

    Parameters
    ----------
    config : Union[str, Dict[str, Any]]
        The configuration for the dataset. Can be a path to a YAML file or a dictionary.
    output : Optional[str]
        The output path for the dataset. If None, a temporary directory will be created.
    delta : Optional[List[str]], optional
        List of delta for secondary statistics, by default None.
    is_test : bool, optional
        Flag indicating if the dataset creation is for testing purposes, by default False.

    Returns
    -------
    str
        The path to the created dataset.
    """
    if isinstance(config, dict):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
        yaml.dump(config, temp_file)
        config = temp_file.name

    if output is None:
        output = tempfile.mkdtemp(suffix=".zarr")

    task_factory("init", config=config, path=output, overwrite=True, test=is_test).run()
    task_factory("load", path=output).run()
    task_factory("finalise", path=output).run()
    task_factory("patch", path=output).run()

    if delta is not None:
        task_factory("init_additions", path=output, delta=delta).run()
        task_factory("load_additions", path=output, delta=delta).run()
        task_factory("finalise_additions", path=output, delta=delta).run()

    task_factory("cleanup", path=output).run()

    if delta is not None:
        task_factory("cleanup", path=output, delta=delta).run()

    task_factory("verify", path=output).run()

    return output
