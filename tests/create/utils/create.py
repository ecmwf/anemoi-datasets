# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import tempfile
from typing import Any

import yaml

from anemoi.datasets.create.tasks import run_task


class TestingContext:
    pass


def create_dataset(
    *,
    recipe: str | dict[str, Any],
    output: str | None,
    delta: list[str] | None = None,
    finalise_parts: int | None = None,
) -> str:
    """Create a dataset based on the provided configuration.

    Parameters
    ----------
    recipe : str | dict[str, Any]
        The configuration for the dataset. Can be a path to a YAML file or a dictionary.
    output : Optional[str]
        The output path for the dataset. If None, a temporary directory will be created.
    delta : Optional[List[str]], optional
        List of delta for secondary statistics, by default None.
    finalise_parts : Optional[int], optional
        If set, load each group in its own ``load --parts`` call, then split the finalise into
        ``finalise_parts`` parts (``finalise --prepare``, ``--load --parts i/finalise_parts``,
        ``--tidy``), by default None.

    Returns
    -------
    str
        The path to the created dataset.
    """

    if isinstance(recipe, dict):
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
        yaml.dump(recipe, temp_file)
        recipe = temp_file.name

    if output is None:
        output = tempfile.mkdtemp(suffix=".zarr")

    run_task("init", recipe=recipe, path=output, overwrite=True)

    if finalise_parts is None:
        run_task("load", path=output)
        run_task("finalise", path=output)
    else:
        from anemoi.datasets.create.dataset import Dataset

        # One loader per group, then the statistics computed in `finalise_parts` contiguous parts.
        groups = Dataset(output).total_todo()
        for i in range(groups):
            run_task("load", path=output, parts=f"{i + 1}/{groups}")
        run_task("finalise", path=output, finalise_stage="prepare")
        for i in range(finalise_parts):
            run_task("finalise", path=output, finalise_stage="load", parts=f"{i + 1}/{finalise_parts}")
        run_task("finalise", path=output, finalise_stage="tidy")

    run_task("patch", path=output)

    if delta is not None:
        run_task("init_additions", path=output, delta=delta)
        run_task("load_additions", path=output, delta=delta)
        run_task("finalise_additions", path=output, delta=delta)

    run_task("cleanup", path=output)

    if delta is not None:
        run_task("cleanup", path=output, delta=delta)

    run_task("verify", path=output)

    # New datasets are written in Zarr format 2, whatever the installed zarr-python version
    assert os.path.exists(os.path.join(output, ".zgroup")), f"{output} is not in Zarr format 2"

    return output
