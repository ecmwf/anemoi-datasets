# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


class TaskDispatcher:
    """A class to create and run dataset creation tasks."""

    def __init__(self, creator):
        self.creator = creator

    def task_init(self):
        return self.creator.task_init()

    def task_load(self):
        return self.creator.task_load()

    def task_size(self):
        return self.creator.task_size()

    def task_patch(self):
        return self.creator.task_patch()

    def task_statistics(self):
        return self.creator.task_statistics()

    def task_finalise(self):
        # The finalise step can be split across independent processes so the zarr can be
        # populated incrementally and in random order. `finalise --prepare/--load/--tidy`
        # run one stage each; no stage flag runs the whole thing in one process. Cleanup
        # (which removes the work_dir tmp files) only runs on the final stage.
        match self.creator.finalise_stage:
            case None:
                self.creator.task_finalise()
                self.creator.task_statistics()
                self.creator.task_size()
                self.creator.task_cleanup()
            case "prepare":
                self.creator.task_finalise_prepare()
            case "rows_per_chunk":
                self.creator.task_finalise_rows_per_chunk()
            case "load":
                self.creator.task_finalise_load()
            case "tidy":
                self.creator.task_finalise_tidy()
                self.creator.task_statistics()
                self.creator.task_size()
                self.creator.task_cleanup()
            case other:
                raise ValueError(f"Unknown finalise stage: {other!r}")

    def task_cleanup(self):
        self.creator.task_cleanup()

    def task_verify(self):
        self.creator.task_verify()

    def task_init_additions(self):
        self.creator.task_init_additions()

    def task_load_additions(self):
        self.creator.task_load_additions()

    def task_finalise_additions(self):
        self.creator.task_finalise_additions()
        self.creator.task_size()

    def task_additions(self):
        self.creator.task_init_additions()
        self.creator.task_load_additions()
        self.creator.task_finalise_additions()
        self.creator.task_cleanup()


def run_task(name: str, recipe=None, **kwargs):

    from anemoi.datasets.create.creator import Creator

    LOG.info(f"Running task: {name}, recipe: {recipe}, kwargs: {kwargs}")

    creator = Creator.from_recipe(recipe, **kwargs)
    dispatch = TaskDispatcher(creator)
    return getattr(dispatch, f"task_{name}")()
