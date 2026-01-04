# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
        self.creator.task_finalise()
        self.creator.task_statistics()
        self.creator.task_size()
        self.creator.task_cleanup()

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

    print(f"Running task: {name}, recipe: {recipe}, kwargs: {kwargs}")

    creator = Creator.from_recipe(recipe, **kwargs)
    dispatch = TaskDispatcher(creator)
    return getattr(dispatch, f"task_{name}")()
