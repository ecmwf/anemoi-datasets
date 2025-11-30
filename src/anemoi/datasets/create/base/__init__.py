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

    def init(self):
        return self.creator.init()

    def load(self):
        return self.creator.load()

    def size(self):
        return self.creator.size()

    def patch(self):
        return self.creator.patch()

    def statistics(self):
        return self.creator.statistics()

    def finalise(self):
        self.creator.statistics()
        self.creator.size()
        self.creator.cleanup()

    def cleanup(self):
        self.creator.cleanup()

    def verify(self):
        self.creator.verify()

    def init_additions(self):
        self.creator.init_additions()

    def load_additions(self):
        self.creator.load_additions()

    def finalise_additions(self):
        self.creator.finalise_additions()
        self.creator.size()

    def additions(self):
        self.creator.init_additions()
        self.creator.load_additions()
        self.creator.finalise_additions()
        self.creator.cleanup()


def run_task(name: str, config=None, observations: bool = False, trace: str | None = None, **kwargs):

    if observations:
        from anemoi.datasets.create.tabular.creator import TabularCreator

        creator = TabularCreator(config=config, **kwargs)
    else:
        from anemoi.datasets.create.gridded.creator import GriddedCreator

        creator = GriddedCreator(config=config, **kwargs)

    dispatch = TaskDispatcher(creator)
    return getattr(dispatch, name)()
