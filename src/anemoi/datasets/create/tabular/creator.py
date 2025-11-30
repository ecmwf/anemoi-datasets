import rich

from ..creator import Creator
from .context import TabularContext


class TabularCreator(Creator):

    def init(self):
        rich.print(self.minimal_input)

    def load(self):
        raise NotImplementedError("Load method not implemented yet.")

    def statistics(self):
        raise NotImplementedError("Statistics method not implemented yet.")

    def size(self):
        raise NotImplementedError("Size method not implemented yet.")

    def cleanup(self):
        raise NotImplementedError("Cleanup method not implemented yet.")

    def init_additions(self):
        raise NotImplementedError("Init additions method not implemented yet.")

    def load_additions(self):
        raise NotImplementedError("Load additions method not implemented yet.")

    def finalise_additions(self):
        raise NotImplementedError("Finalise additions method not implemented yet.")

    def patch(self):
        raise NotImplementedError("Patch method not implemented yet.")

    def verify(self):
        raise NotImplementedError("Verify method not implemented yet.")

    ######################################################

    def context(self):
        return TabularContext()
