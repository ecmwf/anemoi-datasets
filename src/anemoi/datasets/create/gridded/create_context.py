from ..create_context import CreateContextBase


class GriddedCreateContext(CreateContextBase):

    def init(self):
        from .init import Init

        return Init(self.path, self.config, **self.kwargs).run()

    def load(self):
        from .load import Load

        return Load(self.path, self.config, **self.kwargs).run()

    def statistics(self):
        from .statistics import Statistics

        return Statistics(self.path, self.config, **self.kwargs).run()

    def size(self):
        from .size import Size

        return Size(self.path, self.config, **self.kwargs).run()

    def cleanup(self):
        from .cleanup import Cleanup

        return Cleanup(self.path, self.config, **self.kwargs).run()

    def init_additions(self):
        from .additions import InitAdditions

        return InitAdditions(self.path, self.config, **self.kwargs).run()

    def load_additions(self):
        from .additions import LoadAdditions

        return LoadAdditions(self.path, self.config, **self.kwargs).run()

    def finalise_additions(self):
        from .additions import FinaliseAdditions

        return FinaliseAdditions(self.path, self.config, **self.kwargs).run()

    def patch(self):
        from .patch import Patch

        return Patch(self.path, self.config, **self.kwargs).run()

    def verify(self):
        from .verify import Verify

        return Verify(self.path, self.config, **self.kwargs).run()
