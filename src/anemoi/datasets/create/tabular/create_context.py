from earthkit.data.core.order import build_remapping

from anemoi.datasets.create.gridded.context import FieldContext

from ..create_context import CreateContextBase


class TabularCreateContext(CreateContextBase):

    def init(self):
        print(self.minimal_input)  # Ensure minimal_input is initialized

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
        return FieldContext(
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
            use_grib_paramid=self.main_config.build.use_grib_paramid,
        )
