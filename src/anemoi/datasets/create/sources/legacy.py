# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from ..source import Source
from . import source_registry


class LegacySource(Source):

    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs


class legacy_source:
    def __init__(self, name):
        self.name = name

    def __call__(self, execute):

        name = f"Legacy{self.name.title()}Source"

        def execute_wrapper(self, *args, **kwargs):
            """Wrapper method to call the execute function.

            Parameters
            ----------
            *args : tuple
            Positional arguments to pass to the execute function.
            **kwargs : dict
            Keyword arguments to pass to the execute function.
            """
            return execute(self.context, *args, **kwargs)

        klass = type(name, (LegacySource,), {})
        klass.execute = execute_wrapper

        source_registry.register(self.name)(klass)

        return execute
