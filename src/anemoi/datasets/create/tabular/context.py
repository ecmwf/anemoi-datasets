from typing import Any

from anemoi.datasets.create.input.context import Context


class TabularContext(Context):

    def empty_result(self) -> Any:
        import pandas as pd

        return pd.DataFrame()

    def create_result(self, argument, data):
        from .result import TabularResult

        return TabularResult(self, argument, data)
