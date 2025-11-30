from typing import Any

from anemoi.datasets.create.input.context import Context


class TabularContext(Context):
    def __init__(self):
        super().__init__()

    def empty_result(self) -> Any:
        import pandas as pd

        return pd.DataFrame()

    def source_argument(self, argument: Any) -> Any:
        return argument  # .dates

    def filter_argument(self, argument: Any) -> Any:
        return argument

    def create_result(self, argument, data):
        raise NotImplementedError()
