from typing import Any


class Filter:
    def __init__(self, context: Any, *args: Any, **kwargs: Any):
        self.context = context
