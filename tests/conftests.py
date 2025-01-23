import os
import sys

import pytest


# enable_stop_on_exceptions if the debugger is running during a test
def is_debugging():
    return "debugpy" in sys.modules or os.getenv("_PYTEST_RAISE", "0") != "0"


if is_debugging():

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
