# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import json
import re

from anemoi.utils.dates import frequency_to_string

# input.python_prelude(code)
# code1 = "\n".join(prelude)
# rich.print(f"Input prelude:\n{code1}")
# code2 = input.to_python()

# code = f"from anemoi.datasets.recipe import Recipe\nr = Recipe()\n{code1}\nr.input = {code2}\n\nr.dump()"

# code = re.sub(r"[\"\']?\${data_sources\.(\w+)}[\"\']?", r"\1", code)

# try:
#     import black

#     return black.format_str(code, mode=black.Mode())
# except ImportError:
#     LOG.warning("Black not installed, skipping formatting")
#     return code
RESERVED_KEYWORDS = (
    "and",
    "or",
    "not",
    "is",
    "in",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "return",
    "class",
    "def",
    "with",
    "as",
    "import",
    "from",
    "try",
    "except",
    "finally",
    "raise",
    "assert",
    "break",
    "continue",
    "pass",
)


class PythonCode:

    def call(self, name, argument):
        return PythonCall(name, argument)

    def sum(self, actions):
        return PythonChain("+", actions)

    def pipe(self, actions):
        return PythonChain("|", actions)

    def concat(self, argument):
        return PythonConcat(argument)


class PythonConcat(PythonCode):
    def __init__(self, argument):
        self.argument = argument

    def __repr__(self):
        return str(self.argument)


class PythonCall(PythonCode):
    def __init__(self, name, argument):
        self.name = name
        self.argument = argument

    def __repr__(self):
        name = self.name.replace("-", "_")
        config = self.argument

        # def convert(obj):
        #     if isinstance(obj, datetime.datetime):
        #         return obj.isoformat()
        #     if isinstance(obj, datetime.date):
        #         return obj.isoformat()
        #     if isinstance(obj, datetime.timedelta):
        #         return frequency_to_string(obj)
        #     if isinstance(obj, PythonCode):
        #          return obj
        #     raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # config = json.loads(json.dumps(config, default=convert))

        params = []
        for k, v in config.items():
            if isinstance(k, str):
                if k in RESERVED_KEYWORDS or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", k) is None:
                    return f"r.{name}({config})"
            params.append(f"{k}={repr(v)}")

        # for k, v in extra.items():
        #     params.append(f"{k}={v}")

        params = ",".join(params)
        return f"r.{name}({params})"
        # return f"{name}({config})"
        return f"{self.name}({self.argument})"


class PythonChain(PythonCode):
    def __init__(self, op, actions):
        self.op = op
        self.actions = actions

    def __repr__(self):
        return "(" + self.op.join(repr(x) for x in self.actions) + ")"


def _python(name, config, **extra) -> str:
    """Convert the action to Python code.

    Parameters
    ----------
    name : str
        The name of the action.
    config : dict
        The configuration for the action.
    extra : Any
        Additional keyword arguments.

    Returns
    -------
    str
        The Python code representation of the action.
    """

    name = name.replace("-", "_")

    def convert(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, datetime.timedelta):
            return frequency_to_string(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    config = json.loads(json.dumps(config, default=convert))

    params = []
    for k, v in config.items():
        if k in RESERVED_KEYWORDS or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", k) is None:
            return f"r.{name}({config})"
        params.append(f"{k}={repr(v)}")

    for k, v in extra.items():
        params.append(f"{k}={v}")

    params = ",".join(params)
    return f"r.{name}({params})"
    # return f"{name}({config})"
