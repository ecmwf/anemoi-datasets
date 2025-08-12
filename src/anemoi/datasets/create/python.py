# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import re
import sys
from collections import defaultdict

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


def _sanitize_name(name):
    name = name.replace("-", "_")
    if name in RESERVED_KEYWORDS:
        name = f"{name}_"
    return name


class PythonCode:

    def __init__(self, top):
        print(f"Creating {self.__class__.__name__} from {top.__class__.__name__}", file=sys.stderr)
        self.top = top
        self.top.register(self)
        self.key = str(id(self))

    def call(self, name, argument):
        return PythonCall(self.top, name, argument)

    def sum(self, actions):
        return PythonChain(self.top, "+", actions)

    def pipe(self, actions):
        return PythonChain(self.top, "|", actions)

    def concat(self, argument):
        return PythonConcat(self.top, argument)

    def source_code(self):
        return self.top.source_code(self)

    def combine(self, nodes):
        return None


class Argument:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{_sanitize_name(self.name)}"


class PythonSource(PythonCode):

    def __init__(self):
        self._prelude = []
        self.nodes = []
        self._count = defaultdict(int)
        super().__init__(top=self)

    def register(self, child):
        if child is not self:
            self.nodes.append(child)

    def prelude(self):
        return "\n".join(self._prelude)

    def source_code(self, first):

        which = self.nodes.index(first)

        more = True
        while more:
            more = False

            by_class = defaultdict(list)
            for node in self.nodes:
                by_class[(node.__class__, node.key)].append(node)

            for (cls, key), nodes in by_class.items():
                if len(nodes) > 1:
                    print(f"Found multiple nodes of type {cls.__name__}/{key}, merging them", file=sys.stderr)
                    print(f"Nodes: {len(nodes)}", file=sys.stderr)
                    changes = nodes[0].combine(nodes)
                    if changes:
                        self.replace_nodes(changes)
                        more = True

        first = self.nodes[which]

        return "\n\n".join(
            [
                "# Generated Python code for Anemoi dataset creation",
                "from anemoi.datasets.recipe import Recipe",
                "r = Recipe()",
                *self._prelude,
                f"r.input = {repr(first)}",
                "r.dump()",
            ]
        )

    def function(self, key, value, node):

        n = self._count[node.name]
        self._count[node.name] += 1

        name = f"{node.name}_{n}"
        name = _sanitize_name(name)
        key = _sanitize_name(key)

        class Function:
            def __init__(self, name, key, value, node):
                self.name = name
                self.key = key
                self.value = value
                self.node = node

            def __repr__(self):
                return f"{self.name}"

        self._prelude.append(f"def {name}({key}):")
        self._prelude.append(f"   return {node}")
        return Function(name, key, value, node)

    def replace_nodes(self, changes):

        for old, new in changes:
            assert old in self.nodes, f"Node {old} not found in {self.nodes}"
            for i, node in enumerate(self.nodes):

                if node is old:
                    self.nodes[i] = new
                else:
                    node.replace_node(old, new)


class PythonConcat(PythonCode):
    def __init__(self, top, argument):
        super().__init__(top=top)
        self.argument = argument
        for k, v in self.argument.items():
            assert isinstance(v, PythonCode), f"Value must be a PythonCode instance {v}"

    def __repr__(self):
        return f"r.concat({self.argument})"

    def replace_node(self, old, new):
        for k, v in list(self.argument.items()):
            if v is old:
                self.argument[k] = new
            else:
                v.replace_node(old, new)


class PythonCall(PythonCode):
    def __init__(self, top, name, argument, parameters=None):
        super().__init__(top=top)
        self.name = name
        self.argument = argument
        self.key = name
        self.parameters = parameters

    def __repr__(self):
        name = self.name.replace("-", "_")
        config = dict(**self.argument)

        params = []

        for k, v in config.items():
            if isinstance(k, str):

                if k in RESERVED_KEYWORDS:
                    k = f"{k}_"

                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", k):
                    return f"r.{name}({config})"
            params.append(f"{k}={repr(v)}")

        params = ",".join(params)
        return f"r.{name}({params})"

    def replace_node(self, old, new):
        pass

    def combine(self, nodes):

        x = defaultdict(list)
        for node in nodes:
            argument = node.argument
            for k, v in argument.items():
                rest = {k2: v2 for k2, v2 in sorted(argument.items()) if k2 != k}
                x[str(rest)].append((k, v, node))

        for i in sorted(x.values(), key=len, reverse=True):
            key, value, node = i[0]
            if len(i) < 2:
                return

            rest = {k: v for k, v in node.argument.items() if k != key}
            rest[key] = Argument(key)
            call = PythonCall(self.top, self.name, rest)

            func = self.top.function(key, value, node=call)
            changes = []
            for key, value, node in i:

                new = PythonFunction(
                    top=self.top,
                    func=func,
                    argument={key: value},
                )

                changes.append((node, new))

            return changes


class PythonChain(PythonCode):
    def __init__(self, top, op, actions):
        super().__init__(top=top)
        self.op = op
        self.actions = list(actions)
        self.key = op

    def __repr__(self):
        return "(" + self.op.join(repr(x) for x in self.actions) + ")"

    def replace_node(self, old, new):

        for i, node in enumerate(self.actions):

            if node is old:
                self.actions[i] = new
            else:
                node.replace_node(old, new)


class PythonFunction(PythonCode):
    def __init__(self, top, func, argument):
        super().__init__(top=top)
        self.func = func
        self.argument = argument
        self.key = func

    def __repr__(self):
        return f"{self.func}({', '.join(f'{_sanitize_name(k)}={repr(v)}' for k, v in self.argument.items())})"

    def replace_node(self, old, new):
        pass
