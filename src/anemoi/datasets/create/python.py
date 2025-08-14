# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import re
from collections import defaultdict
from functools import cached_property

LOG = logging.getLogger(__name__)

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


def _un_dotdict(x):
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    return x


class PythonCode:

    def __init__(self, top):
        self.top = top
        self.top.register(self)
        self.key = str(id(self))

    def call(self, name, argument):
        return PythonCall(self.top, name, argument)

    def sum(self, actions):
        return PythonChain(self.top, "join", "+", actions)

    def pipe(self, actions):
        return PythonChain(self.top, "pipe", "|", actions)

    def concat(self, argument):
        return PythonConcat(self.top, argument)

    def source_code(self):
        return self.top.source_code(self)

    def combine(self, nodes):
        return None

    def recipe(self, input, data_sources):
        return PythonRecipe(self.top, input, data_sources)

    def prelude(self):
        return None

    def sources(self, sources):
        return PythonSources(self.top, sources)

    def update_anchor(self):
        pass


class Variable(PythonCode):
    def __init__(self, name, node):
        super().__init__(top=node.top)
        self.name = name
        self.node = node

    def __repr__(self):
        return ""

    def replace_node(self, old, new):
        pass

    def prelude(self):
        return [f"{self.name} = {repr(self.node)}", ""]


class InLine(PythonCode):
    def __init__(self, node):
        super().__init__(top=node.top)
        self.node = node

    @cached_property
    def name(self):
        n = self.top.counter["_anchor"]
        self.top.counter["_anchor"] += 1
        return f"_a{n}"

    def __repr__(self):
        return f"({self.name} := {repr(self.node)})"

    def replace_node(self, old, new):
        pass


class PythonRecipe(PythonCode):
    def __init__(self, top, input, data_sources):
        super().__init__(top)
        self.input = input
        self.data_sources = data_sources

    def apply_references(self, *path):
        self.data_sources.apply_references(*path, "data_sources")
        self.input.apply_references(*path, "input")

    def replace_node(self, old, new):
        if self.input is old:
            self.input = new
            return

        if self.data_sources is old:
            self.data_sources = new
            return

        self.input.replace_node(old, new)
        self.data_sources.replace_node(old, new)

    def __repr__(self):
        return repr(self.input)

    def prelude(self):
        return self.data_sources.prelude()


class Argument(PythonCode):

    def __init__(self, top, name):
        super().__init__(top=top)
        self.name = _sanitize_name(name)

    def __repr__(self):
        return self.name

    def replace_node(self, old, new):
        pass


class Anchor(PythonCode):

    def __init__(self, identifier):
        super().__init__(top=identifier.node.top)
        self.identifier = identifier

    @property
    def name(self):
        return self.identifier.name

    def __repr__(self):
        # assert False
        return repr(self.identifier)

    def replace_node(self, old, new):
        pass


class Reference(PythonCode):

    def __init__(self, top, path):
        super().__init__(top)
        self.path = tuple(path)
        self.anchor = None

    def update_anchor(self):

        node = self.top.by_reference.get(self.path, None)
        if node is None:
            LOG.warning(f"Reference {self.path} not found")
            for p in sorted(self.top.by_reference):
                LOG.warning(f"  - {p}")
        else:
            self.anchor = Anchor(node)
            self.top.replace_nodes([(node.node, self.anchor)])

    def __repr__(self):
        if self.anchor is not None:
            return self.anchor.name

        return f"'${{{'.'.join(self.path)}}}'"

    def replace_node(self, old, new):
        pass


class Function(PythonCode):
    def __init__(self, name, node, counter):
        super().__init__(top=node.top)
        self._name = name
        self.node = node
        self.used = False
        self.counter = counter

    def __repr__(self):
        return self.name

    def prelude(self):
        if self.used:
            return None

        self.used = True

        node_prelude = self.node.prelude()

        arguments = self.node.free_arguments()

        return [
            *(node_prelude if node_prelude else []),
            f"def {self.name}({','.join(repr(p) for p in arguments)}):",
            f"   return {self.node}",
        ]

    def free_arguments(self):
        return self.node.free_arguments()

    @cached_property
    def name(self):
        n = self.counter[self._name]
        self.counter[self._name] += 1
        if n == 0:
            return _sanitize_name(self._name)
        return _sanitize_name(f"{self._name}_{n}")

    def replace_node(self, old, new):
        if self.node is old:
            self.node = new


class PythonSources(PythonCode):
    def __init__(self, top, sources):
        super().__init__(top)
        self.sources = sources

    def __repr__(self):
        return ""

    def prelude(self):
        pass

    def replace_node(self, old, new):
        for k, v in list(self.sources.items()):
            if v is old:
                self.sources[k] = new
            else:
                v.replace_node(old, new)

    def apply_references(self, *path):
        for k, v in self.sources.items():
            self.top.by_reference[path + (k,)] = Variable(k, v)


class PythonConcat(PythonCode):
    def __init__(self, top, argument):
        super().__init__(top=top)
        self.argument = _un_dotdict(argument)

    def __repr__(self):
        return f"r.concat({self.argument})"

    def replace_node(self, old, new):
        for k, v in list(self.argument.items()):
            if v is old:
                self.argument[k] = new
            else:
                v.replace_node(old, new)

    def apply_references(self, *path):
        assert "concat" not in path, path
        self.top.by_reference[path + ("concat",)] = InLine(self)
        for i, node in enumerate(self.argument.values()):
            node.apply_references(*path, "concat", str(i))


class PythonChain(PythonCode):
    def __init__(self, top, kind, op, actions):
        super().__init__(top=top)
        self.op = op
        self.kind = kind
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

    def apply_references(self, *path):
        self.top.by_reference[path + (self.kind,)] = InLine(self)
        for i, node in enumerate(self.actions):
            node.apply_references(*path, self.kind, str(i))


class PythonCall(PythonCode):
    def __init__(self, top, name, argument):
        super().__init__(top=top)
        self.name = name
        self.argument = _un_dotdict(argument)
        self.key = name

    def free_arguments(self):
        result = []
        for k, v in self.argument.items():
            if isinstance(v, Argument):
                result.append(v)
        return result

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

        if params:
            params.append("")  # For a trailing comma

        params = ",".join(params)
        return f"r.{name}({params})"

    def replace_node(self, old, new):
        pass

    def combine(self, nodes):

        # Exact similarity

        changes = self._combine0(nodes)
        if changes:
            return changes

        # On key difference
        changes = self._combine1(nodes)
        if changes:
            return changes

    def _combine0(self, nodes):

        x = defaultdict(list)
        for node in nodes:
            key = {k2: v2 for k2, v2 in sorted(node.argument.items())}
            x[str(key)].append(node)

        for i in sorted(x.values(), key=len, reverse=True):
            node = i[0]
            if len(i) < 2:
                return

            call = PythonCall(self.top, self.name, node.argument)

            func = self.top.function(call)
            changes = []
            for node in i:

                new = PythonFunction(top=self.top, func=func)

                changes.append((node, new))

            return changes

    def _combine1(self, nodes):

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
            rest[key] = Argument(self.top, key)
            call = PythonCall(self.top, self.name, rest)

            func = self.top.function(call)
            changes = []
            for key, value, node in i:

                new = PythonFunction(
                    top=self.top,
                    func=func,
                    **{key: value},
                )

                changes.append((node, new))

            return changes

    def apply_references(self, *path):
        self.top.by_reference[path + (self.name,)] = InLine(self)

        for k, v in self.argument.items():
            if isinstance(v, str) and (m := re.match(r"^\${(\w+(?:\.\w+)+)}$", v)):
                path = m.group(1).split(".")
                self.argument[k] = Reference(self.top, path)


class PythonFunction(PythonCode):
    def __init__(self, top, func, **kwargs):
        super().__init__(top=top)
        self.func = func
        self.kwargs = kwargs

    def __repr__(self):

        params = []
        for a in self.func.free_arguments():
            name = _sanitize_name(a.name)
            if a.name in self.kwargs:
                v = self.kwargs[a.name]
                params.append(f"{name}={repr(v)}")
            else:
                params.append(f"{name}={name}")

        return f"{self.func}({', '.join(params)})"

    def replace_node(self, old, new):
        self.func.replace_node(old, new)

    def prelude(self):
        return self.func.prelude()

    def free_arguments(self):
        return [a for a in self.func.free_arguments() if a.name not in self.kwargs]

    def apply_references(self, *path):
        pass


class PythonScript(PythonCode):

    def __init__(self):
        self.nodes = []
        self.counter = defaultdict(int)
        self.by_reference = {}
        super().__init__(top=self)

    def register(self, child):
        if child is not self:
            self.nodes.append(child)

    def prelude(self):
        result = []
        for node in self.nodes:
            prelude = node.prelude()
            if prelude:
                if not isinstance(prelude, (list, tuple)):
                    prelude = list(prelude)
                result.extend(prelude)
        return "\n".join(result)

    def source_code(self, first):

        which = self.nodes.index(first)
        first.apply_references()
        for node in self.nodes:
            node.update_anchor()

        more = True
        while more:
            more = False

            by_class = defaultdict(list)
            for node in self.nodes:
                by_class[(node.__class__, node.key)].append(node)

            for nodes in by_class.values():
                if len(nodes) > 1:
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
                self.prelude(),
                f"r.input = {repr(first)}",
                "r.dump()",
            ]
        )

    def function(self, node):
        return Function(node.name, node, self.counter)

    def replace_nodes(self, changes):

        for old, new in changes:
            assert old in self.nodes, f"Node {old} not found in {self.nodes}"
            for i, node in enumerate(self.nodes):

                if node is old:
                    self.nodes[i] = new
                else:
                    node.replace_node(old, new)
