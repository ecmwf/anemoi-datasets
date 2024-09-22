# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import textwrap
from functools import wraps

LOG = logging.getLogger(__name__)

DEBUG_ZARR_LOADING = int(os.environ.get("DEBUG_ZARR_LOADING", "0"))
DEBUG_ZARR_INDEXING = int(os.environ.get("DEBUG_ZARR_INDEXING", "0"))

DEPTH = 0

# TODO: make numpy arrays read-only
# a.flags.writeable = False


def css(name):
    path = os.path.join(os.path.dirname(__file__), f"{name}.css")
    with open(path) as f:
        return f"<style>{f.read()}</style>"


class Node:
    def __init__(self, dataset, kids, **kwargs):
        self.dataset = dataset
        self.kids = kids
        self.kwargs = kwargs

    def _put(self, indent, result):
        def _spaces(indent):
            return " " * indent if indent else ""

        result.append(f"{_spaces(indent)}{self.dataset.__class__.__name__}")
        for k, v in self.kwargs.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
                v = textwrap.shorten(v, width=40, placeholder="...")
            result.append(f"{_spaces(indent+2)}{k}: {v}")
        for kid in self.kids:
            kid._put(indent + 2, result)

    def __repr__(self):
        result = []
        self._put(0, result)
        return "\n".join(result)

    def graph(self, digraph, nodes):
        label = self.dataset.label  # dataset.__class__.__name__.lower()
        if self.kwargs:
            param = []
            for k, v in self.kwargs.items():
                if k == "path" and isinstance(v, str):
                    v = os.path.basename(v)
                if isinstance(v, (list, tuple)):
                    v = ", ".join(str(i) for i in v)
                else:
                    v = str(v)
                v = textwrap.shorten(v, width=40, placeholder="...")
                # if len(self.kwargs) == 1:
                #     param.append(v)
                # else:
                param.append(f"{k}={v}")
            label = f'{label}({",".join(param)})'

        label += "\n" + "\n".join(
            textwrap.shorten(str(v), width=40, placeholder="...")
            for v in (
                self.dataset.dates[0],
                self.dataset.dates[-1],
                self.dataset.frequency,
                self.dataset.shape,
                self.dataset.variables,
            )
        )

        nodes[f"N{id(self)}"] = label
        for kid in self.kids:
            digraph.append(f"N{id(self)} -> N{id(kid)}")
            kid.graph(digraph, nodes)

    def digraph(self):
        digraph = ["digraph {"]
        digraph.append("node [shape=box];")
        nodes = {}

        self.graph(digraph, nodes)

        for node, label in nodes.items():
            digraph.append(f'{node} [label="{label}"];')

        digraph.append("}")
        return "\n".join(digraph)

    def _html(self, indent, rows):

        kwargs = {}

        for k, v in self.kwargs.items():
            if k == "path" and isinstance(v, str):
                v = v[::-1]
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
            else:
                v = str(v)
            v = textwrap.shorten(v, width=80, placeholder="...")
            if k == "path":
                v = v[::-1]
            kwargs[k] = v
        label = self.dataset.label
        label = f'<span class="dataset">{label}</span>'
        if len(kwargs) == 1:
            k, v = list(kwargs.items())[0]
            rows.append([indent] + [label, v])
        else:
            rows.append([indent] + [label])

            for k, v in kwargs.items():
                rows.append([indent] + [f"<span class='param'>{k}</span>", f"<span class='param'>{v}</span>"])

        for kid in self.kids:
            kid._html(indent + "&nbsp;&nbsp;&nbsp;", rows)

    def html(self):
        result = [css("debug")]

        result.append('<table class="dataset">')
        rows = []

        self._html("", rows)

        for r in rows:
            s = " ".join(str(x) for x in r)
            result.append(f"<tr><td>{s}</td></tr>")

        result.append("</table>")
        return "\n".join(result)

    def _as_tree(self, tree):

        for kid in self.kids:
            n = tree.node(kid)
            kid._as_tree(n)

    def as_tree(self):
        from anemoi.utils.text import Tree

        tree = Tree(self)
        self._as_tree(tree)
        return tree

    @property
    def summary(self):
        return self.dataset.label

    def as_dict(self):
        return {}


class Source:
    """Class used to follow the provenance of a data point."""

    def __init__(self, dataset, index, source=None, info=None):
        self.dataset = dataset
        self.index = index
        self.source = source
        self.info = info

    def __repr__(self):
        p = s = self.source
        while s is not None:
            p = s
            s = s.source

        return f"{self.dataset}[{self.index}, {self.dataset.variables[self.index]}] ({p})"

    def target(self):
        p = s = self.source
        while s is not None:
            p = s
            s = s.source
        return p

    def dump(self, depth=0):
        print(" " * depth, self)
        if self.source is not None:
            self.source.dump(depth + 1)


def _debug_indexing(method):
    @wraps(method)
    def wrapper(self, index):
        global DEPTH
        # if isinstance(index, tuple):
        print("  " * DEPTH, "->", self, method.__name__, index)
        DEPTH += 1
        result = method(self, index)
        DEPTH -= 1
        # if isinstance(index, tuple):
        print("  " * DEPTH, "<-", self, method.__name__, result.shape)
        return result

    return wrapper


def _identity(x):
    return x


if DEBUG_ZARR_INDEXING:
    debug_indexing = _debug_indexing
else:
    debug_indexing = _identity


def debug_zarr_loading(on_off):
    global DEBUG_ZARR_LOADING
    DEBUG_ZARR_LOADING = on_off
