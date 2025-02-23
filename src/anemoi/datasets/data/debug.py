# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import textwrap
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

from anemoi.utils.text import Tree
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dataset import Dataset

LOG = logging.getLogger(__name__)

DEBUG_ZARR_LOADING = int(os.environ.get("DEBUG_ZARR_LOADING", "0"))
DEBUG_ZARR_INDEXING = int(os.environ.get("DEBUG_ZARR_INDEXING", "0"))

DEPTH = 0

# TODO: make numpy arrays read-only
# a.flags.writeable = False


def css(name: str) -> str:
    """Get the CSS content from a file.

    Parameters
    ----------
    name : str
        The name of the CSS file.

    Returns
    -------
    str
        The CSS content.
    """
    path = os.path.join(os.path.dirname(__file__), f"{name}.css")
    with open(path) as f:
        return f"<style>{f.read()}</style>"


class Node:
    """A class to represent a node in a dataset tree."""

    def __init__(self, dataset: "Dataset", kids: List[Any], **kwargs: Any) -> None:
        """Initializes a Node object.

        Parameters
        ----------
        dataset : Dataset
            The dataset associated with the node.
        kids : List[Any]
            List of child nodes.
        kwargs : Any
            Additional keyword arguments.
        """
        self.dataset = dataset
        self.kids = kids
        self.kwargs = kwargs

    def _put(self, indent: int, result: List[str]) -> None:
        """Helper method to add the node representation to the result list.

        Parameters
        ----------
        indent : int
            Indentation level.
        result : List[str]
            List to store the node representation.
        """

        def _spaces(indent: int) -> str:
            return " " * indent if indent else ""

        result.append(f"{_spaces(indent)}{self.dataset.__class__.__name__}")
        for k, v in self.kwargs.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
                v = textwrap.shorten(v, width=40, placeholder="...")
            result.append(f"{_spaces(indent+2)}{k}: {v}")
        for kid in self.kids:
            kid._put(indent + 2, result)

    def __repr__(self) -> str:
        """Returns the string representation of the node.

        Returns
        -------
        str
            String representation of the node.
        """
        result: List[str] = []
        self._put(0, result)
        return "\n".join(result)

    def graph(self, digraph: List[str], nodes: dict) -> None:
        """Generates a graph representation of the node.

        Parameters
        ----------
        digraph : List[str]
            List to store the graph representation.
        nodes : dict
            Dictionary to store the node labels.
        """
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

    def digraph(self) -> str:
        """Returns the graph representation of the node.

        Returns
        -------
        str
            Graph representation of the node.
        """
        digraph = ["digraph {"]
        digraph.append("node [shape=box];")
        nodes = {}

        self.graph(digraph, nodes)

        for node, label in nodes.items():
            digraph.append(f'{node} [label="{label}"];')

        digraph.append("}")
        return "\n".join(digraph)

    def _html(self, indent: str, rows: List[List[str]]) -> None:
        """Helper method to add the node representation to the HTML rows.

        Parameters
        ----------
        indent : str
            Indentation level.
        rows : List[List[str]]
            List to store the HTML rows.
        """
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

    def html(self) -> str:
        """Returns the HTML representation of the node.

        Returns
        -------
        str
            HTML representation of the node.
        """
        result = [css("debug")]

        result.append('<table class="dataset">')
        rows = []

        self._html("", rows)

        for r in rows:
            s = " ".join(str(x) for x in r)
            result.append(f"<tr><td>{s}</td></tr>")

        result.append("</table>")
        return "\n".join(result)

    def _as_tree(self, tree: Any) -> None:
        """Helper method to add the node representation to the tree.

        Parameters
        ----------
        tree : Any
            Tree object to store the node representation.
        """
        for kid in self.kids:
            n = tree.node(kid)
            kid._as_tree(n)

    def as_tree(self) -> Tree:
        """Returns the tree representation of the node.

        Returns
        -------
        Tree
            Tree representation of the node.
        """

        tree = Tree(self)
        self._as_tree(tree)
        return tree

    @property
    def summary(self) -> str:
        """Returns the summary of the node."""
        return self.dataset.label

    def as_dict(self) -> dict:
        """Returns the dictionary representation of the node.

        Returns
        -------
        dict
            Dictionary representation of the node.
        """
        return {}


class Source:
    """A class used to follow the provenance of a data point."""

    def __init__(self, dataset: Any, index: int, source: Optional[Any] = None, info: Optional[Any] = None) -> None:
        """Initializes a Source object.

        Parameters
        ----------
        dataset : Any
            The dataset associated with the source.
        index : int
            Index of the data point.
        source : Optional[Any], optional
            Source of the data point, by default None.
        info : Optional[Any], optional
            Additional information, by default None.
        """
        self.dataset = dataset

        self.index = index
        self.source = source
        self.info = info

    def __repr__(self) -> str:
        """Returns the string representation of the source.

        Returns
        -------
        str
            String representation of the source.
        """
        p = s = self.source
        while s is not None:
            p = s
            s = s.source

        return f"{self.dataset}[{self.index}, {self.dataset.variables[self.index]}] ({p})"

    def target(self) -> Any:
        """Returns the target source.

        Returns
        -------
        Any
            Target source.
        """
        p = s = self.source
        while s is not None:
            p = s
            s = s.source
        return p

    def dump(self, depth: int = 0) -> None:
        """Dumps the source information.

        Parameters
        ----------
        depth : int, optional
            Indentation level, by default 0.
        """
        print(" " * depth, self)
        if self.source is not None:
            self.source.dump(depth + 1)


def _debug_indexing(method: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """Decorator to debug indexing methods.

    Parameters
    ----------
    method : Callable[..., NDArray[Any]]
        The method to be decorated.

    Returns
    -------
    Callable[..., NDArray[Any]]
        The decorated method.
    """

    @wraps(method)
    def wrapper(self: Any, index: Any) -> NDArray[Any]:
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


def _identity(method: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """Identity function.

    Parameters
    ----------
    method : Callable[..., NDArray[Any]]
        Input method.

    Returns
    -------
    Callable[..., NDArray[Any]]
        The input method.
    """
    return method


if DEBUG_ZARR_INDEXING:
    debug_indexing = _debug_indexing
else:
    debug_indexing = _identity


def debug_zarr_loading(on_off: int) -> None:
    """Enables or disables Zarr loading debugging.

    Parameters
    ----------
    on_off : int
        1 to enable debugging, 0 to disable.
    """
    global DEBUG_ZARR_LOADING
    DEBUG_ZARR_LOADING = on_off
