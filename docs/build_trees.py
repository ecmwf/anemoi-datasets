"""Render the view tree of the documentation examples as text.

The example scripts under ``using/code`` (and the configuration under
``using/yaml``) open datasets by their real names, e.g.
``open_dataset("aifs-...-v8.zarr")``. Those stores are not available when the
documentation is built, so this helper temporarily replaces the store loader:
when a name has a JSON sidecar next to the example (``<name>.json`` or
``<script>-<name>.json``), a lightweight :ref:`synthetic <synthetic-datasets>`
dataset described by that file is used instead.

Each example is executed, its ``ds.tree()`` is rendered as a ``rich`` tree
(pure text, no external ``dot`` binary) and written next to the others under
``using/trees`` so it can be included right after the example. Regenerated on
every build (wired from ``conf.py``), so the pictures cannot drift.
"""

from __future__ import annotations

import json
import os
import runpy
from typing import Any

import yaml
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets.usage.misc import _open_dataset
from anemoi.datasets.usage.store import ZarrStore

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "using", "code")
YAML = os.path.join(HERE, "using", "yaml")
OUT = os.path.join(HERE, "using", "trees")

# The examples whose view tree is rendered, and where to write it.
EXAMPLES = [
    {"stem": "compose_tree", "script": os.path.join(CODE, "compose_tree.py")},
    {"stem": "compose_config", "config": os.path.join(YAML, "compose_config.yaml")},
    {"stem": "combine_concat", "script": os.path.join(CODE, "combine_concat.py")},
    {"stem": "combine_cutout", "script": os.path.join(CODE, "combine_cutout.py")},
]

SIDECAR_DIRS = [CODE, YAML]

_NAMES: dict[int, str] = {}


def _sidecar(stem: str, name: str) -> str | None:
    """Return the synthetic description for a dataset name, if any.

    Parameters
    ----------
    stem : str
        The identifier of the example being rendered.
    name : str
        The dataset name passed to ``open_dataset``.

    Returns
    -------
    dict or None
        The parsed synthetic description, or ``None`` when no sidecar exists.
    """
    base = os.path.basename(name)
    for directory in SIDECAR_DIRS:
        for candidate in (f"{stem}-{base}.json", f"{base}.json"):
            path = os.path.join(directory, candidate)
            if os.path.exists(path):
                with open(path) as file:
                    return json.load(file)
    return None


def _patch(stem: str) -> Any:
    """Replace the store loader with the sidecar-backed synthetic loader.

    Parameters
    ----------
    stem : str
        The identifier of the example being rendered.

    Returns
    -------
    callable
        The original, unpatched ``ZarrStore.from_name_or_path`` function.
    """
    original = ZarrStore.from_name_or_path.__func__

    def loader(cls: type, name: str, options: Any = None) -> Any:
        spec = _sidecar(stem, name)
        if spec is None:
            return original(cls, name, options)
        dataset = _open_dataset(synthetic=spec, options=options)
        _NAMES[id(dataset)] = name
        return dataset

    ZarrStore.from_name_or_path = classmethod(loader)
    return original


def _label(node: Any) -> str:
    """Build the tree label for a single view.

    Parameters
    ----------
    node : anemoi.datasets.usage.debug.Node
        The tree node to render.

    Returns
    -------
    str
        The label to display for the node.
    """
    name = _NAMES.get(id(node.dataset))
    if name is not None and not node.kids:
        return f"GriddedZarr {name}"

    label = type(node.dataset).__name__
    params = []
    for key, value in node.kwargs.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        params.append(f"{key}={value}")
    if params:
        label += " (" + ", ".join(params) + ")"
    return label


def _add(node: Any, branch: Tree) -> None:
    """Attach a node and its children to a ``rich`` tree branch.

    Parameters
    ----------
    node : anemoi.datasets.usage.debug.Node
        The tree node to render.
    branch : rich.tree.Tree
        The branch to attach the node to.
    """
    child = branch.add(_label(node))
    for kid in node.kids:
        _add(kid, child)


def _render(dataset: Any) -> str:
    """Render the view tree of a dataset as text.

    Parameters
    ----------
    dataset : anemoi.datasets.usage.dataset.Dataset
        The (possibly composed) dataset to draw.

    Returns
    -------
    str
        The rendered tree as plain text.
    """
    root = dataset.tree()
    tree = Tree(_label(root))
    for kid in root.kids:
        _add(kid, tree)

    console = Console(record=True, width=88, force_terminal=False)
    console.print(tree)
    return console.export_text()


def _write(path: str, text: str) -> None:
    """Write text to a file only when its content would change.

    Parameters
    ----------
    path : str
        The destination path.
    text : str
        The content to write.
    """
    if os.path.exists(path):
        with open(path) as file:
            if file.read() == text:
                return
    with open(path, "w") as file:
        file.write(text)


def generate(*_: Any) -> None:
    """Regenerate every example view tree used by the docs."""
    os.makedirs(OUT, exist_ok=True)

    for example in EXAMPLES:
        stem = example["stem"]
        _NAMES.clear()
        original = _patch(stem)
        try:
            if "script" in example:
                dataset = runpy.run_path(example["script"])["ds"]
            else:
                with open(example["config"]) as file:
                    dataset = _open_dataset(**yaml.safe_load(file)["dataset"])
        finally:
            ZarrStore.from_name_or_path = classmethod(original)

        _write(os.path.join(OUT, f"{stem}.txt"), _render(dataset))


if __name__ == "__main__":
    generate()
