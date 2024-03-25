# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import sys

# https://en.wikipedia.org/wiki/Box-drawing_character
from collections import defaultdict

from termcolor import colored


def dotted_line(n=84, file=sys.stdout):
    print("┈" * n, file=file)


def boxed(text, min_width=80, max_width=80):
    lines = text.split("\n")
    width = max(len(_) for _ in lines)

    if min_width is not None:
        width = max(width, min_width)

    if max_width is not None:
        width = min(width, max_width)
        lines = []
        for line in text.split("\n"):
            if len(line) > max_width:
                line = line[: max_width - 1] + "…"
            lines.append(line)
        text = "\n".join(lines)

    box = []
    box.append("┌" + "─" * (width + 2) + "┐")
    for line in lines:
        box.append(f"│ {line:{width}} │")

    box.append("└" + "─" * (width + 2) + "┘")
    return "\n".join(box)


def bold(text):
    return colored(text, attrs=["bold"])


def red(text):
    return colored(text, "red")


def green(text):
    return colored(text, "green")


class Tree:
    def __init__(self, actor, parent=None):
        self._actor = actor
        self._kids = []
        self._parent = parent

    def adopt(self, kid):
        kid._parent._kids.remove(kid)
        self._kids.append(kid)
        kid._parent = self
        # assert False

    def forget(self):
        self._parent._kids.remove(self)
        self._parent = None

    @property
    def is_leaf(self):
        return len(self._kids) == 0

    @property
    def key(self):
        return tuple(sorted(self._actor.as_dict().items()))

    @property
    def _text(self):
        return self._actor.summary

    @property
    def summary(self):
        return self._actor.summary

    def as_dict(self):
        return self._actor.as_dict()

    def node(self, actor, insert=False):
        node = Tree(actor, self)
        if insert:
            self._kids.insert(0, node)
        else:
            self._kids.append(node)
        return node

    def print(self, file=sys.stdout):
        padding = []

        while self._factorise():
            pass

        self._print(padding, file=file)

    def _leaves(self, result):
        if self.is_leaf:
            result.append(self)
        else:
            for kid in self._kids:
                kid._leaves(result)

    def _factorise(self):
        if len(self._kids) == 0:
            return False

        result = False
        for kid in self._kids:
            result = kid._factorise() or result

        if result:
            return True

        same = defaultdict(list)
        for kid in self._kids:
            for grand_kid in kid._kids:
                same[grand_kid.key].append((kid, grand_kid))

        result = False
        n = len(self._kids)
        texts = []
        for text, v in same.items():
            if len(v) == n and n > 1:
                for kid, grand_kid in v:
                    kid._kids.remove(grand_kid)
                texts.append((text, v[1][1]))
                result = True

        for text, actor in reversed(texts):
            self.node(actor, True)

        if result:
            return True

        if len(self._kids) != 1:
            return False

        kid = self._kids[0]
        texts = []
        for grand_kid in list(kid._kids):
            if len(grand_kid._kids) == 0:
                kid._kids.remove(grand_kid)
                texts.append((grand_kid.key, grand_kid))
                result = True

        for text, actor in reversed(texts):
            self.node(actor, True)

        return result

    def _print(self, padding, file=sys.stdout):
        for i, p in enumerate(padding[:-1]):
            if p == " └":
                padding[i] = "  "
            if p == " ├":
                padding[i] = " │"
        if padding:
            print(f"{''.join(padding)}─{self._text}", file=file)
        else:
            print(self._text, file=file)
        padding.append(" ")
        for i, k in enumerate(self._kids):
            sep = " ├" if i < len(self._kids) - 1 else " └"
            padding[-1] = sep
            k._print(padding, file=file)

        padding.pop()

    def to_json(self, depth=0):
        while self._factorise():
            pass

        return {
            "actor": self._actor.as_dict(),
            "kids": [k.to_json(depth + 1) for k in self._kids],
            "depth": depth,
        }


def table(rows, header, align, margin=0):
    def _(x):
        try:
            x = float(x)
        except Exception:
            pass

        if isinstance(x, float):
            return f"{x:g}"

        if isinstance(x, str):
            return x
        if isinstance(x, int):
            return str(x)

        return str(x)

    tmp = []
    for row in rows:
        tmp.append([_(x) for x in row])

    all_rows = [header] + tmp

    lens = [max(len(x) for x in col) for col in zip(*all_rows)]

    result = []
    for i, row in enumerate(all_rows):
        result.append(
            " │ ".join([x.ljust(i) if align[j] == "<" else x.rjust(i) for j, (x, i) in enumerate(zip(row, lens))])
        )
        if i == 0:
            result.append("─┼─".join(["─" * i for i in lens]))

    result.append("─┴─".join(["─" * i for i in lens]))

    if margin:
        result = [margin * " " + x for x in result]

    return "\n".join(result)


def progress(done, todo, width=80):
    done = min(int(done / todo * width + 0.5), width)
    return green("█" * done) + red("█" * (width - done))
