# Usage documentation — suggestions for discussion

These are proposals arising from the revisit of the **usage** part of the
documentation (the "using an existing dataset" section). Nothing here has
been applied to the shipped docs beyond what is listed under "Done in this
pass"; the rest is up for discussion.

## Done in this pass

- Added a new concepts page `docs/using/how-it-works.rst` that explains
  composability at three levels (high / middle / low), the `tree()`
  method, the metadata-rich nature of the objects, and why a single
  `open_dataset()` driven by a dictionary is used. It now includes two
  Graphviz diagrams showing the tree of objects (a linear pipeline and a
  branching `join`).
- Moved inline code out of `introduction.rst` and `opening.rst` into
  external files under `docs/using/code/` and `docs/using/yaml/`, included
  via `.. literalinclude::` (matching the convention used in
  `docs/howtos/using/`).
- Fixed a broken example in `introduction.rst` (`ds` was used before being
  defined).
- Added a `.. _using-methods:` anchor to `methods.rst` and linked the
  `tree()` / `metadata()` mentions to it.
- Simple corrections during the full review:
  - `subsetting.rst`: a bullet list about `frequency`/`end` was collapsed
    into a single paragraph (broken `-` markers); reformatted as a list.
  - `missing.rst`: exception name `MissingDatesError` -> `MissingDateError`
    (the actual class name).
  - `combining.rst`: fixed six image paths with a double slash
    (`../_static//...` -> `../_static/...`).
  - `grids.rst`: `trim_edge=(3, 10, 4, 2)` was described as removing "10 in
    the east" — corrected to "2 in the east".
  - `synthetic.rst`: title over/underline was too short (Sphinx warning);
    lengthened.
  - `selecting.rst`: removed a dangling, duplicate `.. _number:` target at
    end of file (the canonical one lives in `ensembles.rst`).
  - `parameters.rst`: typos `Concatanate` -> `Concatenate`, `continous` ->
    `continuous`; fixed the `area` ordering to `(north, west, south,
    east)` to match `grids.rst`.
  - `methods.rst`: removed a stray colon after `statistics_tendencies(delta)`.

## Naming the objects (feedback requested)

The objects returned by `open_dataset` and its operations have no name in
the docs. The new page provisionally calls them **dataset views**. Ranked
proposals:

1. **views** — analogy with NumPy views: lazy objects that behave like the
   data they wrap. Clear, short, familiar. (Current choice.)
2. **operators** — emphasises that each is an operation on datasets.
3. **adapters** — emphasises the wrapping/forwarding behaviour.
4. **nodes** — matches the `tree()` / network mental model.
5. **lenses** / **combinators** / **pipelines** — more niche, but capture
   the compositional intent.

Recommendation: standardise on one term and use it consistently across the
whole usage section (and ideally in docstrings / class-level docs).

## Consistency / correctness issues found

- `introduction.rst` toctree references `window`, but there is no
  `docs/using/window.rst`. Either add the page or remove the entry (it is
  likely a broken/leftover reference).
- `parameters.rst` still lists `chain` and `concat` behaviour that reads as
  provisional ("should be skipped for now"). Worth confirming whether these
  notes are still accurate.

## Structural suggestions

- Place `how-it-works.rst` early in the toctree (currently right after
  `opening`) so readers get the mental model before the per-operation
  reference pages.
- Consider grouping the per-operation pages under two explicit headings in
  the toctree: **operations on one dataset** (subset, select, rescale,
  fill missing, ...) and **operations that combine datasets** (join,
  concat, cutout, grids, zip, ensembles). This mirrors the
  `Forwards` / `Combined` split in the code and reinforces composability.
- Add a short "cheat sheet" table mapping each keyword argument of
  `open_dataset` to the page that documents it and to the view class that
  implements it.
