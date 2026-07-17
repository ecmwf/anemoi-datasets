# Usage documentation — suggestions for discussion

These are proposals arising from the revisit of the **usage** part of the
documentation (the "using an existing dataset" section). Nothing here has
been applied to the shipped docs beyond what is listed under "Done in this
pass"; the rest is up for discussion.

## Done in this pass

- Added a new concepts page `docs/using/how-it-works.rst` that explains
  composability at three levels (high / middle / low), the `tree()`
  method, the metadata-rich nature of the objects, and why a single
  `open_dataset()` driven by a dictionary is used.
- Moved inline code out of `introduction.rst` and `opening.rst` into
  external files under `docs/using/code/` and `docs/using/yaml/`, included
  via `.. literalinclude::` (matching the convention used in
  `docs/howtos/using/`).
- Fixed a broken example in `introduction.rst` (`ds` was used before being
  defined).

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
- Several other pages in `docs/using/` still contain inline `.. code::`
  blocks (e.g. `combining.rst`, `selecting.rst`, `subsetting.rst`,
  `methods.rst`, `statistics.rst`, `grids.rst`, `ensembles.rst`,
  `missing.rst`, `zip.rst`, `matching.rst`, `configuration.rst`,
  `other.rst`, `miscellaneous.rst`). For full consistency these should also
  be externalised into `code/` and `yaml/`. Suggest doing this
  page-by-page rather than in one large change.
- Reference-label naming is inconsistent: statistics/ensembles/grids/zip/
  other/missing all use `selecting-*` anchors even though they are not
  about selecting variables. Consider renaming to a consistent scheme.
- `methods.rst` has no `.. _...:` anchor at the top, so it cannot be
  cross-referenced with `:ref:`. Suggest adding one (e.g.
  `.. _using-methods:`) and linking `tree()` / `metadata()` mentions to it.

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
