# Goal — two-step read & multi-cutout reads

Companion to `docs/adr/adr-3-two-step-read.md`. This file states *what we are
trying to achieve and why*; `PLAN.md` is *how*; `progress.md` is *where we are*.

## North star

Make reads from anemoi datasets **share and minimise I/O across a family of
related views** (multiple training samples, multiple cutouts, sharded grids),
without changing the caller-visible `ds[n]` API and without ever sacrificing
correctness.

The two-step read (collect → factorize → execute) is the mechanism. It is an
**opt-in fast path for rectangular reads**, layered on top of the recursive
`__getitem__`, never a replacement for it.

## Objectives

1. **Keep the ADR honest and live.** The ADR must describe what the code actually
   does. Update it in the same change as the code. (Done for the 2026-06-06 state;
   keep doing it.)

2. **Lock in the "eager path is permanent" decision.** Document and defend that
   the recursive `__getitem__` is co-equal infrastructure, required for all
   non-rectangular wrappers (interpolation, rolling average, missing-date
   fill/skip, zip/chain, complement, tabular) and as the correctness oracle.
   Prevent a future "cleanup" PR from deleting it.

3. **Justify the design with a real use case: multi-cutout + sharding.**
   - `open_dataset(multi=dict(a=..., b=..., c=...))` — open named datasets that
     share one collect/factorize/execute pipeline.
   - Make the expensive global store read **once** and reused/dedup'd across
     several cutouts (today independent opens don't factorize together because
     `ReadPart` identity is `id(self.data)`).
   - Avoid the cutout grid over-read: when a shard asks for a grid subset
     (`index[3]`), don't read the full grid of every constituent and discard it.

4. **Extend `ReadPart` to carry a grid index array** so grid-subset pushdown is
   expressible (currently rectangular-only; fancy indices fall back to eager and
   over-read). Read the **union of needed chunks** via zarr orthogonal indexing.

## Non-goals

- Replacing or deleting the eager recursive reader.
- Supporting interpolation / rolling-average / cross-date / regridding ops in the
  two-step path — these stay eager by design.
- Tabular datasets in the two-step path (separate effort).
- A new public `ds[n]` API surface — the optimisation stays transparent.

## Definition of done (per objective)

- (1) ADR sections "Implementation as built", "Why the eager path is permanent",
  "Next steps" match the code; a CHANGELOG/ADR-update reminder exists.
- (2) A test asserts at least one representative eager-only wrapper still serves
  correct data via fallback, and the decision is written down.
- (3) `multi=` opens datasets, a benchmark shows the shared global read happening
  once across ≥2 cutouts. ✅ (also `ds[{name: index}]` for per-member sharding.)
- (4) A sharded cutout read reads **less than the full read**, bit-identical to
  eager. ✅ — realised as **skipping whole constituent stores** a shard doesn't
  touch (25× / 3.4× on a real cutout) + smaller in-memory output. NOTE (measured):
  on the real fleet the grid axis is **one chunk per field**, so there is *no
  within-chunk byte saving* — a true byte reduction additionally needs the grid
  axis chunked at dataset-creation time (then pushdown reads only touched chunks,
  ~7× fewer bytes — synthetic).
