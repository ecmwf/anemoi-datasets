# Two-step read: collect parts, factorize, execute

## Status

Accepted, implemented — last reviewed 2026-06-07. Fallback is by `None` return
(not exception); grid-subset pushdown + `multi=` (incl. `ds[{name: index}]`) +
union factorization shipped; `execute_parts` sequential; kill-switch live.
Known limitation: +48% peak memory on full cutout reads (see below). The design
sketch sections below are the *original proposal* — "Implementation as built" and
the per-workstream sections are the source of truth for what runs.

- The collect → factorize → execute pipeline exists and is wired in as an
  **always-on fast path with automatic fallback** (see "Implementation as built").
- It is implemented for the common rectangular-read wrappers
  (`ZarrStore`/`GriddedZarr`, `Forwards`, `Select`, `Subset`, `Concat`, `Cutout`,
  `Join`, `Merge`, `Ensemble`, `Masked`, …).
- It is **deliberately not** implemented for wrappers whose semantics are not a
  rectangular slice of a store (interpolation, rolling average, missing-date fill
  / skip, `zip`/`chain`, `complement`, tabular). For those the eager
  `__getitem__` runs. **This is permanent, not a migration gap** — see
  "Why the eager `__getitem__` path is permanent".

> **Maintenance**: this ADR is a living document. When you change `__getitem__`,
> `collect_read_parts`, `read_from_buffer`, or the gating in `Dataset.__init_subclass__`,
> update the matching section here in the same change. See `docs/adr/two-step-read/`
> for the goal, plan and live progress log.

## Context

Currently `ds[n]` triggers a recursive `__getitem__` chain that reads zarr immediately at
the leaf level. Each wrapper (`Select`, `Subset`, `Concat`, `Cutout`, …) transforms the
index and calls its child, which calls its child, until `GriddedZarr` / `TrajectoriesZarr`
hits `self.data[n]`.

Problems with this:
- No visibility into what is being read (opaque I/O).
- No opportunity to deduplicate or coalesce reads before executing them.
- `Select` reads all variables then slices in memory — the zarr read fetches more than needed.
- `Cutout` performs multiple independent zarr reads that could be batched.
- No path to parallelise I/O without rewriting every `__getitem__`.

## Goal

Split `ds[n]` into three phases:

```
1. COLLECT  — walk the wrapper tree, accumulate a list of (dataset_path, slice, slice, ...) tuples
2. FACTORIZE — merge / deduplicate overlapping reads
3. EXECUTE  — sequential zarr reads (parallelisation added later)
```

The caller-visible API (`ds[n]`) stays unchanged. Phases run transparently inside.

## Key abstractions

### `ReadPart`

```python
@dataclass(frozen=True)
class ReadPart:
    path: str                   # zarr store path / URL — identifies the physical store
    slices: tuple[slice, ...]   # one normalised slice per zarr dimension
    squeeze: tuple[int, ...]    # axes to squeeze after read (were originally ints)
```

Rules:
- Every integer index is immediately converted to `slice(n, n+1)` with the axis recorded in
  `squeeze`.  No integers are stored.
- Lists / arrays are not stored — callers convert them to contiguous slices or handle via
  `expand_list_indexing` before collecting.  The `squeeze` / `concat` bookkeeping lives in
  the existing `expand_list_indexing` decorator, which stays as-is.
- Shape: gridded → 4 slices `(date, var, ens, grid)`;
         trajectories → 5 slices `(base_date, var, ens, step, cell)`.

### `ReadBuffer`

```python
class ReadBuffer:
    raw: dict[ReadPart, NDArray]      # factorized part → read array
    mapping: dict[ReadPart, tuple[ReadPart, int, list | None]]
    # original part → (merged_part, date_row_offset, grid_cols)  [grid_cols: D2 union]

    def __getitem__(self, part: ReadPart) -> NDArray:
        merged, offset, grid_cols = self.mapping.get(part, (part, 0, None))
        data = self.raw[merged][offset : offset + part_n_rows(part)]
        return data if grid_cols is None else data[..., grid_cols]
```

### New methods on `Dataset` ABC

```python
class Dataset(ABC):

    def collect_read_parts(self, index: FullIndex) -> list[ReadPart] | None:
        """Phase 1: return all zarr reads for this index (no I/O), or None to
        signal 'use eager' (a normal outcome, not an error).  Base default: None."""
        return None

    def read_from_buffer(self, index: FullIndex, buffer: ReadBuffer) -> NDArray:
        """Phase 3: reconstruct result using the pre-fetched buffer."""
        raise NotImplementedError   # only reached for a supported wrapper
```

`collect_read_parts` + `read_from_buffer` are mirror images of `__getitem__`: same index
transformations, but `collect_read_parts` accumulates leaf reads while `read_from_buffer`
assembles the result from cached data.

## Per-class implementation sketch

### `Forwards` (default for all transparent wrappers)

```python
def collect_read_parts(self, n):
    return self.forward.collect_read_parts(n)

def read_from_buffer(self, n, buffer):
    return self.forward.read_from_buffer(n, buffer)
```

### `GriddedZarr` / `TrajectoriesZarr` — leaf nodes

```python
def collect_read_parts(self, n):
    slices, squeeze = index_to_slices(n, self.data.shape)
    return [ReadPart(self.path, slices, squeeze)]

def read_from_buffer(self, n, buffer):
    slices, squeeze = index_to_slices(n, self.data.shape)
    part = ReadPart(self.path, slices, squeeze)
    data = buffer[part]
    return apply_index_to_slices_changes(data, squeeze)
```

### `Select` — variable subsetting

```python
def collect_read_parts(self, n):
    if isinstance(n, tuple):
        index, _ = index_to_slices(n, self.shape)
        index, _ = update_tuple(index, 1, slice(None))  # request all vars
        return self.dataset.collect_read_parts(index)
    return self.dataset.collect_read_parts(n)

def read_from_buffer(self, n, buffer):
    # exact mirror of __getitem__ but recursive call uses read_from_buffer
    if isinstance(n, tuple):
        index, changes = index_to_slices(n, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.dataset.read_from_buffer(index, buffer)
        result = result[:, self.indices][:, previous]
        return apply_index_to_slices_changes(result, changes)
    row = self.dataset.read_from_buffer(n, buffer)
    if isinstance(n, slice):
        return row[:, self.indices]
    return row[self.indices]
```

### `Subset` — date index remapping

```python
def collect_read_parts(self, n):
    if isinstance(n, slice):
        inner_indices = [self.indices[i] for i in range(*n.indices(self._len))]
        inner = make_slice_or_index_from_list_or_tuple(inner_indices)
    elif isinstance(n, tuple):
        # remap date axis
        ...
    else:
        inner = self.indices[n]
    return self.dataset.collect_read_parts(inner)

def read_from_buffer(self, n, buffer):
    # same remapping, then delegate
    ...
```

### `Concat` — time concatenation across multiple datasets

```python
def collect_read_parts(self, n):
    if isinstance(n, tuple):
        index, _ = index_to_slices(n, self.shape)
        slices = length_to_slices(index[0], [d.shape[0] for d in self.datasets])
        parts = []
        for d, s in zip(self.datasets, slices):
            if s is not None:
                parts.extend(d.collect_read_parts(update_tuple(index, 0, s)[0]))
        return parts
    ...

def read_from_buffer(self, n, buffer):
    # same split, collect results, np.concatenate
    ...
```

### `Cutout` — multi-dataset with spatial masks

```python
def collect_read_parts(self, index):
    if isinstance(index, (int, slice)):
        index = (index, slice(None), slice(None), slice(None))
    index, _ = index_to_slices(index, self.shape)
    parts = []
    for lam in self.lams:
        parts.extend(lam.collect_read_parts(index[:3]))
    parts.extend(self.globe.collect_read_parts(index[:3]))
    return parts

def read_from_buffer(self, index, buffer):
    # same reads from buffer, apply masks, concatenate — mirror of _get_tuple
    ...
```

`Cutout` is a key case: when LAMs and the globe share the same zarr path (they don't in
practice) factorization would merge their date reads.  More realistically the benefit is
when training loops call `ds[i]`, `ds[i+1]`, `ds[i+2]` in a batch — factorization merges
those three date reads into one.

## Factorization algorithm

```
factorize(parts: list[ReadPart]) -> (list[ReadPart], dict[ReadPart, (ReadPart, int)]):

  group by (path, slices[1:])   # same path + same non-date axes

  for each group:
    sort by slices[0].start
    merge all date slices into slice(min_start, max_stop)   # simple bounding-box merge
    merged_part = ReadPart(path, (merged_date_slice,) + rest_slices, squeeze=())
    for each original part in group:
      offset = part.slices[0].start - merged_date_slice.start
      mapping[original] = (merged_part, offset)

  return deduplicated merged_parts, mapping
```

Notes:
- Bounding-box merge may read more data than needed when parts are sparse.  Start with
  bounding-box; refine to gap-aware merging later if profiling shows wasted bandwidth.
- Parts with different var/ens/grid slices are NOT merged (different groups).  Accept this
  for v1; the common training loop always requests the same var/ens/grid slice per batch.

## Execution

```python
def execute_parts(factorized: list[ReadPart]) -> dict[ReadPart, NDArray]:
    buffer = {}
    for part in factorized:
        store = _open_zarr(part.path)          # cached open, not re-opened each time
        buffer[part] = store["data"][part.slices]
    return buffer
```

Sequential for now.  Later: `concurrent.futures.ThreadPoolExecutor` wrapping this loop,
since zarr/S3 reads release the GIL.

## Top-level orchestration

New function in `usage/read_parts.py`:

```python
def two_step_read(dataset: Dataset, index: FullIndex) -> NDArray:
    parts = dataset.collect_read_parts(index)
    if READ_PARTS_DEBUG:
        _log_parts("collected", parts)
    factorized, mapping = factorize(parts)
    if READ_PARTS_DEBUG:
        _log_parts("factorized", factorized)
    raw = execute_parts(factorized)
    buffer = ReadBuffer(raw=raw, mapping=mapping)
    return dataset.read_from_buffer(index, buffer)
```

`Dataset.__getitem__` calls `two_step_read` once all classes implement the new methods.
During migration, a fallback stays in place (see migration below).

## Debug logging

```python
READ_PARTS_DEBUG = os.environ.get("ANEMOI_DATASETS_READ_PARTS_DEBUG", "").lower() in ("1", "true")

LOG_READ_PARTS = logging.getLogger("anemoi.datasets.read_parts")

def _log_parts(label: str, parts: list[ReadPart]) -> None:
    LOG_READ_PARTS.debug("%s: %d parts", label, len(parts))
    for p in parts:
        LOG_READ_PARTS.debug("  %s  slices=%s", p.path, p.slices)
```

Enable with `ANEMOI_DATASETS_READ_PARTS_DEBUG=1` (forces debug) or by setting log level on
`anemoi.datasets.read_parts` logger.

Decorator variant for quick profiling on hot paths:

```python
def track_read_parts(method):
    @wraps(method)
    def wrapper(self, index, *args, **kwargs):
        if READ_PARTS_DEBUG:
            parts = self.collect_read_parts(index)
            LOG_READ_PARTS.debug("%s[%s] → %d parts", type(self).__name__, index, len(parts))
        return method(self, index, *args, **kwargs)
    return wrapper
```

## Implementation as built (2026-06-06)

The shipped code diverges from the original sketch above in several ways. The
sketch is kept for design intent; this section is the source of truth for *what
actually runs*.

### Wiring: always-on, fallback by **return value (`None`)**, not by exception

`Dataset.__init_subclass__` transparently wraps every subclass `__getitem__`:

```python
def __init_subclass__(cls, **kwargs):
    if "__getitem__" in cls.__dict__:
        _orig = cls.__dict__["__getitem__"]
        @functools.wraps(_orig)
        def _gated(self, n, _f=_orig):
            if not READ_PARTS_ENABLED:        # kill-switch (ANEMOI_DATASETS_READ_PARTS=0)
                return _f(self, n)
            result = two_step_read(self, n)   # collect → factorize → execute
            return _f(self, n) if result is None else result   # None → eager
        cls.__getitem__ = _gated
```

**Fallback is a normal return value, not an exception.** `collect_read_parts`
returns ``None`` to say "I do not support the two-step fast path for this index";
``two_step_read`` then returns ``None`` and the gate uses the eager reader.
Reading via eager is an *expected* outcome (most non-rectangular wrappers do it),
so it must not be signalled by an exception. Rules:

- A wrapper that does not support two-step (interpolation, rolling average, fill /
  skip missing, zip / chain, complement, tabular, trajectory subsets) returns
  ``None`` from `collect_read_parts` (the base `Dataset.collect_read_parts` default).
- A multi-dataset wrapper (Concat, Join, Merge, Grids, Cutout, Multi) returns
  ``None`` if **any** child returns ``None`` (helper `gather_parts`).
- The only exceptions that mean "fall back" are raised by `index_to_slices` for an
  index that is not expressible as rectangular slices (a list/array on a non-grid
  axis); `two_step_read` catches `(ValueError, TypeError, AttributeError)` there
  and returns ``None``. **Real errors propagate** — notably `IndexError` for
  out-of-range integers (the leaf validates int bounds with `check_int_bounds`
  *before* `index_to_slices`, which otherwise silently `i % size`-wraps, e.g.
  `ds[n_dates, …]` → date 0).

Consequences:

- The two-step path is **always attempted first** unless the kill-switch is off.
- Fallback is **per node, not whole-tree**. If a deep wrapper returns ``None``, it
  propagates up (via `gather_parts` / direct return) to the *nearest enclosing*
  `_gated.__getitem__`, which runs that node's eager `__getitem__`; that eager
  method indexes its children with `child[...]`, itself gated → re-enters
  `two_step_read`. So a tree with one unsupported node still gets two-step reads
  for the supported subtrees below it. Cost: the top-level collect walk is
  attempted and discarded before the eager path re-walks — redundant traversal,
  no redundant I/O (collect does no reads).

### `ReadPart` carries a live store reference, not a path to re-open

The sketch keyed reads by `path: str` and re-opened the store in `execute_parts`.
The shipped `ReadPart` instead holds the **live `zarr.Array`** (`self.data`) and
keeps `path` only as a logging/repr label. Identity (`__hash__`/`__eq__`) is
`(id(self.data), self.slices)`. This means:

- factorization only merges parts that point at the *same in-memory array
  object* — correct for both on-disk and in-memory/test stores, and it sidesteps
  a store-reopen cache.
- It also means cross-process / cross-open sharing is **not** available: two
  `open_dataset` calls that open the same URL produce different array objects and
  will not factorize together. This matters for the `multi=` proposal below.

Slices are stored normalised as `(start, stop, step)` int-tuples (`NormSlice`),
never `slice` objects, ints, or lists.

### Indices: rectangular fast path + a grid-axis index array

`ZarrStore.collect_read_parts` reduces most indices to plain slices. As of the
grid-subset pushdown (workstream D, below) it *also* accepts an **integer index
array on the last (grid) axis**: `split_grid_index` peels it off and the part
carries it as `ReadPart.grid_index`, executed via zarr **orthogonal indexing**
(`data.oindex[...]`). All other fancy indices (arrays on non-grid axes, boolean
masks) still fall back to eager (the leaf returns `None`) via
`expand_list_indexing`.

### Kill-switch (`ANEMOI_DATASETS_READ_PARTS`)

`READ_PARTS_ENABLED` is now a **real switch**, no longer vestigial. Default on;
set `ANEMOI_DATASETS_READ_PARTS=0` (or `false`/`no`) to force the eager path for
the whole process. The gate (`Dataset.__init_subclass__`) reads the module
attribute each call, so it can be toggled/patched at runtime (tests use this to
fetch a eager oracle result). (Resolves the old "task A4 — decide its fate".)

### Execution is sequential (cross-part thread pool removed)

`execute_parts` reads parts **sequentially**. An earlier version used a
`ThreadPoolExecutor` to read parts concurrently (phase 5, added speculatively).
Real-data benchmarking showed it made a full-grid cutout read **~35% slower** than
eager: reading the 112 MB single-chunk globe and the LAM chunk concurrently
contends on the GIL during blosc **decompression** (CPU/GIL-bound, not I/O-bound),
so the pool was pure overhead. Concurrency where it *does* help — overlapping S3
latency across the chunks of a single array — already happens inside zarr/anemoi
`getitems` (PR #617). So the cross-part pool was removed entirely (along with the
`ANEMOI_DATASETS_READ_PARTS_THREADS` env var and the frozen `num_threads`
default-arg, which silently ignored runtime changes anyway). Sequential is at
parity with eager (0.97–0.99×) and the code is simpler. If a future latency-bound,
many-small-part workload wants cross-part concurrency, add it back as an explicit,
measured option — not a global default.

### Known limitation: peak memory on FULL cutout reads (not yet addressed)

A **full-grid cutout read** (the common training pattern, `ds[t]`) uses ~+48%
peak memory under two-step vs eager (measured 1830 vs 1238 MB on a real
`meps`→`n320` cutout). The `ReadBuffer` holds every constituent's full array
*and* `read_from_buffer` then builds the concatenated output — both alive at peak,
where eager frees intermediates sooner. It is cutout-specific (plain-store reads
are at parity) and a **memory**, not correctness, issue. Sharded reads (the
pushdown case) use *less* memory than eager. Left as a follow-up — options:
drop `ReadBuffer` entries once consumed, or route full-grid cutout reads to eager
(with a `multi=` carve-out so cross-cutout sharing is preserved). Track before
relying on two-step for memory-bound full-cutout training.

## Why the eager `__getitem__` path is permanent (decision)

**Decision: the recursive `__getitem__` is not "old code to delete in Phase 4".
It is a permanent, co-equal execution path.** The two-step read is an *opt-in
fast path for rectangular reads*, layered on top of — not a replacement for —
the recursive reader.

The two-step model can only express an operation whose result is a **rectangular
(affine) selection of one or more stores**: index remapping, variable/member
selection, spatial masking, and concatenation. Any wrapper whose output is *not*
such a selection cannot produce `ReadPart`s, and **must** run eager code. These
are not unfinished migrations — they are structurally outside the model:

| Wrapper | Why it cannot be a rectangular read |
|---|---|
| `InterpolateFrequency` | Output date `n` is a weighted combination of two *different* source dates — cross-date arithmetic, not a slice. |
| `InterpolateNearest` | Output grid is produced by a spatial interpolation/regridding matrix — output cells are linear combos of input cells. |
| `RollingAverage` | Output date `n` reads a *window* of dates and reduces them — many reads + reduction, not one slice. |
| `MissingDatesFill` | Fills a missing date from *adjacent* dates — conditional, data-dependent source selection. |
| `MissingDates` / `SkipMissingDates` | Synthesises NaN arrays (no read) or re-maps around gaps with complex semantics. (The plain "raise on missing" `ZarrWithMissingDates` *is* supported — it only validates then delegates.) |
| `ZipBase` (`xy`) / `Chain` (`unchecked`) | Return a **tuple of arrays**, not a single array — different return contract. |
| `Complement` | Multi-source fill where each output variable may come from a different store with its own regridding. |
| Tabular (`TabularZarr`, `WindowView`, …) | Row/window access pattern, explicitly out of scope (see below). |

The happy path most users hit — open zarr → `select` → `subset`/`start`/`end` →
`concat`/`join` → `cutout` → train — is *fully* covered by two-step, which is
**why a casual read of the code suggests the eager path is dead**. It is not:
the moment a recipe uses `interpolate_frequency`, `rolling_average`,
`fill_missing_dates`, a `zip`/`xy` dataset, `complement`, or any tabular dataset,
the eager reader is the *only* thing that can serve the request. The eager path
also remains the **correctness oracle**: the two-step path is validated by
asserting it returns byte-identical results to eager `__getitem__`.

Practical rules that follow from this decision:

1. **Never delete a working `__getitem__`** when adding `collect_read_parts` /
   `read_from_buffer`. Both must stay and must agree.
2. A new wrapper may ship with **only** `__getitem__` (no two-step). **Returning
   ``None`` from `collect_read_parts`** is a valid, supported state — it just means
   "this node always uses the eager path". (The base `Dataset.collect_read_parts`
   returns ``None``, so a wrapper that doesn't override it is unsupported by
   default.) Do **not** raise to signal this — fallback is normal, not an error.
3. Any change to a supported wrapper's `__getitem__` must be mirrored in its
   `read_from_buffer`, and vice-versa, or the oracle test will (rightly) fail.

## Motivating use cases: cutout grid-subset pushdown and `multi=`

These justify the two-step investment beyond a single simple cutout (where the
payoff is otherwise marginal).

### Problem 1 — cutout over-reads the grid when sharding — **IMPLEMENTED**

The eager `_get_tuple` reads the **full grid** of every constituent
(`index[:3]` slices only date/var/ens), builds the full masked concatenation,
and only *then* applies the output grid slice `index[3]`:

```python
lam_data    = [lam[index[:3]] for lam in self.lams]            # full grid each
globe_data  = self.globe[index[:3]][..., self.global_mask]
result      = np.concatenate(lam_data + [globe_data], axis=self.axis)[..., index[3]]
```

When a training shard asks for a *subset* of the output grid (`index[3] = g0:g1`),
that reads every grid point of the global store (the expensive one) and every
LAM, then discards most of it — ≈N× over-read for a 1/N shard.

**Now fixed in the two-step path.** Files: `ReadPart.grid_index` + `oindex`
execution in `usage/read_parts.py`; `split_grid_index` + grid-array support in
`ZarrStore.collect_read_parts`/`read_from_buffer` (`usage/store.py`);
`Cutout._pushdown_plan` / `_grid_pushdown` / `_is_full_grid_slice` and the
rewritten `collect_read_parts`/`read_from_buffer` (`usage/gridded/grids.py`).
Tests: `tests/test_grid_pushdown.py`.

How it works:

- A `ReadPart` may carry an integer `grid_index` (the points to read on the last
  axis); `execute` then uses `data.oindex[..., grid_index]` (zarr orthogonal
  indexing) so only those points are read.
- `Cutout` precomputes `_pushdown_plan`: the output grid axis is the
  concatenation of each LAM's full grid then the globe's `global_mask` points.
  For a requested output range it maps each output position back to its
  constituent and that constituent's original grid index, and asks the
  constituent for just those points (an index array on the grid axis).
- The leaf (`ZarrStore`) turns a grid-axis index array into a `grid_index`
  `ReadPart`. `Select` (var axis) and `Subset` (date axis) are **grid-array
  transparent** — they peel the last-axis array (`split_grid_index`), apply their
  own transform, reattach and delegate — so pushdown reaches the leaf in real
  cutouts (members opened with `select`/`adjust`). Any other wrapper that can't
  take a grid array is detected by `Cutout._pushdown_supported` (a cached,
  I/O-free probe) and the cutout falls back to the eager full read.
  **Correctness is preserved either way.** (Without the probe, such an array
  raised `ValueError` inside the wrapper and escaped the gate — that crash is now
  fixed.)

Guards (keep it byte-identical to eager where it activates):

- Only for a **non-full** grid slice with **step ≥ 1** (the sharding case);
  full-grid reads keep the existing path unchanged.
- Only when the segment layout total equals `shape[-1]` (single LAM / non-
  overlapping LAMs). Multi-LAM-overlap, where eager `shape` and `_get_tuple`
  already disagree, falls back rather than guessing.
- Only when constituents can serve a grid-axis index array
  (`Cutout._pushdown_supported`, an I/O-free probe); else eager. `Select`/`Subset`
  are grid-transparent, so real (var-selected, date-adjusted) cutouts qualify.
- **Store-skip guard is conditioned on grid chunking** (`Cutout._grid_is_chunked`,
  via the forwarded `chunks` property): if the grid is **one chunk per field**
  (anemoi default), a shard touching *every* store gains nothing (whole field
  decompressed regardless) and the `oindex` gather is slower than a slice → fall
  back. If the grid **is chunked**, `oindex` reads only the touched chunks (a
  within-store saving), so push down even when every store is touched (e.g.
  boundary-spanning shards). No production dataset is grid-chunked today, so this
  branch is exercised by tests (`TestChunkedGridGuard`) until one exists.

Verified: `tests/test_grid_pushdown.py` asserts pushdown output is byte-identical
to the eager path (kill-switch off) for sub-slices within a LAM, across the
LAM/globe boundary, within the globe, single-point, and strided; and that the
factorized parts read **exactly the requested number of grid points** (vs the
full grid for a `slice(None)` read).

**When does it actually help? (measured on real S3 datasets)**

First, the **chunking reality**: a survey of 15 production datasets (ERA5 globals
o48→o1280, regional LAMs aemet/cerra/carra/meps/icon/… up to 3M points) shows
**every one keeps the whole grid in a single chunk** (`data.chunks[-1] == grid`).
This is the anemoi default: `create/recipe/output.py::get_chunking` defaults to
`{"dates": 1, "ensembles": 1}` and gives every other axis — *including variables
and the grid* — its full length. So **there is no within-chunk byte saving to be
had** on current data: reading any point of a store decompresses its whole field.

But the real win on a cutout is different and large: pushdown reads only the
**constituent stores the shard maps into**, skipping the others entirely. Real
cutout (`o48` "lam" inside `n320` globe, one chunk each, 1/16 grid shard,
nvars=101):

| shard lands in | eager | pushdown | speed / mem |
|---|---|---|---|
| the LAM (globe store **skipped**) | 443 ms / 234 MB | 17 ms / 6 MB | **25× / 42×** |
| the globe (its one chunk must be read) | 408 ms / 234 MB | 371 ms / 220 MB | 1.1× / 1.1× |
| spanning the boundary (both read) | 403 ms / 234 MB | 531 ms / 220 MB | **0.8×** (slower) |

Reading of the empirical result:

- **Big win when a shard avoids a constituent store** — the dominant case for
  cutout grid-sharding, where a shard usually lies in one region (the LAM, or the
  globe). Skipping the *other* store(s) avoids decompressing their full field.
- **Neutral when the shard must read a one-chunk store** (no within-chunk saving).
  For a *real* cutout (small LAM inset, large globe that supplies most of the
  output) there is still a **memory** win on globe-region shards, because pushdown
  returns the shard-sized array instead of materialising the full
  `concat(full LAMs + masked globe)` then slicing — that full array is grid-sized
  (GBs for a 6.6M-cell globe × ~100 vars), pushdown's is shard-sized. (The
  artificial `o48`-in-`n320` geometry above under-shows this because its globe
  contributes few points.)
- **Minor regression** for a shard that *spans* a LAM/globe boundary: it reads
  both one-chunk stores via `oindex` (a scattered gather) where eager does a
  contiguous slice + concat, ~20% slower with no I/O saving. There are only
  `n_lams` such boundaries, so this is a small fraction of shards; the kill-switch
  (`ANEMOI_DATASETS_READ_PARTS=0`) is the escape hatch if it ever bites.
- **Full-grid reads are unaffected** (`_is_full_grid_slice` → eager path), so
  non-sharded usage sees no change.

Bottom line: on today's single-chunk-per-field datasets the value is **(a) skipping
whole constituent stores a shard doesn't touch, and (b) not materialising the
full-grid output** — not within-chunk byte trimming. A real *byte* saving from
within-store sub-selection would additionally require chunking the grid axis at
creation time (set a `values` chunk in the recipe). Re-run
`docs/adr/two-step-read/benchmarks/grid_pushdown_benchmark.py` on a genuine cutout
pair for production numbers (measured runs in that folder's `RESULTS.md`).

**Union factorization (D2) — IMPLEMENTED.** Parts with *different* `grid_index`
on the *same* store are now **unioned**: `factorize` groups grid-index parts by
`(id(data), slices[1:])` (ignoring the specific points), reads the **sorted union**
of all grid points once via `oindex`, and maps each original part to its columns
within that union (`mapping[part] = (merged, row_offset, grid_cols)`;
`ReadBuffer.__getitem__` applies `data[offset:offset+rows][..., grid_cols]`). So a
`multi=` of cutouts that each request a *different* grid subset of a shared global
reads that global **once**. Verified by
`tests/test_grid_pushdown.py::TestMultiCutoutGridUnion` (two cutouts sharing a
globe, grid-subset → exactly one globe read, byte-identical per member).
Non-grid parts keep the original date-bounding-box merge unchanged.

### Problem 2 — `open_dataset(multi=dict(a=..., b=..., c=...))` — **IMPLEMENTED**

A top-level key (sibling of `cutout`, `grids`, `concat` in `misc.py::_open_dataset`)
that opens several **named** datasets sharing a single collect → factorize →
execute pipeline. `ds[n]` returns `{name: array}` — one array per member.

Files: `usage/gridded/multi.py` (`Multi`, `multi_factory`); the
`shared_zarr_opens()` context manager + open cache in `usage/store.py`; the
`multi` branch in `usage/misc.py::_open_dataset`. Tests: `tests/test_multi.py`.

As built:

- `Multi(Combined)` stores members as an ordered `{name: dataset}`. It implements
  `collect_read_parts` (concatenate every member's parts) and `read_from_buffer`
  (return `{name: member.read_from_buffer(...)}`), so the gate's `two_step_read`
  gathers **all** members' leaf reads into one factorize/execute pass.
- **Indexing has two forms** (`_member_indices`): a *broadcast* index (`ds[t]` /
  `ds[t0:t1]`) applies the same index to every member — the natural shared-date
  read; a *per-member dict* `ds[{name: index, …}]` indexes each member by its own
  index. The dict form is the explicit, unambiguous way to **grid-shard** a multi
  (members have different grids → no single shared "multi grid"); keys must be
  members (a subset is allowed). Both forms still share I/O through one factorize
  pass, so a store common to several members is read once even with *different*
  per-member shards (D2 union). A broadcast tuple carrying a grid component would
  resolve it per-member against each member's own shape — defined, but the dict
  form is preferred for that.
- `__getitem__` returns the same `{name: array}` dict on the eager/fallback path,
  so a member containing a non-rectangular wrapper (rolling average, …) still
  works — it just falls back per member with no cross-member sharing.
- Member metadata (`shape`, `variables`, `statistics`, `latitudes`, …) is exposed
  as dicts keyed by name; scalar/time metadata forwards to the first member.
  Members must share length; grids/variables may differ (compatibility checks are
  **off** by default — pass `check_compatibility=True` to enable).
- `collect_supporting_arrays` and `metadata_specific` are overridden to namespace
  each member's arrays/metadata under its `multi` key (so e.g. a cutout member's
  masks are collected as `(<member-name>, "lam_0", …)`), instead of `Combined`'s
  default which warns and keys by the member's own (usually `None`) name. This is
  what makes a `multi` of cutouts usable downstream (training fetches masks here).
- `missing` is the union of members' missing indices (`Combined` raises for it).

The load-bearing piece: `multi_factory` opens every member inside one
`shared_zarr_opens()` block. While active, `ZarrStore.from_name_or_path` returns
the **same** `ZarrStore` object (hence the same `zarr.Array`) for repeated opens
of the same name. Because `ReadPart` identity is `(id(self.data), slices)`,
`factorize` then merges reads of a store shared by several members into **one**
physical read. Without this cache, two `open_dataset(cutout=…)` calls open the
global store as two distinct array objects and do **not** dedup.

Why it is worth doing (and the measured benefit):

1. **API clarity** — even for one cutout, naming members
   (`multi=dict(lam=…, globe=…)`) beats the positional `cutout: [lam, globe]`
   "last item is the global" convention.
2. **Shared reads across multiple cutouts (confirmed)** — `tests/test_multi.py`
   builds two cutouts (different LAMs) over one shared global and asserts
   `factorize` collapses to **3 reads (lamA, lamB, globe) instead of 4**. The
   contrast test (`test_independent_opens_do_not_share`) shows separately-opened
   globes give 4. Generalises to **N+1 reads instead of 2N** for N cutouts over a
   shared global — the global (the expensive, full-resolution store) is read once.
3. **Grid-subset efficiency (future)** — the `multi=` container is the natural
   owner of the grid-index pushdown from Problem 1: every named cutout contributes
   grid-index `ReadPart`s, the container factorizes across all of them and reads
   only the union of needed chunks per store, once. (Not yet implemented — depends
   on extending `ReadPart` beyond rectangular slices; see workstream D.)

In short: a single simple cutout barely benefits from two-step, which made the
original motivation look thin. **Multiple cutouts (and, later, sharding) are where
it pays off**, and `multi=` is the structural enabler. This reframes the two-step
read from "make one read faster" to "share and minimise reads across a family of
related views" — the real justification for the design.

Known limitations / follow-ups:

- Sharing is keyed by the **exact name string** passed to `from_name_or_path`;
  two spellings of the same store (e.g. a name vs its resolved path) will not
  dedup. Fine for v1; could normalise later.
- `multi` takes no container-level subset options — put `select`/`start`/… inside
  each member spec. (`multi_factory` asserts on leftover kwargs.)

## Migration plan

> **Historical.** Phases 1–3 are done. Phase 4 has been **revised** — the eager
> path is *not* removed (see "Why the eager `__getitem__` path is permanent").

Phase 1 — new file, no behaviour change:
- Add `ReadPart`, `ReadBuffer`, `factorize`, `execute_parts`, `two_step_read` to
  `usage/read_parts.py`.
- Add abstract `collect_read_parts` + `read_from_buffer` to `Dataset` with `NotImplementedError`.
- Implement only in `GriddedZarr` + `TrajectoriesZarr` (leaf nodes).  All others raise.

Phase 2 — wrapper by wrapper:
- Implement in `Forwards` (default pass-through).
- Implement in `Select`, `Subset`, `Rename`, `Rescale`, `Masked`, `Thinning` (simple transforms).
- Implement in `Concat`, `Cutout`, `GivenAxis`, `Merge` (complex multi-dataset cases).
- Add `_two_step_enabled` flag per class; flip to `True` as each class is covered.

Phase 3 — wire into `__getitem__`:
- `Dataset.__getitem__` calls `two_step_read` when `_two_step_enabled` is True all the way
  to the leaves.
- Gate behind `ANEMOI_DATASETS_READ_PARTS=1` env var for opt-in testing.
- Add test suite comparing outputs of old vs new path on gridded, trajectories, tabular datasets.

Phase 4 — flip default, ~~remove old path~~ **keep old path permanently** (REVISED):
- The default is flipped: two-step is attempted first for every node, with
  automatic per-node fallback to eager when `collect_read_parts` returns `None`.
  `ANEMOI_DATASETS_READ_PARTS=0` is a real **kill-switch** to force eager (A4) —
  no longer vestigial.
- **Do NOT delete the eager `__getitem__` implementations.** They are the
  required execution path for all non-rectangular wrappers and the correctness
  oracle for the supported ones. "Remove old path" was the original intent and is
  now explicitly rejected — see "Why the eager `__getitem__` path is permanent".

Phase 5 — ~~parallelise~~ **reverted to sequential** (REVISED):
- A cross-part `ThreadPoolExecutor` was tried, then **removed**: it is GIL-bound
  on blosc decompress and measured ~35% *slower* for the few-large-chunks case;
  per-array S3 latency overlap already lives in zarr/anemoi `getitems`. See
  "Execution is sequential". `execute_parts` takes only `parts`.
- The real I/O win came instead from grid-subset pushdown skipping whole
  constituent stores (see "Problem 1"), not from cross-part threads.

## Files to create / modify

| Action | File |
|--------|------|
| Create | `src/anemoi/datasets/usage/read_parts.py` — `ReadPart`, `ReadBuffer`, `factorize`, `execute_parts`, `two_step_read`, `_log_parts` |
| Modify | `src/anemoi/datasets/usage/dataset.py` — add abstract methods to `Dataset` |
| Modify | `src/anemoi/datasets/usage/forwards.py` — default impl in `Forwards` |
| Modify | `src/anemoi/datasets/usage/store.py` — base `ZarrStore` hooks |
| Modify | `src/anemoi/datasets/usage/gridded/store.py` — leaf impl for `GriddedZarr` |
| Modify | `src/anemoi/datasets/usage/trajectories/store.py` — leaf impl for `TrajectoriesZarr` |
| Modify | `src/anemoi/datasets/usage/gridded/select.py` | 
| Modify | `src/anemoi/datasets/usage/gridded/subset.py` |
| Modify | `src/anemoi/datasets/usage/gridded/grids.py` — `Concat`, `Cutout` |
| Modify | `src/anemoi/datasets/usage/gridded/merge.py` |
| Modify | `src/anemoi/datasets/usage/gridded/masked.py` |
| Create | `tests/usage/test_read_parts.py` |

## Out of scope (v1)

- Tabular datasets (`TabularZarr` / `WindowView`) — different access pattern, tackle separately.
- Fancy index support beyond slices (boolean masks, non-contiguous integer lists) — these stay
  on the `expand_list_indexing` path.
- Parallelisation — Phase 5.
- Prefetching / lookahead — separate ADR.
