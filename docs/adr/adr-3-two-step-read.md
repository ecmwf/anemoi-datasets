# Two-step read: collect parts, factorize, execute

## Status

Proposed - 2026-06-04

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

### `ReadCache`

```python
class ReadCache:
    raw: dict[ReadPart, NDArray]      # factorized part → read array
    mapping: dict[ReadPart, tuple[ReadPart, int]]
    # original part → (merged_part, date_offset_within_merged)

    def __getitem__(self, part: ReadPart) -> NDArray:
        merged, offset = self.mapping.get(part, (part, 0))
        data = self.raw[merged]
        length = part.slices[0].stop - part.slices[0].start
        return data[offset : offset + length]
```

### New methods on `Dataset` ABC

```python
class Dataset(ABC):

    def collect_read_parts(self, index: FullIndex) -> list[ReadPart]:
        """Phase 1: return all zarr reads required for this index, without executing them."""
        raise NotImplementedError

    def read_from_cache(self, index: FullIndex, cache: ReadCache) -> NDArray:
        """Phase 3: reconstruct result using pre-fetched cache."""
        raise NotImplementedError
```

`collect_read_parts` + `read_from_cache` are mirror images of `__getitem__`: same index
transformations, but `collect_read_parts` accumulates leaf reads while `read_from_cache`
assembles the result from cached data.

## Per-class implementation sketch

### `Forwards` (default for all transparent wrappers)

```python
def collect_read_parts(self, n):
    return self.forward.collect_read_parts(n)

def read_from_cache(self, n, cache):
    return self.forward.read_from_cache(n, cache)
```

### `GriddedZarr` / `TrajectoriesZarr` — leaf nodes

```python
def collect_read_parts(self, n):
    slices, squeeze = index_to_slices(n, self.data.shape)
    return [ReadPart(self.path, slices, squeeze)]

def read_from_cache(self, n, cache):
    slices, squeeze = index_to_slices(n, self.data.shape)
    part = ReadPart(self.path, slices, squeeze)
    data = cache[part]
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

def read_from_cache(self, n, cache):
    # exact mirror of __getitem__ but recursive call uses read_from_cache
    if isinstance(n, tuple):
        index, changes = index_to_slices(n, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.dataset.read_from_cache(index, cache)
        result = result[:, self.indices][:, previous]
        return apply_index_to_slices_changes(result, changes)
    row = self.dataset.read_from_cache(n, cache)
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

def read_from_cache(self, n, cache):
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

def read_from_cache(self, n, cache):
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

def read_from_cache(self, index, cache):
    # same reads from cache, apply masks, concatenate — mirror of _get_tuple
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
    cache = {}
    for part in factorized:
        store = _open_zarr(part.path)          # cached open, not re-opened each time
        cache[part] = store["data"][part.slices]
    return cache
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
    cache = ReadCache(raw=raw, mapping=mapping)
    return dataset.read_from_cache(index, cache)
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

## Migration plan

Phase 1 — new file, no behaviour change:
- Add `ReadPart`, `ReadCache`, `factorize`, `execute_parts`, `two_step_read` to
  `usage/read_parts.py`.
- Add abstract `collect_read_parts` + `read_from_cache` to `Dataset` with `NotImplementedError`.
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

Phase 4 — flip default, remove old path:
- Remove the `ANEMOI_DATASETS_READ_PARTS` gate.
- Remove `_two_step_enabled` flags.
- Old `__getitem__` implementations stay as reference until all tests pass; then delete.

Phase 5 — parallelise:
- Replace sequential loop in `execute_parts` with `ThreadPoolExecutor`.
- Benchmark on S3 vs local zarr.

## Files to create / modify

| Action | File |
|--------|------|
| Create | `src/anemoi/datasets/usage/read_parts.py` — `ReadPart`, `ReadCache`, `factorize`, `execute_parts`, `two_step_read`, `_log_parts` |
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
