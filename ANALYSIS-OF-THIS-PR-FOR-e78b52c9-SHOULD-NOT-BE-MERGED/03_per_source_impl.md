# 03 — Per‑source implementation

One section per source that the branch changes or adds. Each follows the
same template:

- **Accepts** — which typed arguments it handles.
- **Registered overloads** — which `@for_*` decorators it uses.
- **Shared helpers** — what it reuses from the new machinery or from the
  refactored MARS / accumulate packages.
- **Edge cases** — per‑source gotchas encoded in the code.
- **File map** — the files and line ranges that implement the above.

## 1. MarsSource

### 1.1 File layout

Was a single `create/sources/mars.py` (~522 lines). Now a package:

```
create/sources/mars/
  __init__.py          # re-exports source + shared primitives
  source.py            # MarsSource dispatch class
  retrieval.py         # MARS_KEYS, RequestFilter, factorise/compress/fire primitives
```

The split keeps the source class focused on dispatch; the reusable MARS
primitives live in `retrieval.py` so that `hindcasts.py`, `recentre.py`,
`from_trajectories.py` and the `fdb.py` adapter can import them directly
without depending on the source class
([mars/__init__.py](../src/anemoi/datasets/create/sources/mars/__init__.py)).

### 1.2 Accepts & registered overloads

All four typed arguments:

- `@for_valid_dates  → ValidDates`
- `@for_forecast_dates → ForecastDates`
- `@for_intervals → Intervals`
- `@for_forecast_intervals → ForecastIntervals`

See [sources/mars/source.py:54-144](../src/anemoi/datasets/create/sources/mars/source.py#L54-L144).

### 1.3 Shared helpers

- `execute_mars_request(context, dates, *requests, …)` — the full
  expansion/factorisation/fire pipeline for validity‑date requests. Uses
  `factorise_requests` so steps are expanded and merged via
  earthkit `Availability`.
- `fire_prebuilt_requests(context, requests)` — for pre‑built request
  lists (every path except `@for_valid_dates`). Normalises grid keys,
  compresses, validates params, fires via `from_source("mars"|"cds", …)`.
- `RequestFilter` — parses `date: "????-??-01"` + `time: […]` out of a
  request and returns a cleaned request + compiled regex/frozenset. Used
  by the `@for_valid_dates` path through `_expand_mars_request`.
- `Intervals.adjust_request(interval, request)` — the shared helper that
  stamps `date`/`time`/`step` from a `SignedInterval` onto a copy of a
  request. `MarsSource` calls it from both `@for_intervals` and
  `@for_forecast_intervals`.

### 1.4 Edge cases

- **Empty `ValidDates` (repeat‑dates constant mode).** When `dates.dates`
  is empty, the source detects it at
  [source.py:57-66](../src/anemoi/datasets/create/sources/mars/source.py#L57-L66),
  assumes the request already carries its own `date`/`time` (from
  `DateMapperConstant(date=None)` in `repeated_dates.py`), stringifies any
  `datetime.date` in the request, and routes through
  `fire_prebuilt_requests`. The normal `execute_mars_request` path
  wouldn't work here — `factorise_requests` needs at least one validity
  date to expand.
- **Filters on forecast/interval paths.** `_reject_filters` at
  [source.py:29-41](../src/anemoi/datasets/create/sources/mars/source.py#L29-L41) guards
  the three non‑valid‑dates overloads against wildcard `date` filters
  ("only valid in the validity‑date path"). A `ValueError` is raised at
  build time so configs failing silently against `fire_prebuilt_requests`
  would have produced incoherent output.
- **`stream: scda` auto‑selection.** In `_expand_mars_request`
  ([retrieval.py:306-308](../src/anemoi/datasets/create/sources/mars/retrieval.py#L306-L308))
  the ECMWF operational stream is switched from `oper` to `scda` for 06
  and 18 UTC runs. This happens in the validity‑date expansion path; the
  prebuilt path currently *does not* do it (comment in
  [accumulate/source.py:39-45](../src/anemoi/datasets/create/sources/accumulate/source.py#L39-L45)
  flags it as a TODO).
- **Assertions on `interval.base`.** Both `@for_intervals` and
  `@for_forecast_intervals` assert `interval.base is not None` — only
  `grib_index` is allowed to produce base‑less intervals. A `MarsSource`
  that received one would be a bug upstream.
- **`param: False/None/True` guard.** `_validate_params` at
  [retrieval.py:328-347](../src/anemoi/datasets/create/sources/mars/retrieval.py#L328-L347)
  rejects YAML boolean artefacts (`param: no` parses as `False`) at
  build time with a pointer to quoting.

### 1.5 File map

| Concern | File | Lines |
| --- | --- | --- |
| `MarsSource` class, four overloads | `mars/source.py` | 46‑144 |
| `_reject_filters` | `mars/source.py` | 29‑41 |
| `RequestFilter` | `mars/retrieval.py` | 151‑253 |
| `_expand_mars_request` | `mars/retrieval.py` | 256‑312 |
| `factorise_requests` | `mars/retrieval.py` | 383‑416 |
| `compress_prebuilt_requests` | `mars/retrieval.py` | 419‑444 |
| `fire_prebuilt_requests` | `mars/retrieval.py` | 447‑463 |
| `execute_mars_request` | `mars/retrieval.py` | 466‑499 |
| `MARS_KEYS` | `mars/retrieval.py` | 33‑100 |

## 2. FdbSource

### 2.1 Accepts & registered overloads

- `@for_valid_dates → ValidDates`
- `@for_intervals → Intervals`

`@for_forecast_dates` is **not** registered, so a trajectory recipe
cannot route through `fdb:` today. This is a deliberate minimum — FDB
needs the same forecast dispatch, the TODO comment says so; the branch
does not do it.

See [sources/fdb.py:87-109](../src/anemoi/datasets/create/sources/fdb.py#L87-L109).

### 2.2 Shared helpers

Imports `compress_prebuilt_requests` from the new MARS package
(previously `factorise_requests` from the now‑removed `accumulate`
re‑export). The `Intervals.adjust_request` helper is reused verbatim.

### 2.3 Edge cases

- **Empty requests fallback.** `_execute_requests(requests or
  [self.request])` ([fdb.py:120-122](../src/anemoi/datasets/create/sources/fdb.py#L120-L122))
  — symmetric with the MarsSource empty‑dates path.
- **Base‑less intervals rejected** with the same assertion as MARS.
- **Grid / flavour post‑processing** is kept per‑instance inside
  `_execute_requests`, so both overloads share the same grid rewrite and
  flavour mapping.

### 2.4 File map

| Concern | File | Lines |
| --- | --- | --- |
| `FdbSource`, two overloads | `sources/fdb.py` | 32‑134 |
| `_execute_requests` | `sources/fdb.py` | 111‑134 |
| `_time_request_keys` | `sources/fdb.py` | 137‑151 |

## 3. GribIndexSource

### 3.1 Accepts & registered overloads

- `@for_valid_dates → ValidDates`
- `@for_intervals → Intervals`

No forecast dispatch (grib‑index is valid‑time indexed by construction,
no basetime exists).

### 3.2 Edge cases

- **Intervals must be base‑less.** The overload asserts `interval.base is
  None` at [grib_index.py:635-638](../src/anemoi/datasets/create/sources/grib_index.py#L635-L638);
  this is the one source where a non‑`None` base is the bug.
- **No `adjust_request` call.** Instead, `interval.max` is used as the
  valid time and `interval.end − interval.start` as the step length.
- **Flavour + grid post‑processing** identical to FDB.
- **Factorisation** uses the in‑module `factorise(…)` helper
  ([grib_index.py:669-684](../src/anemoi/datasets/create/sources/grib_index.py#L669-L684))
  — merges `(dates, request)` pairs that have identical request dicts.
  Independent from MARS's `factorise_requests`.

### 3.3 File map

| Concern | File | Lines |
| --- | --- | --- |
| `GribIndexSource`, two overloads | `sources/grib_index.py` | 582‑666 |
| `factorise` (request dedup) | `sources/grib_index.py` | 669‑684 |
| `GribIndex` (sqlite store) | `sources/grib_index.py` | 43‑580 |

## 4. AccumulateSource

### 4.1 File layout

Was a single `create/sources/accumulate.py` (~546 lines). Now a package:

```
create/sources/accumulate/
  __init__.py
  source.py              # AccumulateSource with @for_valid_dates + @for_forecast_dates
  covering.py            # Covering ABC, AutoCovering, ForecastCovering, covering_factory
  accumulator.py         # Accumulator + Logs (unchanged algorithmically but now exports basetime support)
  writers.py             # GRIB1/GRIB2 writers: with_valid_time / forecast_field
  field_to_interval.py   # reads (startStep, endStep) from a field → SignedInterval
  interval_generators.py # availability description (SearchableIntervalGenerator, Cycle, …)
  covering_intervals.py  # Dijkstra search over an IntervalGenerator
```

### 4.2 Accepts & registered overloads

- `@for_valid_dates → ValidDates` — the archive branch; uses
  `AutoCovering` to search.
- `@for_forecast_dates → ForecastDates` — the trajectory branch; uses
  `ForecastCovering` with the caller‑imposed basetime.

Note the asymmetry: no `@for_intervals`, no `@for_forecast_intervals`. The
source *produces* `Intervals` / `ForecastIntervals` and hands them to the
inner source; it does not consume them.

### 4.3 Shared pipeline

Both branches converge on `_accumulate_fields(source_object, intervals,
targets, coverages)` at
[sources/accumulate/source.py:166-230](../src/anemoi/datasets/create/sources/accumulate/source.py#L166-L230).
The `target` is a tuple `(valid_date, basetime)`; `basetime=None` on the
archive branch. Each `Accumulator` is keyed by `(*target, group_key)` and
carries its basetime all the way to the writer, which picks between
`write_accumulated_field_with_valid_time` (archive) and
`write_accumulated_forecast_field` (trajectory) based on whether
`accumulator.basetime is None`
([accumulator.py:121-137](../src/anemoi/datasets/create/sources/accumulate/accumulator.py#L121-L137)).

### 4.4 `Covering` layer (new)

Two concrete strategies under the `Covering` ABC
([sources/accumulate/covering.py:45-72](../src/anemoi/datasets/create/sources/accumulate/covering.py#L45-L72)):

- **`AutoCovering`** ([:75-98](../src/anemoi/datasets/create/sources/accumulate/covering.py#L75-L98))
  wraps an `IntervalGenerator` and its Dijkstra search. Raises
  `NotImplementedError` if a non‑`None` `basetime` is passed — archive
  search cannot use an externally imposed basetime.
- **`ForecastCovering`** ([:104-195](../src/anemoi/datasets/create/sources/accumulate/covering.py#L104-L195))
  takes the `period` and `accumulation ∈ {from-zero, from-previous-step}`
  flag. No search: it directly emits the 1‑ or 2‑interval decomposition.
  Window straddling or non‑integer‑hour offsets are rejected with clear
  messages.

`covering_factory(config, source_name=None, source=None)`
([:198-254](../src/anemoi/datasets/create/sources/accumulate/covering.py#L198-L254))
dispatches the recipe `covering:` value on its first key:
- `{auto: …}` → `AutoCovering`,
- `{cycle: …}` → `NotImplementedError`,
- `{forecast: …}` → hard error explaining the forecast branch is implicit,
- anything else (legacy list/dict/string) → treated as `auto`.

### 4.5 Edge cases

- **Auto vs forecast gating.** The archive branch requires `covering:` (or
  legacy `availability:`) and raises if it is missing
  ([source.py:237-243](../src/anemoi/datasets/create/sources/accumulate/source.py#L237-L243)).
  The forecast branch requires `accumulation:` and raises if it is
  missing ([source.py:280-284](../src/anemoi/datasets/create/sources/accumulate/source.py#L280-L284)).
  A forecast recipe that sets `covering:` gets a `DEBUG` log saying it is
  ignored; the source does not fail.
- **Legacy `availability:` deprecation.** The constructor accepts
  `availability=` for one release as a back‑compat alias that emits a
  `DeprecationWarning` and rewrites to `{"auto": value}`
  ([source.py:87-93](../src/anemoi/datasets/create/sources/accumulate/source.py#L87-L93)).
  The full migration path to `covering:` is in
  `commands/recipe/migrate.py:_fix_accumulate_availability`.
- **MARS defaults.** When the inner source is `mars:` without `type` or
  `levtype`, the source logs a warning and injects `type: fc` / `levtype:
  sfc` ([source.py:107-120](../src/anemoi/datasets/create/sources/accumulate/source.py#L107-L120))
  — so the most common mistake (forgetting `type: fc` on an accumulation)
  still works.
- **Hashed caching of the inner source object.** `_create_source_object`
  hashes `(period, source, *extra_hash_parts)` and uses the hash as a
  registry key. The forecast branch passes the `accumulation` flag as an
  extra hash part so the two branches don't share a cached source.
- **Base‑less interval matching.** `Accumulator.compute` matches intervals
  on `(min, max, base)` for forecast and on `(start, end)` alone when
  `base=None` ([accumulator.py:67-73](../src/anemoi/datasets/create/sources/accumulate/accumulator.py#L67-L73))
  — so grib‑index intervals can still drive an accumulation (though the
  source itself only offers the archive branch).

### 4.6 File map

| Concern | File | Lines |
| --- | --- | --- |
| `AccumulateSource`, two overloads, shared pipeline | `accumulate/source.py` | 64‑307 |
| `Covering`, `AutoCovering`, `ForecastCovering`, `covering_factory` | `accumulate/covering.py` | 45‑254 |
| `Accumulator`, basetime‑aware match and write | `accumulate/accumulator.py` | 26‑155 |
| GRIB writers for valid‑time and forecast flavours | `accumulate/writers.py` | 19‑132 |
| Dijkstra search | `accumulate/covering_intervals.py` | 40‑127 |

## 5. HindcastsSource

### 5.1 Accepts & registered overloads

`HindcastsSource` inherits from `LegacySource`, not `Source` — it does not
use the dispatch mixin. It is invoked with a plain list of dates and
resolves them against the `HindcastsDates` provider's mapping.

### 5.2 Change on the branch

The old `mars(context, dates, *requests, date_key="hdate", …)` wrapper is
gone. Instead, the source builds its request list directly and calls
`fire_prebuilt_requests` from the MARS retrieval module
([sources/hindcasts.py:15,89](../src/anemoi/datasets/create/sources/hindcasts.py#L15-L89)).
Date / time formatting switched to `%Y%m%d` / `%H%M` (was `%Y-%m-%d` /
`%H`).

### 5.3 File map

| Concern | File | Lines |
| --- | --- | --- |
| `HindcastsSource._execute` | `sources/hindcasts.py` | 41‑89 |

## 6. RecentreSource

### 6.1 Change on the branch

The `mars(...)` wrapper import is replaced with `execute_mars_request`
from the MARS retrieval module
([sources/recentre.py:17,105](../src/anemoi/datasets/create/sources/recentre.py#L17-L105)).
No other change; `RecentreSource` still derives from `LegacySource` and
does not participate in dispatch.

### 6.2 File map

| Concern | File | Lines |
| --- | --- | --- |
| `load_if_needed` (calls `execute_mars_request`) | `sources/recentre.py` | 86‑106 |
| `RecentreSource._execute` | `sources/recentre.py` | 109‑148 |

## 7. RepeatedDates / DateMapperConstant

Not a source — an *input transformer* that wraps another source in the
action tree. The branch makes `DateMapperConstant` support
`date: null`, which is the mechanism sources need to broadcast a single
prebuilt request across all validity times.

### 7.1 Behaviour

- `date: null` in the recipe → `DateMapperConstant.date is None`
  ([input/repeated_dates.py:246-259](../src/anemoi/datasets/create/input/repeated_dates.py#L246-L259)).
- On transform, it yields one `(empty_group, original_group)` pair
  ([:278-284](../src/anemoi/datasets/create/input/repeated_dates.py#L278-L284)).
- The empty `GroupOfDates` becomes `ValidDates([])` at the dispatch
  entry, so sources must handle the empty list.
- `MarsSource` handles it via its empty‑dates path (§1.4).
- Other sources (`FdbSource`, …) are expected to handle it too; today
  `FdbSource._execute_requests` falls back to `self.request` if the list
  is empty.

### 7.2 Eager validation

Non‑None values are parsed at construction time; invalid values raise a
`ValueError` pointing to the offending field rather than failing at fetch
time ([repeated_dates.py:253-259](../src/anemoi/datasets/create/input/repeated_dates.py#L253-L259)).

### 7.3 File map

| Concern | File | Lines |
| --- | --- | --- |
| `DateMapperConstant` | `input/repeated_dates.py` | 231‑291 |
| `DateMapperClosest`, `DateMapperClimatology` | `input/repeated_dates.py` | 56‑228 |

## 8. FromTrajectoriesSource (new)

### 8.1 What it does

Turns a *gridded* recipe (dates‑driven) into a forecast‑dates request by
picking a `(basetime, step)` pair for each validity time. The
`bases` fnmatch pattern acts on the basetime; `steps` is a MARS‑style
step spec parsed with `expand_to_by` from the MARS package.

### 8.2 Accepts & registered overloads

This source does **not** use `@for_*` dispatch. Its `execute(dates)`
accepts any input with a `.dates` attribute (or iterable of datetimes),
builds a `ForecastDates` from it, and delegates to the inner source
`self.inner.execute(forecast_dates)` — the inner source must have its own
`@for_forecast_dates` overload.

See [sources/from_trajectories.py:147-162](../src/anemoi/datasets/create/sources/from_trajectories.py#L147-L162).

### 8.3 Edge cases

- **Single‑key source dict** required; a clear `ValueError` otherwise
  ([from_trajectories.py:91-95](../src/anemoi/datasets/create/sources/from_trajectories.py#L91-L95)).
- **Smallest step first.** `_pick_basetime` iterates steps in config
  order and returns the first basetime matching `bases_pattern`; this
  effectively prioritises the shortest lead time
  ([from_trajectories.py:113-132](../src/anemoi/datasets/create/sources/from_trajectories.py#L113-L132)).
- **No match** for a given validity time raises an informative
  `ValueError` naming the step and pattern — so misconfigured recipes
  fail at build time, not with a silent empty dataset.

### 8.4 File map

| Concern | File | Lines |
| --- | --- | --- |
| `FromTrajectoriesSource` | `sources/from_trajectories.py` | 52‑162 |
| `_pick_basetime`, `_basetime_matches`, `_as_forecast_dates` | `sources/from_trajectories.py` | 101‑145 |

## 9. Cross‑source assertions

The branch encodes the interval taxonomy with per‑source assertions that
must agree:

| Source        | `@for_intervals` asserts        | `@for_forecast_intervals` asserts |
| ------------- | ------------------------------- | --------------------------------- |
| MarsSource    | `interval.base is not None`     | `interval.base is not None`       |
| FdbSource     | `interval.base is not None`     | *(not registered)*                |
| GribIndexSource | `interval.base is None`       | *(not registered)*                |

And the covering layer plays the matching role:

| Covering          | emits `base=…`?        |
| ----------------- | ---------------------- |
| `AutoCovering`    | `base = interval.base` from the generator (typically not `None`) |
| `ForecastCovering` | `base = caller-imposed basetime` always non‑`None` |
| grib‑index search | not routed through `Covering`; emits `base=None` |

These assertions are the branch's way of keeping the source/covering
contract explicit without needing a separate type per interval flavour.

## Pointers

- `src/anemoi/datasets/create/sources/mars/` (whole package).
- `src/anemoi/datasets/create/sources/fdb.py`.
- `src/anemoi/datasets/create/sources/grib_index.py`.
- `src/anemoi/datasets/create/sources/accumulate/` (whole package).
- `src/anemoi/datasets/create/sources/hindcasts.py`.
- `src/anemoi/datasets/create/sources/recentre.py`.
- `src/anemoi/datasets/create/input/repeated_dates.py`.
- `src/anemoi/datasets/create/sources/from_trajectories.py`.
- `src/anemoi/datasets/create/arguments.py` — argument type contracts.
- `src/anemoi/datasets/create/intervals.py` — `SignedInterval`.
