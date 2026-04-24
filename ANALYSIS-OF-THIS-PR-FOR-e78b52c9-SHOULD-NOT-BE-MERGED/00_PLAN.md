# Analysis plan — `feat/trajectories` branch of `anemoi-datasets`

Scope. Analyse the diff between `main` and `feat/trajectories` and produce a
set of companion markdown files that describe, for the current state of the
branch:

1. what a recipe author / user sees (surface),
2. what the implementation does internally (concepts + architecture),
3. how each source was adapted to the new machinery,
4. the pros and cons of the architecture as a whole,
5. the pros and cons of each source's implementation.

The goal is a base from which developer documentation, onboarding slides, and
machine-readable context for coding agents can be produced without further
reverse engineering. No ADR content or `skills/*.md` is used — everything is
drawn from the code of the `feat/trajectories` branch.

The branch adds one new dataset layout (`trajectories`) next to the existing
`gridded` and `tabular` layouts, plus the plumbing required to feed it from
the existing MARS / FDB / accumulate / grib‑index sources, plus a new
`from-trajectories` source that lets a normal gridded recipe consume
`(basetime, step)` pairs. The diff also rewrites the MARS and accumulate
sources into small packages, and introduces a typed‑argument + decorator
dispatch system shared by every source that can be asked for either
instants, forecast pairs, or accumulation intervals.

## Files to produce

All files live in [`ANALYSIS/`](./). Each is self‑contained (can be read on
its own) but numbered so the whole folder reads top‑to‑bottom.

### `00_PLAN.md` (this file)
Master plan + short map of the branch (below).

### `01_usage_recipe_author.md` — user view
What a human writing a YAML recipe has to know.
- New top‑level recipe keys: `base_dates:`, `steps:`, `output.layout:
  trajectories`, and why `dates:` is rejected for trajectories and
  `base_dates:` for everything else.
- `output.layout: trajectories` implicit on‑disk shape `(base_dates, variables,
  ensembles, steps, cells)` and the chunking keys it accepts.
- The `from-trajectories:` source in a *gridded* recipe — `bases:` fnmatch
  pattern, `steps:` MARS‑style spec, and the inner `source:` it wraps.
- MARS wildcard date/time filter (`date: "????-??-01"`, `time: [0, 1200]`)
  exercised from `tests/create/mars-date-filter.yaml` and
  `tests/create/mars-time-filter.yaml`.
- `accumulate:` renamed discriminator: `availability:` → `covering: { auto: … }`;
  new `accumulation: from-zero | from-previous-step` flag required in the
  trajectory branch; `covering: { forecast: … }` explicitly rejected.
- `repeat-dates` with `mode: constant, date: null` — the supported "no
  date at all" path (forcings that share a prebuilt MARS request).
- Behaviour changes on the *reader* side that recipe authors will see:
  `output.order_by` is deprecated (ignored if it matches the fixed default,
  rejected otherwise); `output.flatten_grid` is gone, flattening is always on.
- Recipe migration (`anemoi datasets recipe migrate`) — the new
  `_fix_accumulate_availability` rewrites legacy recipes.

### `02_implementation_concepts.md` — architecture
Concept‑level description of everything new that sources / creators sit on
top of.
- **Typed arguments** ([`create/arguments.py`](../src/anemoi/datasets/create/arguments.py)).
  `Argument` ABC; the 2×2 matrix `ValidDates` /
  `ForecastDates` / `Intervals` / `ForecastIntervals`; subclassing
  (`Intervals ⊂ ValidDates`, `ForecastIntervals ⊂ ForecastDates`) so that
  MRO fallback gives sources one overload per *kind* of input rather than
  one per cross‑product; conversion helpers (`as_intervals`, `with_basetime`,
  `as_forecast_intervals`, `adjust_request`).
- **SignedInterval** ([`create/intervals.py`](../src/anemoi/datasets/create/intervals.py)).
  The `(start, end, base)` triple with `sign` and `valid_time` helpers,
  `__neg__`, hashing and rich `__repr__`. `base=None` is the marker that the
  interval came from a valid‑time‑indexed backend (grib‑index).
- **Dispatch decorator** ([`create/dispatch.py`](../src/anemoi/datasets/create/dispatch.py)).
  `@for_valid_dates`, `@for_forecast_dates`, `@for_intervals`,
  `@for_forecast_intervals`, accumulation into a `_MultiDispatch`
  descriptor via `sys._getframe(1)` class‑body inspection; the
  `DispatchedSource` mixin and the MRO/backwards‑compat wrapping of
  `list[datetime]` and `GroupOfDates` into `ValidDates`.
- **Context of source** ([`create/input/context.py`](../src/anemoi/datasets/create/input/context.py)
  and per‑layout subclasses: [`gridded/context.py`](../src/anemoi/datasets/create/gridded/context.py),
  [`trajectories/context.py`](../src/anemoi/datasets/create/trajectories/context.py)).
  Role of the context object: it carries the layout's fixed `order_by`, the
  remapping, origin tracking, and how results are built
  (`create_result`). Explain the ownership move: `order_by` and
  `flatten_grid` used to live on `Output` / `GriddedResult`; now `order_by`
  is a class attribute on the context and `flatten_grid` is gone.
- **Layout split**: `gridded` / `trajectories` / `tabular`.
  `GriddedCreator` is now an ABC with two concrete subclasses
  (`SimpleGriddedCreator`, `TrajectoryGriddedCreator`); `GriddedResult` is the
  shared implementation with `SimpleGriddedResult` and
  `TrajectoryGriddedResult` concrete subclasses; the `_metadata_date_range`
  and `_metadata_dates` hooks let trajectories expose
  base‑date metadata while gridded keeps the original single `dates` axis.
- **Dates providers and groups** ([`dates/__init__.py`](../src/anemoi/datasets/dates/__init__.py),
  [`dates/groups.py`](../src/anemoi/datasets/dates/groups.py)).
  `TrajectoryDates` (Cartesian product of basetimes × steps, with
  `factorise()`); `TrajectoryGroups` (yields `ForecastDates`, not
  `GroupOfDates`); `GrouperByKey._key` now unwraps `(basetime, step)` tuples;
  `GroupOfDates` accepts tuples opaquely.
- **Composite + Pipe** ([`create/composite.py`](../src/anemoi/datasets/create/composite.py)).
  The programmatic composition ABC; how it differs from the YAML
  `input: pipe:` action tree and from anemoi‑transform `Filter`.
- **Zarr on‑disk layout**.
  Arrays for each layout (`dates` vs `base_dates`/`steps`), dimension names,
  `ensemble_dimension`/`step_dimension` metadata, `Dataset.dates` fallback to
  `base_dates`, deprecated `flatten_grid` attribute removed.
- **Reader side** ([`usage/trajectories/store.py`](../src/anemoi/datasets/usage/trajectories/store.py),
  [`usage/trajectories/subset.py`](../src/anemoi/datasets/usage/trajectories/subset.py)).
  `TrajectoriesZarr`; `StepSubset`, `SingleStepView`, `Subset`; envelope
  logic for `start`/`end` filtering; why `dates` is aliased to `base_dates`;
  `usage_factory_load` per‑package hook; new forwarded properties on
  `Forwards` (`base_dates`, `base_frequency`, …). `inspect` CLI picks up the
  trajectories path automatically via `steps` attribute.

### `03_per_source_impl.md` — per‑source adoption
One section per source, each following the same template: *what it accepts*,
*which dispatch overloads it registers*, *what internal helpers it shares or
reuses*, *edge cases it has to handle*.
- **MarsSource** (package split under [`sources/mars/`](../src/anemoi/datasets/create/sources/mars/)).
  `source.py` has the four `@for_*` overloads;
  `retrieval.py` holds the reusable `RequestFilter` and the
  `factorise_requests` / `fire_prebuilt_requests` /
  `execute_mars_request` primitives. Empty‑dates branch for
  `repeat-dates mode=constant, date=null`; `_reject_filters` bar for
  forecast/accum paths; `stream: scda` auto‑selection; `adjust_request`
  delegation.
- **FdbSource** ([`sources/fdb.py`](../src/anemoi/datasets/create/sources/fdb.py)).
  `@for_valid_dates` + `@for_intervals` using the same `Intervals.adjust_request`
  as MARS, but now imports `compress_prebuilt_requests` from the MARS
  package instead of the removed `factorise_requests` path; the flavour and
  grid hooks are still per‑instance.
- **GribIndexSource** ([`sources/grib_index.py`](../src/anemoi/datasets/create/sources/grib_index.py)).
  `@for_valid_dates` + `@for_intervals`; intervals must have `base=None`
  (flat valid‑time index); does not go through `adjust_request`.
- **AccumulateSource** (package [`sources/accumulate/`](../src/anemoi/datasets/create/sources/accumulate/)).
  Split into `source.py`, `accumulator.py`, `covering.py`, `writers.py`,
  `field_to_interval.py`, `interval_generators.py`, `covering_intervals.py`.
  `@for_valid_dates` builds `Intervals` via `AutoCovering`;
  `@for_forecast_dates` builds `ForecastIntervals` via `ForecastCovering`
  (trivial 1‑ or 2‑interval decomposition driven by `accumulation`). Shared
  `_accumulate_fields` over both branches; writer chosen by whether the
  accumulator has a `basetime`.
- **HindcastsSource** and **recentre** ([`sources/hindcasts.py`](../src/anemoi/datasets/create/sources/hindcasts.py),
  [`sources/recentre.py`](../src/anemoi/datasets/create/sources/recentre.py)).
  Call `fire_prebuilt_requests` / `execute_mars_request` directly now that
  those are stable entry points in the MARS package.
- **RepeatedDates / DateMapperConstant** ([`create/input/repeated_dates.py`](../src/anemoi/datasets/create/input/repeated_dates.py)).
  `date=None` produces an empty `GroupOfDates`; MarsSource handles that by
  firing the prebuilt request as‑is.
- **FromTrajectoriesSource** ([`sources/from_trajectories.py`](../src/anemoi/datasets/create/sources/from_trajectories.py)).
  New source that rewrites an incoming `ValidDates` into a `ForecastDates`
  by picking a `(basetime, step)` pair matching a `bases` fnmatch pattern;
  the inner source is invoked via its own dispatch.

### `04_global_pros_cons.md`
Pros/cons of the *global* architectural choices: typed arguments, frame‑
inspection dispatch, covering/availability split, context‑as‑layout‑owner,
layout‑specific creator+result subclasses, Zarr 5‑D layout, coexistence of
`dates`/`base_dates`, deprecation policy for `order_by` and `availability`.
Each bullet explicitly argues a benefit and a cost, with pointers to the
file where the tradeoff manifests.

### `05_per_source_pros_cons.md`
Fine‑grained pros/cons for each adapted source. Where the new machinery
works well, where a source has to reach outside it (MARS empty‑dates path,
accumulate's `accumulation` flag mandatory only in the forecast branch,
grib‑index's refusal to go through `adjust_request`, FDB's duplication with
MARS, hindcasts bypassing dispatch entirely via `fire_prebuilt_requests`,
from‑trajectories not itself a `DispatchedSource`).

## High‑level map of the branch

Only two commits diverge from `main` (`git log main..feat/trajectories`):

| Commit     | Summary |
| ---------- | ------- |
| `70e63741` | added trajectories (the real change) |
| `0b8dcea5` | adr draft (`docs/adr/adr-2-trajectories.md`) |

Diff shape (largest additions only):

- New packages: `create/trajectories/`, `create/sources/mars/`,
  `create/sources/accumulate/`, `usage/trajectories/`.
- New modules: `create/arguments.py`, `create/dispatch.py`,
  `create/intervals.py`, `create/composite.py`,
  `create/sources/from_trajectories.py`.
- Restructured: `create/gridded/{context,creator,result}.py`,
  `create/recipe/{__init__,output}.py`, `create/creator.py`,
  `create/dataset.py`, `dates/__init__.py`, `dates/groups.py`,
  `usage/dataset.py`, `usage/forwards.py`, `usage/store.py`,
  `commands/inspect.py`, `commands/recipe/migrate.py`.
- Thin adapters: `sources/fdb.py`, `sources/hindcasts.py`,
  `sources/recentre.py`, `input/repeated_dates.py`.
- Tests: `tests/test_trajectories.py` (new, 613 lines),
  `tests/test_request_filter.py` (new), `tests/create/test_covering_factory.py`,
  `tests/create/test_forecast_covering.py`,
  `tests/create/trajectories*.yaml`, `tests/create/mars-{date,time}-filter.yaml`,
  `tests/create/hindcasts.yaml`.

## Execution order for the next files

1. `01_usage_recipe_author.md` — grounded in the test YAMLs, so any coverage
   gap in the recipe surface is found first.
2. `02_implementation_concepts.md` — uses the user surface as motivation.
3. `03_per_source_impl.md` — builds on (2).
4. `04_global_pros_cons.md` — only after (2) and (3), otherwise the
   tradeoffs would be listed before the mechanism they apply to.
5. `05_per_source_pros_cons.md` — uses (3) to anchor each bullet to a
   file:line.

Each file ends with a "pointers" section listing the exact files and lines
that substantiate its claims, so a downstream agent (or a human writing
real docs) can check the source without re‑reading the whole diff.
