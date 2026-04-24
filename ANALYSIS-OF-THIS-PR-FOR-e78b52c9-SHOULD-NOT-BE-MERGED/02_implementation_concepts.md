# 02 — Implementation concepts

The concepts the rest of the code on this branch is built on. Each section
describes *what* the concept is and *why* it exists, with direct pointers to
the source.

## 1. Typed source arguments (`create/arguments.py`)

Sources on `main` used to accept "a list of datetimes" (`list[datetime]` or
`GroupOfDates`) and were expected to handle accumulations via a sibling
`IntervalsDatesProvider`. The new branch turns this into a small, typed,
Pydantic‑free class hierarchy. Every source's `execute(...)` now receives
one of four concrete argument classes.

### 1.1 The 2×2 matrix

| Basetime? | Instant (one snapshot)           | Aggregate (one value per window) |
| --------- | -------------------------------- | -------------------------------- |
| No        | `ValidDates`                     | `Intervals` *(subclass of `ValidDates`)* |
| Yes       | `ForecastDates`                  | `ForecastIntervals` *(subclass of `ForecastDates`)* |

- `ValidDates` — a list of validity times. Analysis, reanalysis,
  observations, repeat‑dates constants.
- `ForecastDates` — list of `(valid_time, basetime)` pairs. Instantaneous
  NWP forecasts; the primary input to trajectories sources.
- `Intervals` — a list of dates *plus* a flat list of `SignedInterval`s
  that cover the requested accumulation windows. Subclass of `ValidDates`
  so that a source which only registered `@for_valid_dates` still receives
  instants for each interval via MRO fallback (see §3).
- `ForecastIntervals` — list of `(valid_time, basetime, period)` triples,
  plus a flat `SignedInterval` list whose `.base` is the basetime. Subclass
  of `ForecastDates`.

Files: [`create/arguments.py:52-66`](../src/anemoi/datasets/create/arguments.py#L52-L66)
for the base, [:61-117](../src/anemoi/datasets/create/arguments.py#L61-L117) for
`ValidDates`, [:125-160](../src/anemoi/datasets/create/arguments.py#L125-L160)
for `ForecastDates`, [:168-252](../src/anemoi/datasets/create/arguments.py#L168-L252)
for `Intervals`, [:260-333](../src/anemoi/datasets/create/arguments.py#L260-L333)
for `ForecastIntervals`.

### 1.2 Conversion helpers

Each class has methods that build the other types without losing
information. They are used by composite sources (e.g.
`FromTrajectoriesSource`) and by `AccumulateSource` to hand the inner source
the shape it expects.

- `ValidDates.as_intervals(period)` — trivial one‑interval covering.
- `ValidDates.with_basetime(fn)` — given a `valid_time → basetime` pure
  function, produce a `ForecastDates`.
- `ForecastDates.as_forecast_intervals(period)` — trivial covering for
  each pair.
- `Intervals.with_basetime(...)` — explicitly **not** implemented
  (`NotImplementedError`): the archive‑resolved intervals have their own
  basetimes; if you want forecast accumulations, use `AccumulateSource`
  with `ForecastDates` input.
- `Intervals.adjust_request(interval, request)` — returns
  `(valid_time, adjusted_request, step_hours)` for a single interval;
  stamps `date`/`time`/`step` onto the request. Asserts `interval.base is
  not None`. The same signature exists on `ForecastIntervals`.

### 1.3 Why subclassing and not composition

The `Intervals ⊂ ValidDates` / `ForecastIntervals ⊂ ForecastDates`
relationship is intentional and interacts directly with the dispatch (§3).
A source that does not care about the accumulation period (e.g. a
re‑projecting grib‑index for instants) registers only `@for_valid_dates` and
still receives `Intervals` via MRO lookup — its `execute` iterates over
`argument.dates` as usual, gets the snapshots, and the covering is handled
elsewhere. A source that *does* care (MARS, FDB) registers `@for_intervals`
explicitly, gets the flat interval list, and calls `adjust_request`.

## 2. `SignedInterval` (`create/intervals.py`)

A `SignedInterval` is the datum the covering algorithms emit and the
accumulators consume. It is *signed* — `start > end` is legal — so a
`from-zero` archive stored as `a(0, sE)` can be subtracted from `a(0, sA)`
by negating the second interval.

Interface: `start`, `end`, optional `base` (the model‑run time); derived
`length`, `sign`, `min`, `max`, `valid_time == max`; `__neg__`, `__eq__`,
`__hash__`; a rich (terminal‑colour aware) `__repr__`. File:
[`create/intervals.py`](../src/anemoi/datasets/create/intervals.py).

`base=None` marks an interval that came from a *valid‑time‑indexed* backend
(grib‑index). The asymmetry is load‑bearing: `Intervals.adjust_request`
asserts `base is not None`, so grib‑index has to bypass the helper — which
is exactly why it owns its own `@for_intervals` overload
([sources/grib_index.py:622-644](../src/anemoi/datasets/create/sources/grib_index.py#L622-L644)).

## 3. Dispatch decorators (`create/dispatch.py`)

The goal is that every source expose exactly one method named `execute`
with *multiple signatures*, dispatched on the argument type — without
wrapping every method in manual `isinstance(...)` ladders and without
pulling in `functools.singledispatchmethod` (which dispatches on `self` as
well, and does not handle descriptors cleanly across inheritance).

### 3.1 The `_MultiDispatch` descriptor

`_MultiDispatch` ([dispatch.py:62-128](../src/anemoi/datasets/create/dispatch.py#L62-L128))
is a Python descriptor whose `__get__` returns a closure that looks up the
right overload by argument type. The registry is a simple `dict[type, fn]`;
the lookup walks `type(argument).__mro__` so subclasses dispatch correctly.

Back‑compat shims in the dispatch closure
([dispatch.py:99-108](../src/anemoi/datasets/create/dispatch.py#L99-L108)):
a plain `list[datetime]` is wrapped in `ValidDates`, a `GroupOfDates` is
unwrapped into `ValidDates(group.dates)`. That keeps old sources and the
trajectories `TrajectoryGroups` (which already yields `ForecastDates`)
working through the same entry point.

### 3.2 Frame‑inspection accumulation

Each `@for_*` decorator uses `sys._getframe(1).f_locals` to see whether a
`_MultiDispatch` already exists under the current `def execute` name, and
either reuses it or creates one
([dispatch.py:136-165](../src/anemoi/datasets/create/dispatch.py#L136-L165)).
The effect is that writing

```python
class MySource(Source):
    @for_valid_dates
    def execute(self, dates: ValidDates): ...

    @for_forecast_dates
    def execute(self, dates: ForecastDates): ...
```

yields a single `_MultiDispatch` descriptor on the class with two entries
registered — despite the fact that each `def` normally overwrites the
previous binding.

### 3.3 `DispatchedSource` mixin

Sources that opt in multiple‑dispatch inherit from `DispatchedSource`
([dispatch.py:204-228](../src/anemoi/datasets/create/dispatch.py#L204-L228)).
It is a pure marker today (the decorator handles everything) but reserves
the spot for future invariants (e.g. asserting that `DispatchedSource` is
last in MRO so `_MultiDispatch` is not shadowed by a `LegacySource.execute`,
as noted in the module docstring at
[dispatch.py:44-48](../src/anemoi/datasets/create/dispatch.py#L44-L48)).

### 3.4 Lazy decorator factories

The `for_*` names are not imported at module load; they are produced on
demand by `__getattr__` ([dispatch.py:172-196](../src/anemoi/datasets/create/dispatch.py#L172-L196))
which imports the argument classes from `create/arguments.py` only when
asked, to avoid circular imports during package initialisation.

## 4. `Composite` and `Pipe` (`create/composite.py`)

A `Composite` is a callable that wraps a `Source`. `Pipe(source,
[composites])` applies them left‑to‑right. This is the programmatic
counterpart of the YAML `pipe:` action tree, used when a source wants to
construct an internal pipeline (rather than a recipe author stitching one
together).

`Composite` is deliberately distinct from
`anemoi.transform.filter.Filter`:
- `Composite` wraps a *source* (pre‑retrieval).
- `Filter` transforms a *FieldList* (post‑retrieval).

File: [`create/composite.py`](../src/anemoi/datasets/create/composite.py).
Callers today: placeholder for future refactors; the branch primarily
exposes the ABC so downstream helpers stop reaching for lambdas.

## 5. Context — per‑layout owner of cube geometry

On `main`, `GriddedContext` owned the cube `order_by`, the `flatten_grid`
flag, the remapping. On the branch, the `Context` abstract base
([input/context.py:18-69](../src/anemoi/datasets/create/input/context.py#L18-L69))
stays a thin service locator (`create_result`, `create_source`, `trace`,
`resolve`, `register`, `join`) and each layout subclasses it:

- `SimpleGriddedContext`
  ([gridded/context.py:25-68](../src/anemoi/datasets/create/gridded/context.py#L25-L68)) —
  fixed `order_by = ["valid_datetime", "param_level", "number"]`.
  `flatten_grid` is gone; cube flattening is always on. Builds
  `SimpleGriddedResult`.
- `TrajectoryGriddedContext`
  ([trajectories/context.py:43-130](../src/anemoi/datasets/create/trajectories/context.py#L43-L130)) —
  fixed `order_by = ["traj_point", "param_level", "number"]`, where
  `traj_point` is a *composite remapping key* injected as
  `"{date}_{time}_{step}"` so that the cube collapses
  `(date, time, step)` into one axis (avoiding a spurious Cartesian
  product on non‑rectangular basetime × step sets). Builds
  `TrajectoryGriddedResult`.

The `order_by` attribute is now a class‑level constant — not something the
user can configure. The deprecated recipe field `output.order_by` is
validated against the context's fixed value at `Recipe.__init__` time
([recipe/output.py:73-99](../src/anemoi/datasets/create/recipe/output.py#L73-L99)),
and `GriddedCreator.collect_metadata` reads it off the context, not the
recipe ([gridded/creator.py:68-70](../src/anemoi/datasets/create/gridded/creator.py#L68-L70)).

## 6. Layout split — creator × result × context

Each layout is a triple of classes sharing a fixed interface:

| Layout        | Creator                       | Context                       | Result                        |
| ------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| gridded       | `SimpleGriddedCreator`        | `SimpleGriddedContext`        | `SimpleGriddedResult`         |
| trajectories  | `TrajectoryGriddedCreator`    | `TrajectoryGriddedContext`    | `TrajectoryGriddedResult`     |
| tabular       | `TabularCreator` *(unchanged)* | `TabularContext`              | `TabularResult`               |

`GriddedCreator` is now an abstract base
([gridded/creator.py:35-336](../src/anemoi/datasets/create/gridded/creator.py#L35-L336))
with two abstract methods:

- `initialise_dataset(dataset)` — create the Zarr arrays (shape, dims).
- `context()` — return the layout‑specific context.

Shared logic on the base class:

- `_metadata_dates()` / `_metadata_date_range()` hooks —
  `SimpleGriddedCreator` uses `groups.provider.values`,
  `TrajectoryGriddedCreator` uses the factorised base dates and returns the
  valid‑time envelope.
- `collect_metadata()` — fills `order_by`, `start_date`, `end_date`,
  `statistics_*`, `variables`, etc.; overrides on
  `TrajectoryGriddedCreator` add `base_dates`, `steps`, `ensemble_dimension
  = 2`, `step_dimension = -2`, `layout = "trajectories"`
  ([trajectories/creator.py:81-100](../src/anemoi/datasets/create/trajectories/creator.py#L81-L100)).
- `load_result()` — on gridded it reshapes by valid datetime axis; on
  trajectories the creator overrides `load_result` entirely to write
  per‑cubelet into the 5‑D `(date, var, ens, step, cell)` array by reading
  `(basetime, step, variable, ensemble)` off each field's metadata
  ([trajectories/creator.py:143-235](../src/anemoi/datasets/create/trajectories/creator.py#L143-L235)).
- Statistics (`_compute_partial_statistics`) reads `dataset.dates`, which
  now transparently falls back to `base_dates` for trajectories
  ([create/dataset.py:217-224](../src/anemoi/datasets/create/dataset.py#L217-L224)).

`GriddedResult` becomes the shared implementation, with `SimpleGriddedResult`
and `TrajectoryGriddedResult` as concrete subclasses
([gridded/result.py:308-694](../src/anemoi/datasets/create/gridded/result.py#L308-L694)).
`build_coords()` is now layout‑agnostic: it takes the last two `order_by`
keys as `(variables, ensembles)` and hands any leading extra keys (`step`
for trajectories) to a `_post_build_coords` hook
([gridded/result.py:570-598](../src/anemoi/datasets/create/gridded/result.py#L570-L598)).
Cube flattening is always on (`flatten_values=True`,
[gridded/result.py:369](../src/anemoi/datasets/create/gridded/result.py#L369)).

## 7. `TrajectoryDates` and `TrajectoryGroups`

The time axis of a trajectory dataset is `base_dates × steps`; the machinery
that feeds the creator has to yield one group per basetime‑bunch and, inside
each group, `(valid_time, basetime)` pairs rather than plain dates.

- `TrajectoryDates` ([dates/__init__.py:440-504](../src/anemoi/datasets/dates/__init__.py#L440-L504)).
  Builds the basetimes via the existing `DatesProvider.from_config(...)`
  (so `start`/`end`/`frequency`/`missing`/`hindcasts` all work) and turns
  `steps: {start,end,frequency}` into a numpy `timedelta64[ns]` array.
  `values` is the flat list of `(basetime, step)` tuples; `factorise()`
  returns `(sorted_unique_basetimes, sorted_unique_steps)`; `frequency` is
  forwarded to the basetimes provider.
- `TrajectoryGroups` ([dates/groups.py:379-438](../src/anemoi/datasets/dates/groups.py#L379-L438)).
  Subclass of `Groups`. Its `__iter__` groups the basetimes via the normal
  `Grouper` (monthly/daily/…), filters missing via the `Filter`, and yields
  `ForecastDates([(basetime + step, basetime), …])` — a typed argument
  ready for dispatch. `one_date()` returns a single‑item `ForecastDates`
  used by `minimal_input` probing. `first_date` / `last_date` are the
  envelope (`min(basetime) + first_step`, `max(basetime) + last_step`).
- `GrouperByKey._key` ([dates/groups.py:332-337](../src/anemoi/datasets/dates/groups.py#L332-L337))
  now unwraps `(basetime, step)` tuples and applies the key to the
  basetime, so `group_by: monthly` works for trajectory providers.
- `GroupOfDates.__init__` accepts tuples opaquely
  ([dates/groups.py:44-52](../src/anemoi/datasets/dates/groups.py#L44-L52)) —
  so the tuple‑aware grouper can still feed the stock `GroupOfDates` to
  intermediate machinery.

## 8. Recipe discriminator + `base_dates` / `steps`

`Recipe` ([recipe/__init__.py:39-120](../src/anemoi/datasets/create/recipe/__init__.py#L39-L120)):

- Adds `base_dates: DotDictField | None` and `steps: DotDictField | None`.
- The `_check_steps` model validator enforces mutual exclusion between the
  gridded/tabular key (`dates`) and the trajectories keys
  (`base_dates` + `steps`).
- When serialising recipe metadata, any unused trajectory‑only key is
  dropped so non‑trajectory Zarr stores keep the same metadata shape they
  had on `main` ([recipe/__init__.py:119-125](../src/anemoi/datasets/create/recipe/__init__.py#L119-L125)).

`Output` ([recipe/output.py](../src/anemoi/datasets/create/recipe/output.py)):

- Pydantic `Annotated[Union[...], Discriminator(_output_discriminator)]`.
- Discriminator accepts both `layout:` and legacy `format:`.
- `TrajectoriesOutput` has its own default chunking
  `{base_dates: 1, steps: 1, ensembles: 1}` and a 5‑D `get_chunking`.

## 9. Zarr on‑disk layout

Three concrete on‑disk shapes live under the same store format:

### 9.1 Gridded (unchanged)

Arrays: `data(time, variable, ensemble, cell)` + `dates(time)` +
`latitudes(cell)` + `longitudes(cell)`.
Attributes: `start_date`, `end_date`, `frequency`, `variables`,
`variables_metadata`, `missing_dates`, `order_by` = ["valid_datetime",
"param_level", "number"], `ensemble_dimension` = len(ensembles),
`allow_nans`, `layout` (new, == `gridded`).
`flatten_grid` is no longer written
([usage/misc.py:676](../src/anemoi/datasets/usage/misc.py#L676)).

### 9.2 Trajectories (new)

Arrays: `data(time, variable, ensemble, step, cell)` + `base_dates(time)`
+ `steps(step)` + `latitudes` + `longitudes`.
Attributes: `layout = "trajectories"`, `steps = recipe.steps`,
`ensemble_dimension = 2`, `step_dimension = -2`, plus the shared
`start_date` / `end_date` / `base_dates` envelope
([trajectories/creator.py:81-100](../src/anemoi/datasets/create/trajectories/creator.py#L81-L100)).

### 9.3 Tabular (unchanged)

Left untouched by the branch. Mentioned here only because the store
discriminator has to handle three cases.

### 9.4 Reader side — `Dataset` base

`Dataset.dates` now falls back to `base_dates` for trajectories
([create/dataset.py:217-224](../src/anemoi/datasets/create/dataset.py#L217-L224)),
and cached properties `base_dates` / `steps` expose the new arrays directly.

## 10. Reader side — `usage/`

`ZarrStore.open` ([usage/store.py:211-222](../src/anemoi/datasets/usage/store.py#L211-L222))
dispatches on the `layout` attribute; `"trajectories"` builds a
`TrajectoriesZarr`.

`TrajectoriesZarr`
([usage/trajectories/store.py:34-380](../src/anemoi/datasets/usage/trajectories/store.py#L34-L380)):

- Exposes `base_dates`, `steps`, `base_start_date`, `base_end_date`,
  `base_frequency`, `step_start`, `step_end`, `step_frequency`.
- Aliases `dates` to `base_dates` so shared `Dataset` logic works.
- Removes the single‑axis `frequency` (raises `AttributeError` pointing
  to `base_frequency` / `step_frequency`).
- Envelope logic for `_dates_to_indices(start, end)`: a base date is kept
  iff `[base + step_start, base + step_end] ⊂ [start, end]`, enforced in
  [:304-340](../src/anemoi/datasets/usage/trajectories/store.py#L304-L340).
- `usage_factory_load` wires `Subset`, `StepSubset`, `SingleStepView` from
  the per‑package module ([:378-381](../src/anemoi/datasets/usage/trajectories/store.py#L378-L381)).

`StepSubset`, `SingleStepView`, `Subset`
([usage/trajectories/subset.py](../src/anemoi/datasets/usage/trajectories/subset.py)):

- `StepSubset` narrows the step axis (`-2`). Shape is `(..., k_steps,
  cell)`.
- `SingleStepView` selects one step and drops the step axis — produces a
  4‑D view indistinguishable from a gridded dataset at that step.
- `Subset` narrows the base‑date axis (axis 0) — the trajectory analogue
  of the gridded `Subset`.

`Forwards` ([usage/forwards.py:97+](../src/anemoi/datasets/usage/forwards.py#L97))
now forwards `base_dates`, `base_start_date`, `base_end_date`,
`base_frequency`, `step_frequency`, `start_date`, `end_date` so any
wrapper (`Mask`, `Rename`, …) keeps working transparently on trajectory
datasets.

`Dataset._subset` ([usage/dataset.py:323-360](../src/anemoi/datasets/usage/dataset.py#L323-L360))
adds the `step`, `steps`, `step_start`, `step_end`, `step_frequency`,
`base_start`, `base_end` kwargs. `_subset` returns the right wrapper.

## 11. Writer side — `AccumulateSource`'s `covering` layer

Accumulation used to interleave "what intervals the archive can provide"
and "which intervals we pick to cover a window" in the same class tree
(`IntervalGenerator` → `covering_intervals`). The branch splits the two:

- **Availability** — what the archive contains.
  `IntervalGenerator` hierarchy under
  [`sources/accumulate/interval_generators.py`](../src/anemoi/datasets/create/sources/accumulate/interval_generators.py).
  Unchanged in shape.
- **Covering** — how a `[start, end]` window is expressed as a signed sum
  of available intervals. ABC `Covering` with two concrete implementations:
  - `AutoCovering` — wraps the existing Dijkstra search.
  - `ForecastCovering` — no search; given an externally‑imposed basetime
    and the `accumulation` flag, emits the trivial 1‑ or 2‑interval
    decomposition directly.

File: [`sources/accumulate/covering.py`](../src/anemoi/datasets/create/sources/accumulate/covering.py).

The split lets the trajectory branch of `AccumulateSource` skip the search
entirely — it already knows the basetime from the `ForecastDates` argument.
The archive branch keeps the search.

## 12. Composition summary

End‑to‑end for a trajectory accumulation recipe:

1. Recipe is parsed → `TrajectoriesOutput` discriminator picks the
   trajectories branch.
2. `TrajectoryGriddedCreator.groups` builds a `TrajectoryGroups` over the
   `TrajectoryDates` provider.
3. Iterating `TrajectoryGroups` yields `ForecastDates` — one per group.
4. The input tree walks normally. For each leaf source:
   - `MarsSource` / `FdbSource` register `@for_forecast_dates` and
     `@for_forecast_intervals` — they consume the `ForecastDates`
     directly.
   - `AccumulateSource` receives `ForecastDates`, builds a
     `ForecastCovering`, and produces a `ForecastIntervals` that it hands
     to the inner source via the same dispatch.
5. Fields come back; `TrajectoryGriddedContext.create_result(...)` builds
   a `TrajectoryGriddedResult`, whose cube is ordered by
   `(traj_point, param_level, number)`.
6. `TrajectoryGriddedCreator.load_result` reads `(date, time, step, param,
   levelist, number)` from each field's metadata and writes into the
   5‑D `data` array.

## Pointers

- Typed arguments: `src/anemoi/datasets/create/arguments.py`.
- Dispatch: `src/anemoi/datasets/create/dispatch.py`.
- SignedInterval: `src/anemoi/datasets/create/intervals.py`.
- Composite/Pipe: `src/anemoi/datasets/create/composite.py`.
- Context ABC: `src/anemoi/datasets/create/input/context.py`.
- Contexts: `src/anemoi/datasets/create/gridded/context.py`,
  `src/anemoi/datasets/create/trajectories/context.py`.
- Creators: `src/anemoi/datasets/create/creator.py` (factory),
  `src/anemoi/datasets/create/gridded/creator.py` (base + simple),
  `src/anemoi/datasets/create/trajectories/creator.py`.
- Results: `src/anemoi/datasets/create/gridded/result.py`,
  `src/anemoi/datasets/create/trajectories/result.py`.
- Dates providers/groups: `src/anemoi/datasets/dates/__init__.py`,
  `src/anemoi/datasets/dates/groups.py`.
- Recipe: `src/anemoi/datasets/create/recipe/__init__.py`,
  `src/anemoi/datasets/create/recipe/output.py`.
- Zarr reader: `src/anemoi/datasets/usage/store.py`,
  `src/anemoi/datasets/usage/trajectories/store.py`,
  `src/anemoi/datasets/usage/trajectories/subset.py`,
  `src/anemoi/datasets/usage/dataset.py`,
  `src/anemoi/datasets/usage/forwards.py`.
- Covering: `src/anemoi/datasets/create/sources/accumulate/covering.py`,
  `src/anemoi/datasets/create/sources/accumulate/interval_generators.py`,
  `src/anemoi/datasets/create/sources/accumulate/covering_intervals.py`.
