# 01 — Recipe author view

What changes in the YAML recipe surface. Everything here is grounded in the
test recipes under `tests/create/` and in the Pydantic models under
[`src/anemoi/datasets/create/recipe/`](../src/anemoi/datasets/create/recipe/).

## 1. Three dataset layouts, three top‑level recipe shapes

`output.layout` (or its predecessor `output.format`) picks the creator:

```yaml
output:
  layout: gridded        # or: tabular, trajectories
```

The discriminator is `_output_discriminator` in
[recipe/output.py:184](../src/anemoi/datasets/create/recipe/output.py#L184):
it accepts both `layout:` and the legacy `format:` key. Dispatched to
one of `GriddedOutput`, `TabularOutput`, `TrajectoriesOutput`.

Creator selection is a `match` on the same key in
[create/creator.py:101-117](../src/anemoi/datasets/create/creator.py#L101-L117):
`"gridded"` → `SimpleGriddedCreator`, `"tabular"` → `TabularCreator`,
`"trajectories"` → `TrajectoryGriddedCreator`.

### 1.1 Gridded (unchanged top keys)

```yaml
dates:
  start: 2020-01-01 00:00:00
  end:   2020-01-11 00:00:00
  frequency: 12h

input: …
output: {layout: gridded}
```

`dates:` remains required; `base_dates:` is rejected. The recipe validator
`_check_steps` in
[recipe/__init__.py:39-63](../src/anemoi/datasets/create/recipe/__init__.py#L39-L63)
enforces the mutual exclusion.

### 1.2 Trajectories (new)

```yaml
base_dates:
  start: 2021-01-01 00:00:00
  end:   2021-01-02 00:00:00
  frequency: 12h

steps:
  start: 6
  end: 30
  frequency: 6h

input:
  mars: {type: fc, class: od, expver: "0001", grid: 20./20.,
         param: [q, t], levtype: pl, level: [50], stream: oper}

output: {layout: trajectories}
```

(Full example: [`tests/create/trajectories.yaml`](../tests/create/trajectories.yaml).)

Rules enforced by `_check_steps`:

- `layout: trajectories` ⇒ `base_dates:` and `steps:` are required; `dates:`
  is forbidden.
- Any other layout ⇒ `dates:` required; `base_dates:`/`steps:` forbidden.

`steps` accepts `start` / `end` / `frequency` as a timedelta spec. The
internal object is a `Steps` instance built in
[trajectories/context.py:25-40](../src/anemoi/datasets/create/trajectories/context.py#L25-L40)
(and its twin in `TrajectoryDates.__init__`,
[dates/__init__.py:460-475](../src/anemoi/datasets/dates/__init__.py#L460-L475)).
What actually gets materialised on disk is the Cartesian product
`basetimes × steps` — `TrajectoryDates.values` in
[dates/__init__.py:470](../src/anemoi/datasets/dates/__init__.py#L470).

#### On‑disk shape

5‑D array `(base_dates, variables, ensembles, steps, cells)` with dims
`("time", "variable", "ensemble", "step", "cell")`, plus coordinate arrays
`base_dates` and `steps`
([trajectories/creator.py:108-141](../src/anemoi/datasets/create/trajectories/creator.py#L108-L141)).
In the Zarr attributes, `layout = "trajectories"`,
`ensemble_dimension = 2`, `step_dimension = -2`
([trajectories/creator.py:82-94](../src/anemoi/datasets/create/trajectories/creator.py#L82-L94)).

#### Chunking

`TrajectoriesOutput` defaults to `{base_dates: 1, steps: 1, ensembles: 1}`
([recipe/output.py:154](../src/anemoi/datasets/create/recipe/output.py#L154)).
`get_chunking` only accepts the five dimension‑coord keys and rejects unknown
entries — consistent with `GriddedOutput.get_chunking` but the key set is
different.

### 1.3 Tabular — unchanged by this branch

`output: {layout: tabular}` is still the only tabular switch. The only
change is the new `.layout` property alias, so `_output_discriminator`
works for both `layout:` and the legacy `format:` key.

## 2. `from-trajectories:` — use a trajectory archive from a gridded recipe

New source registered at [sources/from_trajectories.py:51](../src/anemoi/datasets/create/sources/from_trajectories.py#L51)
(key `from-trajectories`). Lets a gridded recipe (`dates: …` driven) pull
fields from a forecast archive by picking a `(basetime, step)` pair for each
validity time.

```yaml
dates: {start: 2023-01-01, end: 2023-01-10, frequency: 6h}

input:
  from-trajectories:
    bases: "????-??-?? 00:00:00"   # fnmatch pattern on basetime; 00Z only
    steps: 6/to/24/by/6            # steps expanded via expand_to_by()
    source:
      mars:
        type: fc
        class: od
        param: [t, q]
        …
```

- `bases:` is an `fnmatch` wildcard against `basetime.strftime("%Y-%m-%d %H:%M:%S")`
  ([from_trajectories.py:101-111](../src/anemoi/datasets/create/sources/from_trajectories.py#L101-L111)).
  Omit to accept any basetime.
- `steps:` accepts the MARS step syntax (`"6/to/24/by/6"`, list, single int),
  parsed by `expand_to_by` from `sources/mars/retrieval.py`
  ([from_trajectories.py:76-81](../src/anemoi/datasets/create/sources/from_trajectories.py#L76-L81)).
  Omit to default to step 0.
- For each validity time, the source iterates the configured steps
  (smallest first) and picks the first `candidate = valid_time − step` that
  matches `bases`; if none matches it raises
  ([from_trajectories.py:113-132](../src/anemoi/datasets/create/sources/from_trajectories.py#L113-L132)).
- Inner `source:` must be a single‑key dict; the source is instantiated via
  `create_source` and then invoked with a `ForecastDates` argument — it must
  therefore be a source that registers `@for_forecast_dates` (MARS today).

## 3. MARS wildcard date / time filter

Pattern exercised in `tests/create/mars-date-filter.yaml` and
`tests/create/mars-time-filter.yaml`:

```yaml
mars:
  class: od
  type: fc
  date: "????-??-01"    # wildcard-date pattern
  time: 0               # becomes a base-time filter when date is a wildcard
  step: "0/to/240/by/12"
  param: [10u]
```

Semantics, implemented by `RequestFilter` in
[sources/mars/retrieval.py:151-253](../src/anemoi/datasets/create/sources/mars/retrieval.py#L151-L253):

- If `date` is a string containing `?`, it is popped out of the request and
  compiled to a regex (`?` → `.`, `-` stripped) matching `YYYYMMDD`.
- Any accompanying `time:` becomes a set of normalised `HHMM` strings.
- These filters are applied to the *computed base date / base time* for each
  expanded request. The MARS request sent over the wire never sees the
  wildcard.
- Only the valid‑dates dispatch path goes through the filter; the forecast /
  interval / forecast‑interval paths refuse filters
  (`_reject_filters` in [sources/mars/source.py:29-41](../src/anemoi/datasets/create/sources/mars/source.py#L29-L41)).
- Legacy `user_date` / `user_time` keys are rejected with a loud error
  ([retrieval.py:198-204](../src/anemoi/datasets/create/sources/mars/retrieval.py#L198-L204)).

## 4. `accumulate:` — new `covering` + `accumulation`

Three related changes.

### 4.1 `availability:` renamed to `covering: { auto: … }`

Old:
```yaml
accumulate:
  period: 6h
  availability: [(0, "0-6/0-12"), (12, "0-6/0-12")]
  source: {mars: {class: od}}
```

New canonical form:
```yaml
accumulate:
  period: 6h
  covering: {auto: [(0, "0-6/0-12"), (12, "0-6/0-12")]}
  source: {mars: {class: od}}
```

- Factory: `covering_factory` in
  [sources/accumulate/covering.py:198-254](../src/anemoi/datasets/create/sources/accumulate/covering.py#L198-L254).
- `covering: { auto: … }` picks `AutoCovering` (the Dijkstra search over an
  `IntervalGenerator`, i.e. the behaviour of the old `availability:`).
- `covering: { cycle: … }` is reserved (raises `NotImplementedError`).
- `covering: { forecast: … }` is **explicitly rejected** — the forecast
  branch is selected *implicitly* by passing `ForecastDates` to the source
  (i.e. by using the `accumulate:` block inside a `trajectories:` recipe).
- Legacy back‑compat: a bare list/string/mars‑dict at `covering:` is treated
  as the `auto:` value (same in the legacy `availability:` key, still parsed
  by `AccumulateSource.__init__` with a `DeprecationWarning` at
  [sources/accumulate/source.py:87-93](../src/anemoi/datasets/create/sources/accumulate/source.py#L87-L93)).
- `anemoi datasets recipe migrate` now calls `_fix_accumulate_availability`
  ([commands/recipe/migrate.py:190-210](../src/anemoi/datasets/commands/recipe/migrate.py#L190-L210))
  which rewrites old recipes in place.

### 4.2 `accumulation: from-zero | from-previous-step`

Required on the accumulate block **only** when the recipe is a trajectory
recipe (i.e. the caller passes `ForecastDates`). If it is missing,
`AccumulateSource.execute(ForecastDates)` raises with a clear message
([sources/accumulate/source.py:277-284](../src/anemoi/datasets/create/sources/accumulate/source.py#L277-L284)).

Meaning:

- `from-zero` — archive stores `a(0, step)` accumulations from the basetime;
  the window `[bt+sA, bt+sE]` is built as `+a(0, sE) − a(0, sA)` (two
  signed intervals).
- `from-previous-step` — archive stores per‑step increments
  `a(step−period, step)`; the window is the single interval `a(sA, sE)`.

Implemented in `ForecastCovering.cover` at
[sources/accumulate/covering.py:136-195](../src/anemoi/datasets/create/sources/accumulate/covering.py#L136-L195).

### 4.3 The full trajectory‑accumulation recipe pattern

```yaml
base_dates: {start: 2021-01-01, end: 2021-01-03, frequency: 12h}
steps:      {start: 6, end: 30, frequency: 3h}

input:
  join:
    - mars: {…}
    - pipe:
        - accumulate:
            period: 1h
            accumulation: from-zero
            source: {mars: {…}}
        - rename: {param: {tp: tp_accum_1h}}

output: {layout: trajectories}
```

From `tests/create/trajectories_accumulation.yaml`. Key points:

- No `covering:` key on the trajectory branch — the covering is computed
  from `accumulation` and the caller‑imposed basetime.
- `accumulate:` may be used multiple times with different periods to
  produce `tp_accum_1h`, `tp_accum_3h`, …, merged via `join:`.

## 5. `repeat-dates` with `mode: constant, date: null`

Edge case needed for forcings that do not depend on the group's dates
(single orography field, say). `DateMapperConstant`
([input/repeated_dates.py:231-291](../src/anemoi/datasets/create/input/repeated_dates.py#L231-L291))
accepts `date=None`: the inner source is invoked with an empty
`GroupOfDates`, and `MarsSource.execute(ValidDates=[])` bypasses the normal
validity‑date path and just fires the prebuilt request from the recipe
([sources/mars/source.py:54-69](../src/anemoi/datasets/create/sources/mars/source.py#L54-L69)).

```yaml
input:
  pipe:
    - repeat-dates:
        mode: constant
        date: null
        source:
          mars: {class: od, param: z, date: 20200101, …}
```

A non‑null `date:` is still supported and is parsed eagerly — invalid values
raise at config time.

## 6. Removed / deprecated recipe keys

- `output.order_by` — deprecated. Still parsed so old recipes keep running,
  but must equal the fixed default `["valid_datetime", "param_level",
  "number"]`; any other value raises
  ([recipe/output.py:73-99](../src/anemoi/datasets/create/recipe/output.py#L73-L99)).
  The `DeprecationWarning` is emitted unconditionally when the key is
  present.
- `output.flatten_grid` — removed. Flattening is always on
  ([gridded/result.py:369](../src/anemoi/datasets/create/gridded/result.py#L369)).
  The attribute is no longer written to Zarr
  ([usage/misc.py](../src/anemoi/datasets/usage/misc.py), the
  `root.attrs["flatten_grid"] = True` line is deleted).
- `output.remapping` — already deprecated on `main`; unchanged.
- `mars` `user_date` / `user_time` — now raise with a clear message pointing
  to the wildcard shorthand.

## 7. Reader‑side changes a recipe author will notice

Even though the reader API is not the focus, a few surface changes matter
because they are how a user validates their own build:

- `anemoi datasets inspect` now detects trajectory datasets (via the
  `.steps` attribute) and prints a different summary block with
  `Date start` / `Date end` (envelope of base+step) and a base‑date / step
  sub‑summary
  ([commands/inspect.py:255-305](../src/anemoi/datasets/commands/inspect.py#L255-L305)).
- In Python: `Dataset.dates` returns `base_dates` for trajectory stores
  ([create/dataset.py:217-224](../src/anemoi/datasets/create/dataset.py#L217-L224)
  and `usage/trajectories/store.py:110-117`). `Dataset.frequency` is
  explicitly unavailable for trajectories — `base_frequency` and
  `step_frequency` must be used
  ([usage/trajectories/store.py:248-257](../src/anemoi/datasets/usage/trajectories/store.py#L248-L257)).
- Subsetting acquires trajectory‑specific kwargs on `open_dataset(...)`:
  `step`, `steps`, `step_start`, `step_end`, `step_frequency`,
  `base_start`, `base_end`
  ([usage/dataset.py:323-360](../src/anemoi/datasets/usage/dataset.py#L323-L360)).
  `step=…` returns a `SingleStepView` that looks exactly like a gridded
  dataset at that forecast step; `steps=[…]` or a step range returns a
  `StepSubset`.

## 8. Summary table

| Key | Layout | Required | Default | Mutually exclusive with |
| --- | ------ | -------- | ------- | ----------------------- |
| `dates`                | gridded, tabular | yes | — | `base_dates`, `steps` |
| `base_dates`           | trajectories | yes | — | `dates` |
| `steps`                | trajectories | yes | — | `dates` |
| `output.layout`        | any | yes (was `format`) | `gridded` | — |
| `output.order_by`      | gridded | no (deprecated) | `["valid_datetime","param_level","number"]` | — |
| `output.flatten_grid`  | any | removed | — | — |
| `accumulate.covering`  | gridded (archive) | yes | — | `availability` |
| `accumulate.availability` | gridded (archive) | no (deprecated) | — | `covering` |
| `accumulate.accumulation` | trajectories | yes in trajectory branch | — | — |

## Pointers

- `src/anemoi/datasets/create/recipe/__init__.py` — `_check_steps`
  validator, `base_dates` / `steps` fields, metadata sanitisation.
- `src/anemoi/datasets/create/recipe/output.py` — `GriddedOutput`,
  `TabularOutput`, `TrajectoriesOutput`, `_output_discriminator`,
  `_FIXED_ORDER_BY`.
- `src/anemoi/datasets/create/creator.py:101-117` — layout → creator match.
- `src/anemoi/datasets/create/sources/from_trajectories.py` — whole file.
- `src/anemoi/datasets/create/sources/mars/retrieval.py:151-253` —
  `RequestFilter`.
- `src/anemoi/datasets/create/sources/accumulate/covering.py:198-254` —
  `covering_factory`.
- `src/anemoi/datasets/create/sources/accumulate/source.py:87-93,277-284` —
  `availability` deprecation and `accumulation` gate.
- `src/anemoi/datasets/create/input/repeated_dates.py:231-291` —
  `DateMapperConstant(date=None)`.
- `src/anemoi/datasets/commands/recipe/migrate.py:190-210` —
  `_fix_accumulate_availability`.
- `src/anemoi/datasets/commands/inspect.py:255-305` — trajectory branch.
- Recipes: `tests/create/trajectories.yaml`,
  `tests/create/trajectories_accumulation.yaml`,
  `tests/create/mars-date-filter.yaml`,
  `tests/create/mars-time-filter.yaml`,
  `tests/create/hindcasts.yaml`.
