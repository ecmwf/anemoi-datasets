---
data_stage: source_data
product: trajectories
version: 0.1.0
status: draft
source: adr-2.md
---

## 1. Introduction

This document specifies the architecture and storage requirements for **trajectories datasets** in anemoi-datasets — a third layout alongside the existing `gridded` and `tabular` layouts.

A trajectories dataset is a dataset where each sample covers a sequence of forecast steps, in addition to the date and variable dimensions of a gridded dataset. The name of the layout is `trajectories`.

### Motivation

Gridded datasets store one snapshot per valid datetime. For ML models that reason about the evolution of the atmosphere over a forecast horizon (trajectory models, rollout training, diffusion-based forecasters), the training sample is naturally a sequence of states at successive forecast steps. Rather than reconstructing step sequences at data-load time from a flat date axis, we introduce a dedicated `trajectories` layout that encodes the step dimension explicitly at build time.

Reconstructing step sequences at data-load time is fragile and error-prone: it requires the loader to infer which dates belong to the same run, to know the expected step list, to detect and handle missing steps, and to reason about whether a given valid time comes from a short or a long forecast. Any mismatch between the assumed and actual step structure silently produces wrong training samples. Encoding the step dimension at build time eliminates this class of bugs entirely.

### Scope

This specification covers:

- The storage layout for trajectories datasets.
- The recipe configuration needed to build such datasets.
- The changes to sources and creators needed to produce trajectory (step-indexed) data.
- The Python API changes needed to open and query trajectories datasets.

It does **not** cover:

- Combining trajectories datasets with non-trajectories datasets at training time (deferred to a future ADR).
- Irregular-grid (tabular) trajectories.

### Conceptual Model

A trajectories dataset adds a **step dimension** to the gridded layout.
The implemented on-disk layout places the step axis at position ``-2``
(just before the grid-cell axis), keeping the first three axes identical
to the gridded layout:

```
data[base_date, variable, ensemble, step, grid_cell]
```

| Dimension   | Axis | Description                                                      |
|-------------|------|------------------------------------------------------------------|
| `base_date` | 0    | Forecast initialisation (analysis) datetime                      |
| `variable`  | 1    | Atmospheric/ocean variable (same as gridded)                     |
| `ensemble`  | 2    | Ensemble member index (1 for deterministic)                      |
| `step`      | -2   | Forecast lead time in hours                                      |
| `grid_cell` | -1   | Flattened spatial index (same as gridded)                        |

Keeping variables on axis 1 means per-variable statistics and inspection
code written for gridded datasets work unchanged on trajectories.

The `base_date` dimension refers to the **analysis time**, not the valid
time.  The valid time for a given element is `base_date + step`.  On disk
this dimension is stored in an array called ``base_dates`` (not
``dates``), and the recipe uses a ``base_dates:`` block to configure it.


## 2. Usage API

### 2.1 Step Selection

Analogous to date selection, step selection should be supported at open time. The exact API is to be decided; candidates are:

```python
# Option A — keyword arguments
ds = anemoi.datasets.open_dataset("...", start_step=6, end_step=72, step_frequency=6)
ds.shape  # (730, 80, 1, 40320, 12)  — (dates, variables, ensemble, grid_cells, steps)

# Option B — steps dict (mirrors start/end/frequency for dates)
ds = anemoi.datasets.open_dataset("...", steps=dict(start=6, end=72, frequency=6))
ds.shape  # (730, 80, 1, 40320, 12)  — (dates, variables, ensemble, grid_cells, steps)
```

Both options are equivalent; one will be chosen during implementation.

A list of explicit steps can also be passed to select an arbitrary subset:

```python
# Option C — explicit list of steps
ds = anemoi.datasets.open_dataset("...", steps=[6, 24, 48, 72])
ds.shape  # (730, 80, 1, 40320, 4)  — (dates, variables, ensemble, grid_cells, steps)
```

> When steps are selected as an explicit list (Option C), the step spacing is no longer guaranteed to be uniform. In this case `ds.step_frequency` returns `None`.

Finally, selecting a **single step** should drop the step dimension entirely, making the dataset behave like a standard gridded dataset — the same shape and indexing semantics as the current non-trajectories datasets:

```python
# Single step — step dimension is dropped
ds = anemoi.datasets.open_dataset("...", step=24)
ds.shape  # (730, 80, 1, 40320)  — (dates, variables, ensemble, grid_cells)
```

> Step values above are given as plain integers (hours) for illustration. String formats such as `"6h"` or any other notation already in use by anemoi users for date frequencies should also be supported, following whatever convention is adopted consistently across the API.

### 2.2 Temporal Metadata

Trajectories datasets have **two time axes** (base dates and steps), so some properties that are unambiguous for gridded datasets become ambiguous. The API uses explicit prefixed names to avoid confusion.

#### What fails

| Property       | Gridded       | Trajectory | Rationale                                     |
|----------------|---------------|------------|-----------------------------------------------|
| `ds.dates`     | valid times   | **fails**  | Ambiguous — base dates or valid times?         |
| `ds.frequency` | date interval | **fails**  | Two frequencies exist (base and step)          |

These properties raise an error with a message directing the user to the explicit alternatives below.

#### Envelope properties (generic, work for any dataset type)

| Attribute      | Type       | Description                                                        |
|----------------|------------|--------------------------------------------------------------------|
| `ds.start_date`| `datetime` | Earliest valid time in the dataset (`first_base + first_step`)     |
| `ds.end_date`  | `datetime` | Latest valid time in the dataset (`last_base + last_step`)         |

For gridded datasets these are identical to the first/last entry of `ds.dates`. For trajectory datasets they describe the valid-time envelope.

#### Base-date axis

| Attribute             | Type       | Description                          |
|-----------------------|------------|--------------------------------------|
| `ds.base_dates`       | `ndarray`  | Array of analysis (base) datetimes   |
| `ds.base_start_date`  | `datetime` | First base date                      |
| `ds.base_end_date`    | `datetime` | Last base date                       |
| `ds.base_frequency`   | `timedelta`| Interval between consecutive base dates |

#### Step axis

| Attribute             | Type        | Description                          |
|-----------------------|-------------|--------------------------------------|
| `ds.steps`            | `ndarray`   | Array of step values as timedeltas   |
| `ds.step_start`       | `timedelta` | First step                           |
| `ds.step_end`         | `timedelta` | Last step                            |
| `ds.step_frequency`   | `timedelta` | Step interval                        |

> The exact timedelta type (`numpy.timedelta64` or `datetime.timedelta`) is to be decided during implementation, following the convention already used for dates in the rest of the API.

> **Naming convention:** top-level envelope properties use property-first naming (`start_date`, `end_date`). Axis-specific properties use axis-first naming (`base_*`, `step_*`). This allows tab-completion to group properties by axis (e.g. `ds.base_<tab>` lists all base-date properties).

### 2.3 Date Range Selection and Filtering

The existing selectors should continue to work unchanged on trajectories datasets:

- `select` / `drop` — variable selection
- `cutout` and area selection — grid subsetting

#### Strict envelope filtering

Date range selection via `open_dataset(path, start=x, end=y)` uses **strict envelope filtering**: a base date is kept if and only if its **entire** step range falls within the requested window.

```python
# A base date is kept iff:
base + step_start >= x   AND   base + step_end <= y

# Equivalently, filtering on the base-date axis:
base >= x - step_start   AND   base <= y - step_end
```

This guarantees:

- **No data outside [x, y].** Every valid time (`base + step`) in the result lies within the requested window. The consumer never needs to post-filter.
- **Rectangular output.** The result is a simple slice on the base-date axis; the shape stays `(n_bases, n_steps, ...)` with no ragged arrays or masking.
- **Self-consistency.** `open_dataset(start=ds.start_date, end=ds.end_date)` returns all base dates (roundtrip property).
- **Uniform contract with gridded.** In the gridded case, `open_dataset(start=x, end=y)` also returns only data within [x, y]. The same guarantee holds here.

> **Minimum window size.** The window `[x, y]` must be at least as wide as the step span (`step_end - step_start`), otherwise no base date can satisfy the constraint and the result is empty.

### 2.4 Merging

Merging two trajectories datasets is supported only when their step configurations are identical (same start, end, and frequency). Merging datasets with partially overlapping or differently spaced steps is not supported initially and will be addressed later if the need arises.

Merging a trajectories dataset with a non-trajectories dataset is **out of scope** for now and should raise a clear, descriptive error if attempted. If this proves easy to add during the implementation, these feature may be implemented, but use cases for mixed merging could also be addressed in a future ADR when concrete needs arise.

## 3. Building trajectories datasets

### 3.1 Step Configuration

Steps stored in the dataset are assumed to be **regularly spaced**. The step range is specified in the recipe by:

```yaml
base_dates:
  start: 2020-01-01
  end: 2022-12-31
  frequency: 24h

steps:
  start: 6       # first step in hours; default is 6
  end: 240       # last step (inclusive) in hours
  frequency: 6   # step interval in hours

output:
  layout: trajectories
```

Note that trajectories recipes use ``base_dates:`` (the forecast
initialisation times) rather than ``dates:``.  The latter is reserved for
gridded and tabular recipes — the distinction is enforced by the recipe
validator.

This produces steps `[6, 12, 18, …, 240]` stored on disk.

> This constraint applies to **what is built and stored on disk**. Irregular or non-uniformly spaced steps are not supported at build time; the assumption of regular spacing is baked into the `start/end/frequency` configuration and the step metadata attributes. Support for arbitrary step lists could be added later if use cases arise.
>
> At **read time**, step selection is still possible: a dataset built with steps `[6, …, 240]` can be opened with a narrower step range (see §2.1). The on-disk data has a fixed step frequency; the reader simply subsets it.

### 3.2 Step 0

Step 0 is **not supported in this ADR**. Trajectories datasets start at step ≥ 1.

Excluding step 0 is well understood and straightforward to implement. Including it is not: accumulated variables (precipitation, radiation, etc.) have no meaningful value at step 0, and the correct handling is an unsolved problem we do not want to address yet.

The natural workaround — pairing a trajectories dataset (steps 1+) with a separate gridded analysis dataset for the initial state — covers most known use cases. Step 0 support will be added in a future iteration once concrete use cases arise that cannot be served by this approach.

## 4. Naming Conventions

The existing convention (see [anemoi registry naming conventions](https://anemoi.readthedocs.io/projects/registry/en/latest/naming-conventions.html)) is:

```
purpose-content-source-resolution-start-year-end-year-frequency-version[-extra]
```

Example for a gridded dataset:

```
aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5
                                         ^^
                                         date frequency
```

For trajectories datasets the single `frequency` field is ambiguous: there are now two frequencies — the **date frequency** (how often a new forecast run starts) and the **step frequency** (the interval between steps within a run). The proposed extension is to place both frequencies consecutively:

```
purpose-content-source-resolution-start-year-end-year-date-freq-step-freq-version[-extra]
```

Example:

```
aifs-od-an-oper-0001-mars-o96-1979-2022-24h-6h-v5
                                         ^^^  ^^
                                         │    step frequency (6 h between steps)
                                         date frequency (new run every 24 h)
```

Whether to also encode the step range (`start`/`end`) in the name is an open question. Including it would make the name more informative but significantly longer. A possible convention would be to add it as part of the optional `extra` suffix (e.g. `...-steps6to240`) if needed, rather than baking it into the fixed fields. This is to be decided.