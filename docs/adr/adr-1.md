# Support irregular observations datasets

## Status

<!--What is the status? -->

Proposed - 30/04/2025

## Context

<!--What is the issue that we are seeing that is motivating this decision or change?-->

The objective of this change is to support observations data which are not regular.

In contrast with the fields data, where each date contains the same number of points, in the
observations data, the number of points can change for every time window.

To allow storing data with irregular shape, we need to use another format than the Zarr used for
fields. An experimental implementation using xarray-zarr has been developed and is not optimised for
ML training.

## High-level principles

- Ensure that I/O operations are not the bottleneck during training.
- The way fields (gridded data) and observations (tabular data) are handled by `anemoi-datasets`
  should be as similar as possible.
- Allow users to control which samples are presented during training via configuration files, when
  possible.

## Decisions

- Observations datasets in Anemoi will use Zarr (but with a different structure than fields).
- The Zarr store will contain a single 2D array, where rows represent individual observations (e.g.,
  different dates and locations) and columns represent the observed quantities (pressure,
  temperature, etc.).
- Each dataset will contain only one observation type.
- The total number of datasets required to cover all observation types should be small (tens).
  Similar observation types should be combined into a single dataset, padding with NaNs if needed,
  as long as the padding remains small.
- The Zarr store will contain additional metadata, such as statistics and a possible index to access
  ranges of observations (“windows”).
- Window sizes should not be prescribed at dataset-creation time but should instead be specified
  when using the dataset.
- Combining several types of observations (or fields and observations) will be handled by the data
  loader at training time.
- Information about which data sources were used during training must be carried through to
  inference via the checkpoint metadata.
- As with fields, the open_dataset call will allow users to specify run-time transformations on the
  data, such as thinning, sub-area extraction, etc. This feature will allow researchers to
  experiment without needing to recreate datasets.
- For observations, date-times are *rounded* to the nearest second.

### Zarr array layout

#### Data

The `data` table contains one row per observation. It has four mandatory columns (`date`, `time`,
`latitude`, `longitude`), followed by additional columns that are specific to each data source.

All longitudes will be normalised between 0 and 360.

Rows are sorted in lexicographic order of the columns.

| Date       | Time     | Latitude | Longitude | Col 1  | Col 2 | ... | Col N  |
|------------|----------|----------|-----------|--------|-------|-----|--------|
| 2020-01-01 | 00:00:00 | 51.5074  | -0.1278   | 1013.2 | 7.5   | ... | 23.5   |
| 2020-01-01 | 06:00:08 | 48.8566  | 2.3522    | 1012.8 | 6.8   | ... | -4.5   |
| 2020-01-01 | 18:07:54 | 40.7128  | -74.0060  | 1014.1 | 5.2   | ... | 12.9   |
| 2020-01-01 | 23:02:01 | 35.6895  | 139.6917  | 1011.7 | 8.0   | ... | 0.0    |
| 2020-01-02 | 00:00:05 | 55.7558  | 37.6173   | 1013.5 | -2.1  | ... | -4.2   |
| ...        | ...      | ...      | ...       | ...    | ...   | ... | ...    |

The `date` and `time` columns are separated because `float32` encoding is used. Dates are encoded as
days since the Unix epoch. The largest integer that can be represented exactly by a 32-bit float is
2²⁴ − 1 = 16,777,215. When interpreted as seconds, this corresponds to approximately 194 days, which
is insufficient. When interpreted as days, it corresponds to roughly 46,000 years, which is
sufficient.

#### Chunking

Because of the variability of the size of samples, it is not possible to select a "best" chunking.
We will set the chunk sizes to match the best I/O block size (64MB ~ 256MB on Lustre). Chunks will
also be cached in memory to avoid unnecessary reads.

### Index

Ranges of rows sharing the same date/time are indexed together for fast access when extracting
windows of observations.

Indexes are 2D arrays of integers layouts as follows:

| epoch      | start   | length |
|------------|---------|--------|
| ...        | ...     | ...    |
| 1767225600 | 123000  | 42     |
| 1767225601 | 123042  | 109    |
| 1767225605 | 123151  | 7      |
| ...        | ...     | ...    |

Where `epoch` is the date as Unix epoch in seconds, `start` is the offset of the first record for
that date in the `data` array, and `length` is the number of records sharing the same date (recall
that the `data` array is sorted by date, and that dates are rounded to the nearest second).

Two indexing methods are implemented, and others can be added via anemoi's plugin mechanism:

#### binary search (bisect)

The index is a simple 2D array with the `date`  (as integer Unix epochs) and the `start` and
`length` as described above. Date lookup are done using Python's
[bisect]((https://docs.python.org/3/library/bisect.html)) package.



#### B-tree

In that case, the index is a Zarr-backed [b+tree](https://en.wikipedia.org/wiki/B-tree). (`dates`,
`start`, `length`) tuple as are grouped into *pages*.

The btree organises its entries in a balanced tree of pages containing several keys. Leaf nodes
contain the  (`dates`, `start`, `length`) tuple, while non-leaf node contains only dates and
references to other nodes.

Leaf nodes are linked from smaller keys to larger keys, so that *ranges searched* only traverse the
tree once.

##### Tests

Both methods will search for a date is in the order of O(log<sub>2</sub>(N). For a billion dates
(~32 years), this is around 30 comparaisons, and the number of zarr chunks accessed can be of the
same order of magnitude.

The difference between binary search and b+trees is the speed of *range searches*, which is what is
required to select windows.

Tests have been performed on a 100-years index, with values every second (3,155,760,000 entries):

- With bisect, the average time to retrieve the entries for a 3h window is 366 ms (<3 per seconds)
  (394M on disk with the default Zarr compression).

- With a b+tree with pages of 256 entries, the average retrival time is ~62 milliseconds (~16
  seconds) (590M on disk with the default Zarr compression).

In both cases, chunk level caching (512 MB) has been used, and the chunk sizes were identical (64
MB).


### Using the dataset

Example of a call to `open_dataset`:

```python
ds = open_dataset(
    path,
    start=1979,
    end=2020,
    window="(-3,+3]",
    frequency="6h")
```

The parameters `path`, `start`, `end`, and `frequency` have the same meaning as for fields. As with
fields, `start` and `end` can be full date-times. If they are not, they are internally transformed
to [full dates](https://anemoi.readthedocs.io/projects/datasets/en/latest/using/subsetting.html).

#### Windows

Windows are **relative** time intervals that can be open or closed at each end. A round bracket
indicates an open end, while a square bracket indicates a closed end. The default units are hours.

Examples:

```python
"[-3,+3]" # Both ends are included
"(-1d,0]" # Start is open, end is closed
```

#### Dates

Unlike for fields, where the `start` and `end` must be within the list of available dates, in the
case of observations, `start` and `end` can be anything, and empty records will be returned to the
user when no observations are available for a given window.

The dates of the dataset are then defined as all dates between `start` and `end` with a step of
`frequency`:

```python
result = []
date = start
while date <= end:
   result.append(date)
   date += frequency
```

The pseudo-code above builds the list returned by `ds.dates`.

As a result, the number of samples is:

```python
n = (end - start) // frequency + 1
```

which is also the length of the dataset:

```python
len(ds)
```

#### Sample selection

A sample `ds[i]` is defined by the start date and the frequency (i.e., the date of the sample). The
`window` specifies how many observations around the sample date should be considered part of the
sample.

So the sample `ds[i]` will return all observations around the ith date according to the window (the
example below ignores the open/close ends of the window):

```python
date = start + i * frequency
return all_records_between(date + window.start, date + window.end)
```

When the user requests data that do not exist for a given window, an empty sample is returned,
provided that the requested dates lie between `start` and `end`. Otherwise, an error is raised.

#### Sample format

What does `ds[i]` return to the user? Unlike fields, the sample needs to contain the actual dates
and positions of the observations, plus their time relative to the start of the window.

Several options for what `ds[i]` can be:

#### Option 1 - (Return a timedelta column - Preferred option)

_anemoi-dataset_ can compute the difference between the reference date of the sample (e.g., the
"middle" of the window) and the observation date, in seconds. Example (assuming the sample date is
`2020-01-02T00:00:00`):

| Deltatime | Latitude  | Longitude  | Col 1   | Col 2 | ... | Col N  |
|-----------|-----------|------------|---------|-------|-----|--------|
| -86400    | 51.5074   | -0.1278    | 1013.2  | 7.5   | ... | 23.5   |
| -64792    | 48.8566   | 2.3522     | 1012.8  | 6.8   | ... | -4.5   |
| -21126    | 40.7128   | -74.0060   | 1014.1  | 5.2   | ... | 12.9   |
| -3479     | 35.6895   | 139.6917   | 1011.7  | 8.0   | ... | 0.0    |
| 5         | 55.7558   | 37.6173    | 1013.5  | -2.1  | ... | -4.2   |
| ...       | ...       | ...        | ...     | ...   | ... | ...    |

#### Option 2 - (Mask out the four first columns)

The sample only contains the actual data (what will be fed to the model).

| Col 1   | Col 2 | ...  | Col N  |
|---------|-------|------|--------|
| 1013.2  | 7.5   | ...  | 23.5   |
| 1012.8  | 6.8   | ...  | -4.5   |
| 1014.1  | 5.2   | ...  | 12.9   |
| 1011.7  | 8.0   | ...  | 0.0    |
| 1013.5  | -2.1  | ...  | -4.2   |
| ...     | ...   | ...  | ...    |

The other information is provided using another method:

```python
x = ds.details(i)
x.latitudes # Returns the corresponding ROWS latitudes
x.longitudes # Returns the corresponding ROWS longitudes
x.dates # Returns the corresponding ROWS dates
x.timedeltas # Returns the (ROWS) times (e.g., in seconds) of the observations relative to the end of the window
```

For the sake of symmetry, the `ds.detail()` method can be implemented for fields as well.

### Statistics

#### Global Statistics

Statistics will be calculated per-column and stored as metadata for the mean, min, max, standard
deviation and nan-count. This can either be done in a single post-processing pass or using something
similar to the current `ai-obs-experimental-data` implementation which calculates statistics on the
fly for each intermediate data chunk before then combining these in the post-processing of the
dataset.

When combining observations from separate sources (e.g. different satellite missions or different
conventional sensor types) statistics will be calculated on the full dataset and not
per-observation-type. If observation types have distinct enough distributions they should be split
into separate columns or datasets.

> **Question:**
>
> What is the best way to compute statistics for irregular observations datasets?
>
> - Compute the statistics using all records
> - Compute the statistics using the first 80% of the records (or another percentage)
> - Compute the statistics using all observations within 80% of the period covered (or another
>   percentage)

#### Tendency Statistics

As for fields, there may eventually be the requirement to provide statistics on the variability in
time of the observations. This is slightly more involved for non-stationary observations and will
involve some form of defining a common grid for which to compute departures. There is an existing
`dask.dataframe` implementation of this that could be used for inspiration (it also uses existing
filters for assigning grid indices to each row of the dataset).

### Building datasets

#### Sources

A source is instantiated with a config (dict-like) which is derived from the YAML recipe provided to
`anemoi-datasets create`.

Sources are called several times during dataset creation with a range of dates (Python `datetime`):
`start_date` and `end_date`. They must return a Pandas frame, with three compulsory columns: `date`
(datetime64), `latitude` (float) and `longitude` (float) as well as a number of data columns (with
arbitrary names). Each row in the dataframe is a different observation.

#### Filters

Filters take the output of sources or other filters (i.e. a Pandas frame) and return a modified
Pandas frame. The only requirement is to ensure that the compulsory columns (`date`, `latitude` and
`longitude`) are still present.

Filters for tabular datasets will have a similar interface to the filters for fields. They will be
class-based, take a configuration object (dict-like, derived from the YAML recipe) on instantiation,
and implement a `transform` method that takes a pandas dataframe and returns a pandas dataframe.

Filters will be selected through a registry, as is done with the existing field filters.


#### Incremental/parallel build

As for fields,  `anemoi-dataset create` will call sources and filters with several ranges of dates,
possibly in parallel. The size of the ranges can be controlled by the user in order not to exceed
available memory resources. The output of all incremental/parallel calls is then sorted using a
lexicographic order (`date`, `latitude`, `longitude`, `data1`, `data2`, ...) and stored in Zarr, the
dates of each row being rounded to the nearest second. **Duplicated rows are discarded**, and the
index is constructed.

## Scope of Change

<!--Specify which Anemoi packages/modules will be affected by this decision.-->

Not a breaking change, this only adds functionality to read observational datasets.

Must be in line with the change related to multi-datasets.

## Consequences

<!--Discuss the impact of this decision, including benefits, trade-offs, and potential technical debt.-->

## Alternatives Considered [Optional]

<!--List alternative solutions and why they were not chosen.-->

## References [Optional]

<!--Links to relevant discussions, documentation, or external resources.-->
