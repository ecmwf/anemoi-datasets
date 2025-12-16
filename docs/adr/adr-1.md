# Support irregular observations datasets

## Status

<!--What is the status? -->

Proposed - 30/04/2025

## Context

<!--What is the issue that we are seeing that is motivating this decision or change?-->

The objective of this change is to support observations data which are not regular.

In contrast with the fields data where each date contains the same number of points,
in the observations data, the number of points can change for every time window.

To allow storing data with irregular shape, we need to use another format than the Zarr used for fields.
An experimental implementation using xarray-zarr has been developed and is not optimised for ML training.

## High-level principles

- Ensure that I/O operations are not the bottleneck during training.
- The way fields (gridded data) and observations (tabular data) are handled by `anemoi-datasets` should be as similar as possible.
- Allow users to control which samples are presented during training via configuration files, when possible.

## Decisions

- Observations datasets in Anemoi will use Zarr (but with a different structure than fields).
- The Zarr store will contain a single 2D array, where rows represent individual observations (e.g., different dates and locations) and columns represent the observed quantities (pressure, temperature, etc.).
- Each dataset will contain only one observation type.
- The total number of datasets required to cover all observation types should be small (tens). Similar observation types should be combined into a single dataset, padding with NaNs if needed, as long as the padding remains small.
- The Zarr store will contain additional metadata, such as statistics and a possible index to access ranges of observations (“windows”).
- Window sizes should not be prescribed at dataset-creation time but should instead be specified when using the dataset.
- Combining several types of observations (or fields and observations) will be handled by the data loader at training time.
- Information about which data sources were used during training must be carried through to inference via the checkpoint metadata.
- As with fields, the open_dataset call will allow users to specify run-time transformations on the data, such as thinning, sub-area extraction, etc. This feature will allow researchers to experiment without needing to recreate datasets.
- For observations, date-times are *rounded* to the nearest second

### Zarr array layout

#### Data

The `data` table contains one row per observation. It has four mandatory columns (`date`, `time`, `latitude`, `longitude`), followed by additional columns that are specific to each data source.

Rows are sorted in lexicographic order of the columns.

| Date       | Time     | Latitude | Longitude | Col 1  | Col 2 | ... | Col N  |
|------------|----------|----------|-----------|--------|-------|-----|--------|
| 2020-01-01 | 00:00:00 | 51.5074  | -0.1278   | 1013.2 | 7.5   | ... | 23.5   |
| 2020-01-01 | 06:00:08 | 48.8566  | 2.3522    | 1012.8 | 6.8   | ... | -4.5   |
| 2020-01-01 | 18:07:54 | 40.7128  | -74.0060  | 1014.1 | 5.2   | ... | 12.9   |
| 2020-01-01 | 23:02:01 | 35.6895  | 139.6917  | 1011.7 | 8.0   | ... | 0.0    |
| 2020-01-02 | 00:00:05 | 55.7558  | 37.6173   | 1013.5 | -2.1  | ... | -4.2   |
| ...        | ...      | ...      | ...       | ...    | ...   | ... | ...    |

The `date` and `time` columns are separated because `float32` encoding is used. Dates are encoded as days since the Unix epoch. The largest integer that can be represented exactly by a 32-bit float is 2²⁴ − 1 = 16,777,215. When interpreted as seconds, this corresponds to approximately 194 days, which is insufficient. When interpreted as days, it corresponds to roughly 46,000 years, which is sufficient.

#### Chunking

Because of the variability of the size of samples, it is not possible to select a "best" chunking. We will set the chunk sizes to match the best I/O block size (64MB ~ 256MB on Lustre). Chunks will also be cached in memory to avoid unnecessary reads.

### Index

Ranges of rows sharing the same date/time are indexed together for fast access when extracting windows of observations.

The index is a Zarr-backed [b+tree](https://en.wikipedia.org/wiki/B-tree) stored in the array `time_index`.

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

The parameters `path`, `start`, `end`, and `frequency` have the same meaning as for fields. As with fields, `start` and `end` can be full date-times. If they are not, they are internally transformed to [full dates](https://anemoi.readthedocs.io/projects/datasets/en/latest/using/subsetting.html)

#### Windows

Windows are **relative** time intervals that can be open or closed at each end. A round bracket indicates an open end, while a square bracket indicates a closed end. The default units are hours.

Examples:

```python
"[-3,+3]" # Both ends are included
"(-1d,0]" # Start is open, end is closed
```


#### Dates

Unlike for fields, where the `start` and `end` must be within the list of available dates, in the case of observations, `start` and `end` can be anything, and empty records will be returned to the user when no observations are available for a given window.

The dates of the dataset are then defined as all dates between `start` and `end` with a step of `frequency`:

```python
result = []
date = start
while date <= end:
   result.append[date]
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

A sample `ds[i]` is defined by the start date and the frequency (i.e., the date of the sample). The `window` specifies how many observations around the sample date should be considered part of the sample.

So the sample `ds[i]` will return all observations around the ith date according to the window (the example below ignores the open/close ends of the window):

```python
date = start + i * frequency
return all_records_between( date + window.start, date + window.end)
```

When the user requests data that do not exist for a given window, an empty sample is returned, provided that the requested dates lie between `start` and `end`. Otherwise, an error is raised.

#### Sample format

What does `ds[i]` return to the user? Unlike fields, the sample needs to contain the actual dates and positions of the observations, plus their time relative to the start of the window.

Several options for what `ds[i]` can be:


#### Option 1 - (Return a timedelta column - Prefered option)

_anemoi-dataset_ can compute the difference between the reference date of the sample (e.g. the "middle" of the window) and the observation date, in seconds. Example (assuming the sample date is `2020-01-02T00:00:00`)

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

For the sake of symetry, the `ds.detail()` method can be implemented fields as well.

### Statistics

(TODO)

When combining similar observations from several sources, can we normalise them using the same statistics?


### Building datasets

## Scope of Change

<!--Specify which Anemoi packages/modules will be affected by this decision.-->
- anemoi-datasets
Not a breaking change, this only adds functionality to read observations datasets.

Must be in line with the change related to multi-datasets.

## Consequences

<!--Discuss the impact of this decision, including benefits, trade-offs, and potential technical debt.-->

## Alternatives Considered [Optional]

<!--List alternative solutions and why they were not chosen.-->

## References [Optional]

<!--Links to relevant discussions, documentation, or external resources.-->
