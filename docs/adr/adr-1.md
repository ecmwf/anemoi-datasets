# Support irregular observations datasets

## Status

<!--What is the status? -->

Proposed - 30/04/2025

## Context

<!--What is the issue that we are seeing that is motivating this decision or change?-->

The objective of this change is to support observations data which is not regular.

In contrast with the fields data where each date contain the same number of points,
in the observations data, the number of points can change for every time window.

The Zarr format fits well the fields data, but does not fit the observations data.

To allow storing data with irregular shape, we need to use another format than the zarr used for fields.
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
- For observations, date-time are *rounded* to the nearest second

Example of a call to `open_datasets`:


```python
ds = open_dataset(
  path,
  start=1979,
  end=2020,
  window="(-3,+3]",
  frequency="6h")
```

The parameters `path`, `start`, `end`, and `frequency` have the same meaning as for fields. As with fields, `start` and `end` can be full date-times.

A sample `ds[i]` is defined by the start date and the frequency (i.e., the date of the sample). The `window` specifies how many observations around the sample date should be considered part of the sample.

When the user requests data that does not exist for a given window, an empty sample is returned, provided that the requested dates lie between `start` and `end`.

## Open questions

1 - What does `ds[i]` returns to the user?

Unlike fields, the sample needs to contain the actual dates and position of the observations, plus their time relative to the start of the window

```python
x = ds[i]
x.data # Returns the [N x M] data array
x.latitudes # Return the corresponding N latitudes
x.longitudes # Return the corresponding N longitudes
x.dates # Return the corresponding N dates
x.timedeltas # Return the (N) times (e.g., in seconds) of the observations relative to the end of the window
```

Note that we can implement a similar scheme for fields, if needed.

 2 - When combining similar  observations from several sources, can we normalise them using the same statistics?


## Scope of Change

<!--Specify which Anemoi packages/modules will be affected by this decision.-->
- anemoi-datasets
Not a breaking change, this only add functionality to read observations datasets.

Must be in line with the change related to multi-datasets.

## Consequences

<!--Discuss the impact of this decision, including benefits, trade-offs, and potential technical debt.-->

## Alternatives Considered [Optional]

<!--List alternative solutions and why they were not chosen.-->

## References [Optional]

<!--Links to relevant discussions, documentation, or external resources.-->
