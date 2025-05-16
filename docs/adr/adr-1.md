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

## Decision

<!--Describe the change that you are proposing.-->

Add a functionality in anemoi-datasets to read observations datasets and provide the data as dictionary/mapper of numpy arrays.

Mimic as much as possible what is done for field datasets :

`ds = open_dataset(....)`
`ds[i]` -> provides the data for a given time window, related to a given reference date. As a dictionary-like object.
`ds.dates` -> list of reference date, `ds.dates[i]` is the reference date for the data provided in `ds[i]`

Also expose the latitudes, longitudes in a sensible way (as `ds.latitudes` and `ds.longitudes` now depend on the dates) and name_to_index and statistics and metadata, etc.

These API choices need to be made on an actual training use case.

Step to achieve this:
- Implement now a prototype format to allow developing ML training code on observation data.
- Performing extensive benchmarking with various formats (explore parquet, and other).
- As the final format is not defined yet, ensure a flexible architecture to allow switching (this will help for benchmarking).


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
