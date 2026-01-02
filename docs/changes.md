This file will be used to populate the PR description will be deleted before merging. It is easier to type that desciption in vscode than in a browser.

[Read about the goal here.](./adr/adr-1.md)


# Changes

## Creating datasets

Most of the changes happened here.

### Recipes

Great care has been taken to ensure that existing recipes are still valid.

#### YAML

A new entry in the YAML file describes the expected output:

```yaml
format: gridded # default
```
or
```yaml
format: tabular
```


#### Implementation

The recipe is now a Pydantic model and sub-models for each group (`dates`, `build`, `output`), with
the exception of the `input` entry, which is still freeform and managed by the "actions" code.

### Creator

This is where most of the existing code has been reorganised, and new code added. The creation of datasets is split in several tasks: `init`, `load`, `statistics`, etc.

This was implemented in the original code as different classes and various mixins. The tasks are now implemented as methods of a `Creator` class, and two subclasses have been introduced: `GriddedCreator` and `TabularCreator`. The `Creator` class will define a few abstract methods that are implemented by the two subclasses. For example, the `Creator` class implements the `load` task, calling the code of the recipe, looping over groups of dates, supporting incremental builds, and only call the `load_result` method of its subclasses that will handle the partial result. Another example is the collection of metadata to be stored in the zarr store: the superclass collect any common information, while each subclass collect format specific metadata.

### Actions

The classes that implement the actions (call a source, call a filter, `pipe`, `join`, etc) are unchanged. They delagate the format-scpecific code to a `Context`, ie managing input and output of each action, ensre that they have the right type (earthkit.data Field for gridded, Panda's frame for tabular).

## CLI

The `anemoi-datasets create` command is unchanged. The commands that implement incremental building of datasets, such a `anemoi-datasets create`, `anemoi-datasets load`, `anemoi-datasets statistics` are still there, but some of them are not usefull anymore (eg, staticts and tendencies statistics are now computed in the same pass, so commands like `anemoi-datasets init-additions` are now noops).

## Using datasets

When using an anemoi dataset, there are three main concepts:

- The Zarr stores themselves
- Combining stores, e.g. concat, join, etc.
- Subsetting, e.g. dates (`start`, `end`, `frequency`, ...), variables (`select`, `drop`, `rename`, ...), grids
    (`thinning`, `area`, `cutout`, ...)


Very few changes. Mostly refactoring:

- Code for using datasets is in "usage" (was "data")
- Code for combining and subsetting that is specific to gridded data is in a "gridded" sub-directory
- Code for combining and subsetting that is specific to tabular data is in a "tabular" sub-directory
- Code for combining and subsetting that is not specific is in "common"

Actual changes:

- The class that handles Zarr stores is now a superclass, with two subclasses: ZarrGridded and
    ZarrTabular
- Subsetting factories are now delegated to ZarrGridded and ZarrTabular, so they can create
    different objects. An example is `area`, which is handled by two different `Cropping` classes, one
    for gridded, one for tabular, while offering the same interface to the user.

## Other changes

There is a new `ChunkCaching` class that implements an aggressive caching of Zarr IOs. This class
implements per-chunk LRU caching. The number of cached chunks can be expressed either as a number of
chunks to cache, or as a maximum memory footprint (default is 512MB). In write mode, chunks are only
written to disk when evicted from the cache or flushed explicitly. This allows the decoupling of
writing to Zarr from writing to disk. In read mode, the class supports read-ahead that greatly
speeds up scanning of whole arrays (e.g. when computing statistics).

Note that this class could be part of a standalone package that would contain general-purpose Zarr
related utilities.

The code that was used to allow concurrent writing to Zarr is gone, as it was using a facility that
will not be ported to Zarr3 (?). Synchronisation/locking is now done independently, using the same
packages that Zarr2 was using.

The `thinning` subsetting method was only supporting 2D lat/lon grids (`every-nth` method). This is not useful of tabular data. It now support three more methods:
- `distance-based`: ensure that each select point are not closer to the others given a minimum distance expressed in km.
- `grid`: select one representative point for each cell of a grid expressed in km (may need a different name)
- `random`: select point at random, using a user-provided ratio (e.g. 50%)

These new `thinning` subsetting method are also available for fields (gridded).

## Testing

All existing tests are passing, with the exception of those which create a dataset and compare the
output to a reference dataset, for the following reasons:
- The Pydantic-based recipe has more entries saved in the dataset metadata
- The statistics being computed at the end means that the Zarr does not contain intermediate arrays
    needed for incremental computations of statistics
- Some obsolete unused information is not stored in the metadata anymore (e.g. `history`)
- Some new information is now present (e.g. `format`, `dtype`)

The code that compares two datasets is completely rewritten. Instead of comparing anemoi datasets, it now compares Zarrs directly. It also collects all differences, fatal and non-fatal, before failing, instead of failing on the first difference.

# What is left?

A lot.

- Migrate all DOP sources to anemoi.
- Agree on a naming convention for observations-based datasets.
- Add tests for tabular data

# Opportunities

The new class structure will allow the creation of other Zarr-based datasets, in particular the one
for training downsampling models (ie by implementing a new `Creator` subclass).

The `ChunkCaching` class can easily be updated to handle Zarr3's sharding, and greatly simplify the
creation of datasets to use that feature.
