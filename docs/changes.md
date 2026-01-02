This file will be used to populate the PR description and will be deleted before merging. It is
easier to type that description in VS Code than in a browser.

In this document, the word "format" does not refer to the file format (which is Zarr), but the data
format (gridded vs tabular).

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

The result is a much cleaner and shorter code, with the caveat that error messages will be less
human-friendly. It also makes it clearer where additions to the recipe syntax must go. We will also
benefit from all the features that Pydantic offers (default values, schema generation, version management,
deprecations, documentation generation, etc.).

### Creator

This is where most of the existing code has been reorganised, and new code added, although the orinal code still exists, just moved elsewhere.

The creation of datasets is split into several tasks: `init`, `load`, `statistics`, etc.

This was implemented in the original code as different classes and various mixins. The tasks are now
implemented as methods of a `Creator` class, and two subclasses have been introduced:
`GriddedCreator` and `TabularCreator`. The `Creator` class defines a few abstract methods that are
implemented by the two subclasses. For example, the `Creator` class implements the `load` task,
calling the code of the recipe, looping over groups of dates, supporting incremental builds, and
only calls the `load_result` method of its subclasses that handle the partial result. Another
example is the collection of metadata to be stored in the Zarr store: the superclass collects any
common information, while each subclass collects format-specific metadata.

In a nutshell, subclasses must implement:

1. Gather metadata
2. Load a single result in the Zarr
3. Compute the statistics
4. Finalise the dataset

The differences between the gridded and tabular subclasses are:

1. Provide format-specific metadata
2. For gridded, the result is added to the Zarr array `data`. For tabular data, the result is
   written to a temporary file, after being sorted by dates, having the longitudes normalised and
   dates rounded to the nearest second (so that none of that is the concern of sources and filters).
3. For gridded, statistics are computed (including tendencies statistics if requested) and stored in
   the Zarr. For tabular data, nothing happens here.
4. For gridded data, nothing happens here. For tabular data, all temporary files are loaded and
   entries are deduplicated, sorted by dates, and the Zarr is loaded and the statistics are computed
   at the same time, to minimise disk access. Temporary files are then deleted. All of this happens
   in a parallel fashion, using threads within a single process.

Note that the code that computes the statistics is the same for both formats, just called at
different times. Also note that statistics can be recomputed later with `anemoi-dataset statistics`
if needed.

As a consequence, a lot of auxiliary code was not needed and was deleted. A couple of classes and
methods were renamed to better reflect their role and purpose.

### Actions

The classes that implement the actions (call a source, call a filter, `pipe`, `join`, etc.) are
unchanged. They delegate the format-specific code to a `Context`, i.e. managing input and output of
each action, ensuring that they have the right type (earthkit.data Field for gridded, Pandas frame
for tabular).

## CLI

The `anemoi-datasets create` command is unchanged. The commands that implement incremental building
of datasets, such as `anemoi-datasets create`, `anemoi-datasets load`, `anemoi-datasets statistics`
are still there, but some of them are not useful any more (e.g. statistics and tendencies statistics
are now computed in the same pass, so commands like `anemoi-datasets init-additions` are now
no-ops).

## Using datasets

When using an anemoi dataset, there are three main concepts:

- The Zarr stores themselves
- Combining stores, e.g. concat, join, etc.
- Subsetting, e.g. dates (`start`, `end`, `frequency`, ...), variables (`select`, `drop`, `rename`,
  ...), grids (`thinning`, `area`, `cutout`, ...)

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

There is a new `ChunkCaching` class that implements aggressive caching of Zarr IOs. This class
implements per-chunk LRU caching. The number of cached chunks can be expressed either as a number of
chunks to cache, or as a maximum memory footprint (default is 512MB). In write mode, chunks are only
written to disk when evicted from the cache or flushed explicitly. This allows the decoupling of
writing to Zarr from writing to disk. In read mode, the class supports read-ahead that greatly
speeds up scanning of whole arrays (e.g. when computing statistics).

Note that this class could be part of a standalone package that would contain general-purpose
Zarr-related utilities.

The code that was used to allow concurrent writing to Zarr is gone, as it was using a facility that
will not be ported to Zarr3 (?). Synchronisation/locking is now done independently, using the same
packages that Zarr2 was using.

The `thinning` subsetting method was only supporting 2D lat/lon grids (`every-nth` method). This is
not useful for tabular data. It now supports three more methods:
- `distance-based`: ensures that each selected point is not closer to the others than a minimum
  distance expressed in km.
- `grid`: selects one representative point for each cell of a grid expressed in km (may need a
  different name)
- `random`: selects points at random, using a user-provided ratio (e.g. 50%)

These new `thinning` subsetting methods are also available for fields (gridded).

## Testing

All existing tests are passing, with the exception of those which create a dataset and compare the
output to a reference dataset, for the following reasons:
- The Pydantic-based recipe has more entries saved in the dataset metadata
- The statistics being computed at the end means that the Zarr does not contain intermediate arrays
  needed for incremental computations of statistics
- Some obsolete unused information is not stored in the metadata any more (e.g. `history`)
- Some new information is now present (e.g. `format`, `dtype`)

The code that compares two datasets is completely rewritten. Instead of comparing anemoi datasets,
it now compares Zarrs directly. It also collects all differences, fatal and non-fatal, before
failing, instead of failing on the first difference.

# What is left?

A lot.

- Migrate all DOP sources to anemoi.
- Implement format-specific entries in the recipe
- Agree on a naming convention for observations-based datasets.
- Review join, concat, select, ... in the context of tabular data (in creation and usage).
- Issue meaningful error messages when combinations are impossible (e.g. combining datasets of
  different types)
- Finalise datasets subsetting and combining tests, covering both gridded and tabular data
- Add tests for tabular data
- Revisit defaults (e.g. `thinning` subsetter)
- That bullet list is not finished. Come back later.

# Opportunities

Use Zarr3. This is the opportunity to change, so we do not create Zarr2 observations datasets.

The new class structure will allow the creation of other Zarr-based datasets, in particular the one
for training downsampling models (i.e. by implementing a new `Creator` subclass).

The `ChunkCaching` class can easily be updated to handle Zarr3's sharding, and greatly simplify the
creation of datasets to use that feature.
