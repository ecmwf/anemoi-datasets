from anemoi.datasets.data.dataset import Dataset
from anemoi.datasets.testing import default_test_indexing

# List of methods called during training. To update the list, run training with ANEMOI_DATASETS_TRACE=1

TRAINING_METHODS = {
    "__getitem__",
    "__len__",
    "latitudes",
    "longitudes",
    "metadata",  # Accessed when checkpointing
    "missing",
    "name_to_index",
    "shape",
    "statistics",
    "supporting_arrays",  # Accessed when checkpointing
    "variables",
}


def ignore(*args, **kwargs):
    """Ignore function to be used as a default for the verify function."""
    pass


def verify(results, dataset, name, kwargs=None, validate=ignore, optional=False):

    print(f"Verifying {name}...")

    try:
        getattr(Dataset, name)
    except AttributeError:
        results[name] = f"❌ Attribute {name} not found in Dataset class."
        return

    try:
        result = getattr(dataset, name)
        if kwargs is not None:
            result = result(**kwargs)
        if callable(result):
            raise ValueError(f"{name} is a callable method, not an attribute. Please pass kwargs.")

        print(f"...{name} result: {result}")

        validate(dataset, name, result)
        results[name] = f"✅ Dataset verification passed for {name}."
    except Exception as e:
        results[name] = f"❌ Dataset verification failed for {name}: {e}"


def verify_dataset(dataset, costly_checks=False):
    """Verify the dataset."""

    results = {}

    if costly_checks:
        # This check is expensive as it loads the entire dataset into memory
        # so we make it optional
        default_test_indexing(dataset)

        for i, x in enumerate(dataset):
            y = dataset[i]
            assert (x == y).all(), f"Dataset indexing failed at index {i}: {x} != {y}"

    verify(results, dataset, "__len__", kwargs={})
    verify(results, dataset, "__getitem__", kwargs={"index": 0})

    verify(results, dataset, "arguments")
    verify(results, dataset, "collect_input_sources")
    verify(results, dataset, "collect_supporting_arrays")
    verify(results, dataset, "computed_constant_fields")
    verify(results, dataset, "constant_fields")
    verify(results, dataset, "dataset_metadata")
    verify(results, dataset, "dates")
    verify(results, dataset, "dates_interval_to_indices")
    verify(results, dataset, "dtype")
    verify(results, dataset, "end_date")
    verify(results, dataset, "field_shape")
    verify(results, dataset, "frequency")
    verify(results, dataset, "get_dataset_names", kwargs={"names": set()})
    verify(results, dataset, "grids")
    verify(results, dataset, "label")
    verify(results, dataset, "latitudes")
    verify(results, dataset, "longitudes")
    verify(results, dataset, "metadata", kwargs={})
    verify(results, dataset, "metadata_specific", kwargs={})
    verify(results, dataset, "missing")
    verify(results, dataset, "mutate", kwargs={})
    verify(results, dataset, "name")
    verify(results, dataset, "name_to_index")
    # verify(results, dataset,'plot', kwargs={'date':0, 'variable':0})
    verify(results, dataset, "provenance", kwargs={})
    verify(results, dataset, "resolution")
    verify(results, dataset, "shape")
    verify(results, dataset, "source", kwargs={"index": 0})
    verify(results, dataset, "start_date")
    verify(results, dataset, "statistics")
    verify(results, dataset, "statistics_tendencies", kwargs={})
    verify(results, dataset, "sub_shape", kwargs={})
    verify(results, dataset, "supporting_arrays", kwargs={})
    verify(results, dataset, "swap_with_parent", kwargs={})
    verify(results, dataset, "to_index", kwargs={"date": 0, "variable": 0})
    verify(results, dataset, "tree", kwargs={})
    verify(results, dataset, "typed_variables")
    verify(results, dataset, "variables")
    verify(results, dataset, "variables_metadata")

    for k, v in results.items():
        print(f"{k}: {v}")
