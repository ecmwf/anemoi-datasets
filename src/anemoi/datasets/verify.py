from anemoi.datasets.data.dataset import Dataset


def ignore(*args, **kwargs):
    """Ignore function to be used as a default for the verify function."""
    pass


def verify(dataset, name, kwargs=None, validate=ignore, optional=False):

    print(f"Verifying {name}...")

    try:
        getattr(Dataset, name)
    except AttributeError:
        print(f"❌ Attribute {name} not found in Dataset class.")
        return

    try:
        result = getattr(dataset, name)
        if kwargs is not None:
            result = result(**kwargs)
        if callable(result):
            raise ValueError(f"{name} is a callable method, not an attribute. Please pass kwargs.")

        print(f"...{name} result: {result}")

        validate(dataset, name, result)
        print(f"✅ Dataset verification passed for {name}.")
    except Exception as e:
        print(f"❌ Dataset verification failed for {name}: {e}")


def verify_dataset(dataset):
    """Verify the dataset."""

    verify(dataset, "__len__", kwargs={})
    verify(dataset, "__getitem__", kwargs={"index": 0})

    verify(dataset, "arguments")
    verify(dataset, "collect_input_sources")
    verify(dataset, "collect_supporting_arrays")
    verify(dataset, "computed_constant_fields")
    verify(dataset, "constant_fields")
    verify(dataset, "dataset_metadata")
    verify(dataset, "dates")
    verify(dataset, "dates_interval_to_indices")
    verify(dataset, "dtype")
    verify(dataset, "end_date")
    verify(dataset, "field_shape")
    verify(dataset, "frequency")
    verify(dataset, "get_dataset_names", kwargs={"names": set()})
    verify(dataset, "grids")
    verify(dataset, "label")
    verify(dataset, "latitudes")
    verify(dataset, "longitudes")
    verify(dataset, "metadata", kwargs={})
    verify(dataset, "metadata_specific", kwargs={})
    verify(dataset, "missing")
    verify(dataset, "mutate", kwargs={})
    verify(dataset, "name")
    verify(dataset, "name_to_index")
    # verify(dataset,'plot', kwargs={'date':0, 'variable':0})
    verify(dataset, "provenance", kwargs={})
    verify(dataset, "resolution")
    verify(dataset, "shape")
    verify(dataset, "source", kwargs={"index": 0})
    verify(dataset, "start_date")
    verify(dataset, "statistics")
    verify(dataset, "statistics_tendencies", kwargs={})
    verify(dataset, "sub_shape", kwargs={})
    verify(dataset, "supporting_arrays", kwargs={})
    verify(dataset, "swap_with_parent", kwargs={})
    verify(dataset, "to_index", kwargs={"date": 0, "variable": 0})
    verify(dataset, "tree", kwargs={})
    verify(dataset, "typed_variables")
    verify(dataset, "variables")
    verify(dataset, "variables_metadata")
