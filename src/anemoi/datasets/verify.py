def verify(dataset, func):
    try:
        func(dataset)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")


def verify_dataset(dataset):
    """Verify the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to verify.

    Raises
    ------
    ValueError
        If the dataset is not valid.
    """

    verify(dataset, "name", lambda ds: ds.name)
    verify(
        dataset,
        "dates_interval_to_indices",
        lambda ds: ds.dates_interval_to_indices(start=dataset.start_date, end=dataset.end_date),
    )
    verify(dataset, "provenance", lambda ds: ds.provenance())
    verify(dataset, "sub_shape", lambda ds: ds.sub_shape(0))
    verify(dataset, "typed_variables", lambda ds: ds.typed_variables)
    verify(dataset, " metadata", lambda ds: ds.metadata())

    verify(dataset, lambda ds: ds.start_date)
    verify(dataset, lambda ds: ds.end_date)
    verify(dataset, lambda ds: ds.dataset_metadata())
    verify(dataset, lambda ds: ds.supporting_arrays())
    verify(dataset, lambda ds: ds.collect_supporting_arrays())

    verify(dataset, lambda ds: ds.metadata_specific())

    verify(dataset, lambda ds: ds.grids())
    verify(dataset, lambda ds: ds.label)
    verify(dataset, lambda ds: ds.computed_constant_fields())

    verify(dataset, lambda ds: ds.to_index())

    verify(dataset, lambda ds: ds[0])

    verify(dataset, lambda ds: len(ds))

    verify(dataset, lambda ds: ds.variables)

    verify(dataset, lambda ds: ds.frequency())
    verify(dataset, lambda ds: ds.dates)
    verify(dataset, lambda ds: ds.resolution())
    verify(dataset, lambda ds: ds.name_to_index())

    verify(dataset, lambda ds: ds.shape())

    verify(dataset, lambda ds: ds.field_shape())
    verify(dataset, lambda ds: ds.dtype())
    verify(dataset, lambda ds: ds.latitudes())
    verify(dataset, lambda ds: ds.longitudes())
    verify(dataset, lambda ds: ds.variables_metadata())
    verify(dataset, lambda ds: ds.missing())
    verify(dataset, lambda ds: ds.constant_fields())
    verify(dataset, lambda ds: ds.statistics())
    verify(dataset, lambda ds: ds.statistics_tendencies())

    verify(dataset, lambda ds: ds.sources())
    verify(dataset, lambda ds: ds.tree())
    verify(dataset, lambda ds: ds.collect_input_sources())
    verify(dataset, lambda ds: ds.get_dataset_names())
