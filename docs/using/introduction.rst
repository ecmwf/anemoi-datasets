.. _using-introduction:

##############
 Introduction
##############

.. warning::

   The code below still mentions the old name of the package,
   `ecml_tools`. This will be updated once the package is renamed to
   `anemoi-datasets`.

An *Anemoi* dataset is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models. It is
organised in such a way that I/O operations are minimised (see
:ref:`overview`).

.. _zarr: https://zarr.readthedocs.io/

To open a dataset, you can use the `open_dataset` function.

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset("path/to/dataset.zarr")

You can then access the data in the dataset using the `ds` object as if
it was a NumPy array.

.. code:: python

   print(ds.shape)

   print(len(ds))

   print(ds[0])

   print(ds[10:20])

One of the main feature of the *anemoi-datasets* package is the ability
to subset or combine datasets.

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset("path/to/dataset.zarr", start=2000, end=2020)

In that case, a dataset is created that only contains the data between
the years 2000 and 2020. Combining is done by passing multiple paths to
the `open_dataset` function:

.. code:: python

   from anemoi_datasets import open_dataset

   ds = open_dataset("path/to/dataset1.zarr", "path/to/dataset2.zarr")

In the latter case, the datasets are combined along the time dimension
or the variable dimension depending on the datasets structure.
