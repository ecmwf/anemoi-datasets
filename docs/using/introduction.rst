.. _using-introduction:

###########################
 Using an existing dataset
###########################

An *Anemoi* dataset is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models. It is
organised in such a way that I/O operations are minimised (see
:ref:`overview`).

.. _zarr: https://zarr.readthedocs.io/

To open a dataset, you can use the `open_dataset` function.

.. code:: python

   print(ds.missing)

You can then access the data in the dataset using the `ds` object as if
it were a NumPy array.

.. code:: python

   print(ds.shape)

   print(len(ds))

   print(ds[0])

   print(ds[10:20])

One of the main features of the *anemoi-datasets* package is the ability
to subset or combine datasets.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("path/to/dataset.zarr", start=2000, end=2020)

In that case, a dataset is created that only contains the data between
the years 2000 and 2020. Combining is done by passing multiple paths to
the `open_dataset` function:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("path/to/dataset1.zarr", "path/to/dataset2.zarr")

In the latter case, the datasets are combined along the time dimension
or the variable dimension depending on the dataset's structure.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   opening
   methods
   subsetting
   combining
   selecting
   ensembles
   grids
   zip
   statistics
   missing
   other
   matching
   miscellaneous
   configuration
